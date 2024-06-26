"""
Get data from the behavior master table and give suggestions
"""
# %%
import logging

import pandas as pd

from aind_auto_train.schema.curriculum import TrainingStage
from aind_auto_train.schema.task import metrics_class, Metrics, DynamicForagingMetrics
from aind_auto_train.util.aws_util import import_df_from_s3, export_df_to_s3
from aind_auto_train.plot.manager import plot_manager_all_progress
from aind_auto_train.curriculum_manager import CurriculumManager

logger = logging.getLogger(__name__)

# Directory for caching df_maseter table
# Map from task name in df_behavior to curriculum name (for compatibility)
task_mapper = {'coupled_block_baiting': 'Coupled Baiting',
               'Coupled Baiting': 'Coupled Baiting',
               'Uncoupled Without Baiting': 'Uncoupled Without Baiting',
               'Coupled Without Baiting': 'Coupled Without Baiting',
               'Uncoupled Baiting': 'Uncoupled Baiting'}


class AutoTrainManager:
    """This is an abstract class for auto training manager
    User should inherit this class and override the following methods:
       download_from_database()
       upload_to_database()
    """

    # Specify the Metrics subclass for a specific task
    _metrics_model: metrics_class

    def __init__(self,
                 manager_name: str,
                 if_rerun_all: bool = False,
                 ):
        """
        manager_name: str
            Name of the manager, will be used for dababase
        """
        self.manager_name = manager_name
        self.df_behavior, self.df_manager = self.download_from_database()

        # Check if all required metrics exist in df_behavior
        self.task_specific_metrics_keys = set(self._metrics_model.model_json_schema()['properties'].keys()) \
            - set(Metrics.model_json_schema()['properties'].keys())
        assert all([col in self.df_behavior.columns for col in
                    list(self.task_specific_metrics_keys)]), "Not all required metrics exist in df_behavior!"

        # If `current_stage_actual` is not in df_behavior, we are in open loop simulation mode
        if 'current_stage_actual' not in self.df_behavior:
            self.if_simulation_mode = True
            logger.warning(
                "current_stage_actual is not in df_behavior, we are in simulation mode!")
        else:
            self.if_simulation_mode = False

        if if_rerun_all:
            self.df_manager = None  # Rerun for debugging purpose

        # Create a new table if df_manager is empty
        if self.df_manager is None:
            logger.warning('No df_manager found, creating a new one...')
            self.df_manager = pd.DataFrame(columns=['subject_id', 'session_date', 'task',
                                                    'session', 'session_at_current_stage',
                                                    'curriculum_name', 'task_schema_version', 'curriculum_version', 'curriculum_schema_version',
                                                    'curriculum_json_name',
                                                    *self.task_specific_metrics_keys,
                                                    'metrics', 'current_stage_suggested', 'current_stage_actual',
                                                    'if_closed_loop', 'if_overriden_by_trainer',
                                                    'decision', 'next_stage_suggested'])

        # Initialize CurriculumManager
        # Use default s3 path to saved_curriculums
        self.curriculum_manager = CurriculumManager()

    def download_from_database(self) -> (pd.DataFrame, pd.DataFrame):
        """The user must override this method! 
        This function must return two dataframes, df_behavior and df_manager
        """
        raise Exception(
            'download_from_database() of AutoTrainManager must be overridden!')

    def upload_to_database(self):
        """The user must override this method!
        This function must somehow upload df_manager to the database
        so that download_from_database() can retrieve it.
        """
        raise Exception(
            'upload_to_database() of AutoTrainManager must be overridden!')

    def _count_session_at_current_stage(self,
                                        df: pd.DataFrame,
                                        subject_id: str,
                                        current_stage: str) -> int:
        """ Count the number of sessions at the current stage (reset after rolling back) """
        session_at_current_stage = 1
        for stage in reversed(df.query(f'subject_id == "{subject_id}"')['current_stage_actual'].to_list()):
            if stage == current_stage:
                session_at_current_stage += 1
            else:
                break

        return session_at_current_stage

    def compute_stats(self):
        """compute simple stats"""
        df_stats = self.df_manager.groupby(
            ['subject_id', 'current_stage_actual'], sort=False
        )['session'].agg([('session_spent', 'count'),  # Number of sessions spent at this stage
                          # First entry to this stage
                          ('first_entry', 'min'),
                          # Last leave from this stage
                          ('last_leave', 'max'),
                          ])

        df_stats['session_spanned'] = (
            df_stats.last_leave - df_stats.first_entry + 1)

        # Count the number of different decisions made at each stage
        df_decision = self.df_manager.groupby(
            ['subject_id', 'current_stage_actual', 'decision'], sort=False
        )['session'].agg('count').to_frame()

        # Reorganize the table and rename the columns
        df_decision = df_decision.unstack(level='decision').fillna(0).astype(
            'Int64').droplevel(level=0, axis=1)
        df_decision.rename(
            columns={col: f'n_{col}' for col in df_decision.columns}, inplace=True)

        # Merge df_decision with df_stats
        df_stats = df_stats.merge(df_decision, how='left', on=[
            'subject_id', 'current_stage_actual'])

        self.df_manager_stats = df_stats

    def plot_all_progress(
        self,
        x_axis: ['session', 'date', 'relative_date'] = 'session',
        sort_by: ['subject_id', 'first_date',
                  'last_date', 'progress_to_graduated'] = 'subject_id',
        sort_order: ['ascending', 'descending'] = 'descending',
        marker_size=10,
        marker_edge_width=2,
        highlight_subjects=[],
        if_show_fig=True
    ):
        return plot_manager_all_progress(manager=self,
                                         x_axis=x_axis,
                                         sort_by=sort_by,
                                         sort_order=sort_order,
                                         marker_size=marker_size,
                                         marker_edge_width=marker_edge_width,
                                         highlight_subjects=highlight_subjects,
                                         if_show_fig=if_show_fig)

    def _get_next_stage_suggested_on_last_session(self, subject_id, session) -> str:
        df_this_mouse = self.df_manager.query(f'subject_id == "{subject_id}"')

        q_current_stage = df_this_mouse.query(
            f"session == {session - 1}")['next_stage_suggested']

        if len(q_current_stage) > 0:
            return q_current_stage.iloc[0]

        # Catch missing session or wrong session number
        id_last_session = df_this_mouse[df_this_mouse.session < session]
        if len(id_last_session) > 0:
            id_last_session = id_last_session.session.astype(int).idxmax()
            logger.warning(
                msg=f"Cannot find subject {subject_id} session {session - 1}, "
                    f"use session {df_this_mouse.loc[id_last_session].session} instead")

            return df_this_mouse.loc[id_last_session, 'next_stage_suggested']

        # Else, (i.e., if session > 1 and no previous session found),
        # assuming "next_stage_suggested" of the last session = "actual_stage" of this session
        # i.e., no override from the trainer
        logger.warning(
            msg=f"subject {subject_id} did not start AutoTrain from session 1.")
        return self.df_behavior.query(
            f'subject_id == "{subject_id}" and session == {session}'
        ).iloc[0]['current_stage_actual']

    def _get_current_stages(self, subject_id, session) -> dict:
        # Hardcode first suggested stage here. Should be extract from the first stage of a curriculum.
        current_stage_suggested = 'STAGE_1_WARMUP' if session == 1 \
            else self._get_next_stage_suggested_on_last_session(subject_id, session)

        if self.if_simulation_mode:
            # Assuming current session uses the suggested stage from the previous session
            return {'current_stage_suggested': current_stage_suggested,
                    'current_stage_actual': current_stage_suggested,
                    'if_closed_loop': False}

        # If not in simulation mode, use the actual stage
        current_stage_actual = self.df_behavior.query(
            f'subject_id == "{subject_id}" and session == {session}'
        )['current_stage_actual'].iloc[0]

        # If current_stage_actual not in TrainingStage (including None), then we are in open loop for this specific session
        if current_stage_actual not in TrainingStage.__members__:
            logger.warning(
                f'current_stage_actual "{current_stage_actual}" is invalid for subject {subject_id}, session {session}, we are in open loop for this session.'
            )
            return {'current_stage_suggested': current_stage_suggested,
                    'current_stage_actual': current_stage_suggested,
                    'if_closed_loop': False}

        # Throw a warning if the fist actual stage is not STAGE_1 (but still use current_stage_actual)
        if session == 1 and 'STAGE_1' not in current_stage_actual:
            logger.warning(
                f'First stage is not STAGE_1 for subject {subject_id}!')

        return {'current_stage_suggested': current_stage_suggested,
                'current_stage_actual': current_stage_actual,
                'if_closed_loop': True}

    def _get_curriculum_to_use(self, df_this):
        dict_this = df_this.to_dict(orient='records')[0]
        if 'curriculum_version' in dict_this:
            return self.curriculum_manager.get_curriculum(
                # Note the distinguish between 'curriculum_name' and 'task'
                curriculum_name=dict_this['curriculum_name'],
                curriculum_version=dict_this['curriculum_version'],
                curriculum_schema_version=dict_this['curriculum_schema_version'],
            )
        else:  # Use default curriculum (for simulation)
            logger.warning(
                msg=f'No curriculum_version specified in df_behavior, use default curriculum '
                f'"Coupled Baiting_curriculum_v0.1_schema_v0.2"')
            return self.curriculum_manager.get_curriculum(
                curriculum_name='Coupled Baiting',
                curriculum_version='1.0',
                curriculum_schema_version='1.0',
            )

    def add_and_evaluate_session(self, subject_id, session):
        """ Add a session to the curriculum manager and evaluate the transition """
        # Get current stages
        _current_stages = self._get_current_stages(subject_id, session)
        current_stage_suggested = _current_stages['current_stage_suggested']
        current_stage_actual = _current_stages['current_stage_actual']
        if_closed_loop = _current_stages['if_closed_loop']

        # Skip if current_stage_suggested is not defined
        if current_stage_suggested is None:
            logger.error(f'Skipping this session...')
            return

        # Get metrics history (sorted by session)
        df_history = self.df_behavior.query(
            f'subject_id == "{subject_id}" and session <= {session}'
        ).sort_values(by=['session'], ascending=True)

        # Task-specific metrics
        task_specific_metrics = {
            perf_key: df_history[perf_key].to_list()
            for perf_key in self.task_specific_metrics_keys
        }

        # Count session_at_current_stage
        session_at_current_stage = self._count_session_at_current_stage(
            self.df_manager, subject_id, current_stage_actual)

        # Evaluate
        metrics = dict(**task_specific_metrics,
                       session_total=session,
                       session_at_current_stage=session_at_current_stage)

        # Get the curriculum to use
        df_this = self.df_behavior.query(
            f'subject_id == "{subject_id}" and session == {session}')        
        _curr = self._get_curriculum_to_use(df_this)

        if _curr is None:  # If no curriculum is found
            logger.error(
                f'No update on {subject_id}, session {session} due to missing curriculum.json')
            return

        curriculum_to_use = _curr['curriculum']
        curriculum_json = _curr['curriculum_json_name']
        metrics_to_use = _curr['metrics']

        # Evaluate the transition with the desired curriculum
        decision, next_stage_suggested = curriculum_to_use.evaluate_transitions(
            current_stage=TrainingStage[current_stage_actual],
            metrics=metrics_to_use(**metrics))

        # Add to the manager
        df_this = self.df_behavior.query(
            f'subject_id == "{subject_id}" and session == {session}').iloc[0]
        self.df_manager = pd.concat(
            [self.df_manager,
             pd.DataFrame.from_records([dict(
                 subject_id=subject_id,
                 session_date=df_this.session_date,
                 session=session,
                 task=task_mapper[df_this.task],
                 curriculum_name=curriculum_to_use.curriculum_name,
                 curriculum_schema_version=curriculum_to_use.curriculum_schema_version,
                 curriculum_version=curriculum_to_use.curriculum_version,
                 curriculum_json_name=curriculum_json,
                 session_at_current_stage=session_at_current_stage,
                 current_stage_suggested=current_stage_suggested,
                 # Note this could be from simulation or invalid feedback-induced open loop session
                 current_stage_actual=current_stage_actual,
                 if_closed_loop=if_closed_loop,
                 if_overriden_by_trainer=current_stage_actual != current_stage_suggested if if_closed_loop else False,

                 # Copy task-specific metrics
                 **{key: df_this[key] for key in self.task_specific_metrics_keys},
                 metrics=metrics,
                 decision=decision.name,
                 next_stage_suggested=next_stage_suggested.name
             )])
             ], ignore_index=True)

        # Logging
        logger.info(f"{subject_id}, {df_this.session_date}, session {session}: " +
                    (f"STAY at {current_stage_actual}" if decision.name == 'STAY'
                     else f"{decision.name} {current_stage_actual} --> {next_stage_suggested.name}"))

    def update(self):
        """update each mouse's training stage"""
        
        # Update df_behavior
        self.df_behavior, _ = self.download_from_database()
        
        session_key = ['subject_id', 'session']

        # Diff the to dataframe to find the new mice / new sessions
        df_merge = self.df_behavior.merge(self.df_manager, on=session_key,
                                          how='left', indicator=True)
        df_new_sessions_all = self.df_behavior[df_merge['_merge']
                                               == 'left_only']
        unique_subjects_to_evaluate = df_new_sessions_all[
            'subject_id'].unique()
        logger.info(
            f"Found {len(df_new_sessions_all)} new sessions from "
            f"{len(unique_subjects_to_evaluate)} mice to evaluate")

        # Loop over all mouse
        for subject_id in unique_subjects_to_evaluate:
            df_new_sessions = df_new_sessions_all.query(
                f'subject_id == "{subject_id}"')
            # Loop over all new sessions
            for session in df_new_sessions['session']:
                self.add_and_evaluate_session(subject_id, session)

        # Compute stats
        self.compute_stats()

        # Save to local cache folder
        self.upload_to_database()


class DynamicForagingAutoTrainManager(AutoTrainManager):

    _metrics_model = DynamicForagingMetrics  # Override the metrics model

    def __init__(
            self,
            manager_name: str = 'Janelia_demo',
            if_rerun_all: bool = False,
            df_behavior_on_s3: dict = dict(bucket='aind-behavior-data',
                                           root='Han/ephys/report/all_sessions/export_all_nwb/',
                                           file_name='df_sessions.pkl'),
            df_manager_root_on_s3: dict = dict(bucket='aind-behavior-data',
                                               root='foraging_auto_training/'),
    ):
        """
        manager_name: str
            Name of the manager, will be used for dababase
        df_behavior_on_s3: dict
            Full path to the behavior master table.
        df_manager_root_on_s3: dict
            Root path to the manager table.
        """

        # --- define database names ---
        self.df_manager_name = f'df_manager_{manager_name}.pkl'
        self.df_manager_stats_name = f'df_manager_stats_{manager_name}.pkl'
        self.df_manager_root_on_s3 = df_manager_root_on_s3
        self.df_behavior_on_s3 = df_behavior_on_s3

        super().__init__(manager_name=manager_name, if_rerun_all=if_rerun_all)

    def download_from_database(self):
        # --- load df_auto_train_manager and df_behavior from s3 ---
        df_behavior = import_df_from_s3(bucket=self.df_behavior_on_s3['bucket'],
                                        s3_path=self.df_behavior_on_s3['root'],
                                        file_name=self.df_behavior_on_s3['file_name'],
                                        )

        if df_behavior is None:
            logger.error('No df_behavior found, exiting...')
            return

        # --- Formatting the behavior master table ---
        # Remove multiIndex on columns, if any
        if df_behavior.columns.nlevels > 1:
            df_behavior.columns = df_behavior.columns.droplevel(
                level=0)
        # Remove session number with NaN, if any
        df_behavior = df_behavior.reset_index().dropna(subset=['session'])

        # Turn subject_id to str for maximum compatibility
        df_behavior['subject_id'] = df_behavior['subject_id'].astype(str)

        # Add curriculum_name if not present (backward compatibility)
        if 'curriculum_name' not in df_behavior.columns:
            df_behavior['curriculum_name'] = df_behavior['task']

        df_behavior = df_behavior[df_behavior.curriculum_name.notna()].sort_values(
            by=['subject_id', 'session'], ascending=True).reset_index()

        # Rename columns to the same as in DynamicForagingMetrics
        df_behavior.rename(
            columns={'foraging_eff': 'foraging_efficiency'}, inplace=True)

        # --- Load curriculum manager table; if not exist, create a new one ---
        df_manager = import_df_from_s3(bucket=self.df_manager_root_on_s3['bucket'],
                                       s3_path=self.df_manager_root_on_s3['root'],
                                       file_name=self.df_manager_name,
                                       )

        return df_behavior, df_manager

    def upload_to_database(self):
        """Upload s3"""

        df_to_upload = {self.df_manager_name: self.df_manager,
                        self.df_manager_stats_name: self.df_manager_stats}

        for file_name, df in df_to_upload.items():
            export_df_to_s3(df=df,
                            bucket='aind-behavior-data',
                            s3_path=self.df_manager_root_on_s3['root'],
                            file_name=file_name,
                            )


if __name__ == "__main__":
    manager = DynamicForagingAutoTrainManager(manager_name='447_demo',
                                              df_behavior_on_s3=dict(bucket='aind-behavior-data',
                                                                     root='foraging_nwb_bonsai_processed/',
                                                                     file_name='df_sessions.pkl'),
                                              df_manager_root_on_s3=dict(bucket='aind-behavior-data',
                                                                         root='foraging_auto_training/'),
                                              if_rerun_all=False
                                              )
    manager.update()
    manager.plot_all_progress()
    print(manager.df_manager)

# %%
