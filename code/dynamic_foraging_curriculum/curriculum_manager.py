"""
Get data from the behavior master table and give suggestions
"""
# %%
import os
import logging

import numpy as np
import pandas as pd

from dynamic_foraging_curriculum.schema.curriculum import TrainingStage, Metrics
from dynamic_foraging_curriculum.curriculums.coupled_baiting import coupled_baiting_curriculum
from dynamic_foraging_curriculum.util.aws_util import download_and_import_df, export_and_upload_df
from dynamic_foraging_curriculum.plot.manager import plot_manager_all_progress

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Directory for caching df_maseter tables
LOCAL_CACHE_ROOT = '/root/capsule/results/curriculum_manager/'
task_mapper = {'coupled_block_baiting': 'Coupled Baiting',
               'Coupled Baiting': 'Coupled Baiting'}


class CurriculumManager:
    def __init__(
            self,
            manager_name: str = 'Janelia_demo',
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
        self.manager_name = manager_name
        self.df_manager_name = f'df_manager_{manager_name}.pkl'
        self.df_manager_stats_name = f'df_manager_stats_{manager_name}.pkl'
        self.df_manager_root_on_s3 = df_manager_root_on_s3

        # --- load df_curriculum_manager and df_behavior from s3 ---
        self.df_behavior = download_and_import_df(bucket=df_behavior_on_s3['bucket'],
                                                  s3_path=df_behavior_on_s3['root'],
                                                  file_name=df_behavior_on_s3['file_name'],
                                                  local_cache_path=LOCAL_CACHE_ROOT,
                                                  )

        if self.df_behavior is None:
            logger.error('No df_behavior found, exiting...')
            return

        # --- Formatting the behavior master table ---
        # Remove multiIndex on columns, if any
        if self.df_behavior.columns.nlevels > 1:
            self.df_behavior.columns = self.df_behavior.columns.droplevel(
                level=0)
        # Remove session number with NaN, if any
        self.df_behavior = self.df_behavior.reset_index().dropna(
            subset=['session'])
        # Turn subject_id to str for maximum compatibility
        self.df_behavior['subject_id'] = self.df_behavior['subject_id'].astype(
            str)
        # TODO: do not hard code the task name
        self.df_behavior = self.df_behavior.query(
            f"task in {[key for key, value in task_mapper.items() if value == 'Coupled Baiting']}").sort_values(
            by=['subject_id', 'session'], ascending=True).reset_index()

        # --- Load curriculum manager table; if not exist, create a new one ---
        self.df_manager = download_and_import_df(bucket=df_manager_root_on_s3['bucket'],
                                                 s3_path=df_manager_root_on_s3['root'],
                                                 file_name=self.df_manager_name,
                                                 local_cache_path=LOCAL_CACHE_ROOT,
                                                 )
        if self.df_manager is None:  # Create a new table
            logger.info('No df_manager found, creating a new one...')
            self.df_manager = pd.DataFrame(columns=['subject_id', 'session_date', 'task',
                                                    'session', 'session_at_current_stage',
                                                    'curriculum_version', 'task_schema_version',
                                                    'foraging_efficiency', 'finished_trials',
                                                    'metrics', 'current_stage_suggested', 'current_stage_actual',
                                                    'decision', 'next_stage_suggested'])

    def upload_to_s3(self):
        """Upload s3"""

        df_to_upload = {self.df_manager_name: self.df_manager,
                        self.df_manager_stats_name: self.df_manager_stats}

        for file_name, df in df_to_upload.items():
            export_and_upload_df(df=df,
                                 bucket='aind-behavior-data',
                                 s3_path=self.df_manager_root_on_s3['root'],
                                 file_name=file_name,
                                 local_cache_path=LOCAL_CACHE_ROOT,
                                 )

    def _count_session_at_current_stage(self,
                                        df: pd.DataFrame,
                                        subject_id: str,
                                        current_stage: str) -> int:
        """ Count the number of sessions at the current stage (reset after rolling back) """
        session_at_current_stage = 1
        for stage in reversed(df.query(f'subject_id == "{subject_id}"')['current_stage_suggested'].to_list()):
            if stage == current_stage:
                session_at_current_stage += 1
            else:
                break

        return session_at_current_stage

    def add_and_evaluate_session(self, subject_id, session):
        """ Add a session to the curriculum manager and evaluate the transition """

        df_this_mouse = self.df_manager.query(f'subject_id == "{subject_id}"')

        # If we don't have feedback from the GUI about the actual training stage used
        if 'actual_stage' not in self.df_behavior:
            if session == 1:  # If this is the first session
                current_stage = 'STAGE_1'
            else:
                # Assuming current session uses the suggested stage from the previous session
                q_current_stage = df_this_mouse.query(
                    f"session == {session - 1}")['next_stage_suggested']

                if len(q_current_stage) > 0:
                    current_stage = q_current_stage.iloc[0]
                else:  # Catch missing session or wrong session number
                    id_last_session = df_this_mouse[df_this_mouse.session < session]
                    if len(id_last_session) > 0:
                        id_last_session = id_last_session.session.idxmax()
                        current_stage = df_this_mouse.loc[id_last_session,
                                                          'next_stage_suggested']
                        logger.warning(
                            msg=f"Cannot find subject {subject_id} session {session - 1}, "
                                f"use session {df_this_mouse.loc[id_last_session].session} instead")
                    else:
                        logger.error(
                            msg=f"Cannot find subject {subject_id} anysession < {session}")
                        return

        # Get metrics history
        df_history = self.df_behavior.query(
            f'subject_id == "{subject_id}" and session <= {session}')

        performance = {
            # Already sorted by session
            'foraging_efficiency': df_history['foraging_eff'].to_list(),
            'finished_trials': df_history['finished_trials'].to_list(),
        }

        # Count session_at_current_stage
        session_at_current_stage = self._count_session_at_current_stage(
            self.df_manager, subject_id, current_stage)

        # Evaluate
        metrics = dict(**performance,
                       session_total=session,
                       session_at_current_stage=session_at_current_stage)

        # TODO: to use the correct version of curriculum
        # Should we allow change of curriculum version during a training? maybe not...
        # But we should definitely allow different curriculum versions for different 
        decision, next_stage_suggested = coupled_baiting_curriculum.evaluate_transitions(
            current_stage=TrainingStage[current_stage],
            metrics=Metrics(**metrics))

        # Add to the manager
        df_this = self.df_behavior.query(
            f'subject_id == "{subject_id}" and session == {session}').iloc[0]
        self.df_manager.loc[len(self.df_manager)] = dict(
            subject_id=subject_id,
            session_date=df_this.session_date,
            session=session,
            task=task_mapper[df_this.task],
            curriculum_version='0.1',  # Allows changing curriculum during training
            task_schema_version='1.0',  # Allows changing task schema during training
            session_at_current_stage=session_at_current_stage,
            current_stage_suggested=current_stage,
            foraging_efficiency=df_this.foraging_eff,
            finished_trials=df_this.finished_trials,
            metrics=metrics,
            decision=decision.name,
            next_stage_suggested=next_stage_suggested.name
        )

        # Logging
        logger.info(f"{subject_id}, {df_this.session_date}, session {session}: " +
                    (f"STAY at {current_stage}" if decision.name == 'STAY'
                     else f"{decision.name} {current_stage} --> {next_stage_suggested.name}"))

    def compute_stats(self):
        """compute simple stats"""
        df_stats = self.df_manager.groupby(
            ['subject_id', 'current_stage_suggested'], sort=False
        )['session'].agg([('session_spent', 'count'),  # Number of sessions spent at this stage
                          # First entry to this stage
                          ('first_entry', 'min'),
                          # Last leave from this stage (Note that session_to_graduation = last_leave of STAGE_FINAL)
                          ('last_leave', 'max'),
                          ])

        df_stats['session_spanned'] = (
            df_stats.last_leave - df_stats.first_entry + 1)

        # Count the number of different decisions made at each stage
        df_decision = self.df_manager.groupby(
            ['subject_id', 'current_stage_suggested', 'decision'], sort=False
        )['session'].agg('count').to_frame()

        # Reorganize the table and rename the columns
        df_decision = df_decision.unstack(level='decision').fillna(0).astype(
            'Int64').droplevel(level=0, axis=1)
        df_decision.rename(
            columns={col: f'n_{col}' for col in df_decision.columns}, inplace=True)

        # Merge df_decision with df_stats
        df_stats = df_stats.merge(df_decision, how='left', on=[
            'subject_id', 'current_stage_suggested'])

        self.df_manager_stats = df_stats
        
    def plot_all_progress(self, if_show_fig=True):
        return plot_manager_all_progress(self, if_show_fig=if_show_fig)

    def update(self):
        """update each mouse's training stage"""
        session_key = ['subject_id', 'session']

        # Diff the to dataframe to find the new mice / new sessions
        df_merge = self.df_behavior.merge(self.df_manager, on=session_key,
                                          how='left', indicator=True)
        df_new_sessions_all = self.df_behavior[df_merge['_merge']
                                               == 'left_only']
        unique_subjects_to_evaluate = df_new_sessions_all['subject_id'].unique(
        )
        logger.info(
            f"Found {len(df_new_sessions_all)} new sessions from {len(unique_subjects_to_evaluate)} mice to evaluate")

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
        self.upload_to_s3()



if __name__ == "__main__":
    manager = CurriculumManager()
    manager.df_manager
    manager.update()
    manager.plot_all_progress()

# %%
