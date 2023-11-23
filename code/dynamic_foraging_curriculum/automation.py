"""
Get data from the behavior master table and give suggestions
"""
# %%
import os
import logging

import pandas as pd

from dynamic_foraging_curriculum.schema.curriculum import TrainingStage, Metrics
from dynamic_foraging_curriculum.curriculums.coupled_baiting import coupled_baiting_curriculum

logger = logging.getLogger(__name__)


class CurriculumManager:

    def __init__(self, df_manager_path: str = None, df_master_path: str = None):
        """load df_curriculum_manager and df_master from s3"""

        # Load behavior master table
        if df_master_path is None:
            df_master_path = "/root/capsule/data/s3_foraging_all_nwb/df_sessions.pkl"
        self.df_master = pd.read_pickle(df_master_path)

        # Tweaks of the master table
        self.df_master = self.df_master.sort_values(
            by=['subject_id', 'session'], ascending=True).reset_index().iloc[:100]
        self.task_mapper = {'coupled_block_baiting': 'Coupled Baiting'}

        # Load curriculum manager table; if not exist, create a new one
        if df_manager_path is None:
            df_manager_path = "/root/capsule/code/dynamic_foraging_curriculum/df_curriculum_manager.csv"

        if os.path.exists(df_manager_path):
            self.df_manager = pd.read_csv(df_manager_path)
        else:
            self.df_manager = pd.DataFrame(columns=['subject_id', 'session_date', 'task',
                                                    'session', 'session_at_current_stage',
                                                    'curriculum_version', 'task_schema_version',
                                                    'foraging_efficiency', 'finished_trials',
                                                    'metrics', 'current_stage_suggested', 'current_stage_actual',
                                                    'decision', 'next_stage_suggested'])

    def upload_df(self):
        """Upload the df_manager_path"""

        pass

    def _count_session_at_current_stage(self,
                                        df: pd.DataFrame,
                                        subject_id: str,
                                        current_stage: str) -> int:
        """ Count the number of sessions at the current stage (reset after rolling back) """
        session_at_current_stage = 1
        for stage in reversed(df.query(f'subject_id == {subject_id}')['current_stage_suggested'].to_list()):
            if stage != current_stage:
                break
            else:
                session_at_current_stage += 1

        return session_at_current_stage

    def add_and_evaluate_session(self, subject_id, session):
        """ Add a session to the curriculum manager and evaluate the transition """

        df_this = self.df_master.query(
            f'subject_id == {subject_id} and session == {session}').iloc[0]

        # If we don't have feedback from the GUI about the actual training stage used
        if 'actual_stage' not in self.df_master:
            if session == 1:  # If this is the first session
                current_stage = 'STAGE_1'
            else:
                # Assuming current session uses the suggested stage from the previous session
                current_stage = self.df_manager.query(
                    f"subject_id == {subject_id} and session == {session - 1}")[
                        'next_stage_suggested'].values[0]

        # Get metrics history
        df_history = self.df_master.query(
            f"subject_id == {subject_id} and session <= {session}")

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
        decision, next_stage_suggested = coupled_baiting_curriculum.evaluate_transitions(
            current_stage=TrainingStage[current_stage],
            metrics=Metrics(**metrics))

        # Add to the manager
        self.df_manager.loc[len(self.df_manager)] = dict(
            subject_id=subject_id,
            session_date=df_this.session_date,
            session=session,
            task=self.task_mapper[df_this.task],
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

    def update(self):
        """update each mouse's training stage"""
        session_key = ['subject_id', 'session']

        # Diff the to dataframe to find the new mice / new sessions
        df_merge = self.df_master.merge(self.df_manager, on=session_key,
                                        how='left', indicator=True)
        df_new_sessions_all = self.df_master[df_merge['_merge'] == 'left_only']
        unique_subjects_to_evaluate = df_new_sessions_all['subject_id'].unique(
        )
        logger.info(
            f"Found {len(df_new_sessions_all)} new sessions from {len(unique_subjects_to_evaluate)} mice to evaluate")

        # Loop over all mouse
        for subject_id in unique_subjects_to_evaluate:
            df_new_sessions = df_new_sessions_all.query(
                f'subject_id == {subject_id}')
            # Loop over all new sessions
            for session in df_new_sessions['session']:
                self.add_and_evaluate_session(subject_id, session)

            pass

    def plot(self):
        # Plot the training history of a mouse

        pass


if __name__ == "__main__":
    manager = CurriculumManager()
    manager.df_manager
    manager.update()

# %%
