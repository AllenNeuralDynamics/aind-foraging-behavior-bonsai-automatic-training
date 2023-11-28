"""
Get data from the behavior master table and give suggestions
"""
# %%
import os
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
        self.df_master = self.df_master.query(
            "task == 'coupled_block_baiting'").sort_values(
            by=['subject_id', 'session'], ascending=True).reset_index()
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
            if stage == current_stage:
                session_at_current_stage += 1
            else:
                break

        return session_at_current_stage

    def add_and_evaluate_session(self, subject_id, session):
        """ Add a session to the curriculum manager and evaluate the transition """

        df_this_mouse = self.df_manager.query(f'subject_id == {subject_id}')

        # If we don't have feedback from the GUI about the actual training stage used
        if 'actual_stage' not in self.df_master:
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
        df_this = self.df_master.query(
            f'subject_id == {subject_id} and session == {session}').iloc[0]
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

        # Logging
        logger.info(f"{subject_id}, {df_this.session_date}, session {session}: " +
                    (f"STAY at {current_stage}" if decision.name == 'STAY'
                     else f"{decision.name} {current_stage} --> {next_stage_suggested.name}"))

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

    def plot_all_progress(self, if_show_fig=True):
        # Plot the training history of a mouse
        df_manager = self.df_manager

        # Define color scale - mapping stages to colors from red to green
        stage_to_numeric = {
            TrainingStage.STAGE_1.name: 'red',
            TrainingStage.STAGE_2.name: 'orange',
            TrainingStage.STAGE_3.name: 'yellow',
            TrainingStage.STAGE_FINAL.name: 'lightgreen',
            TrainingStage.GRADUATED.name: 'green'
        }

        # Preparing the scatter plot
        traces = []
        for n, subject_id in enumerate(df_manager['subject_id'].unique()):
            df_subject = df_manager[df_manager['subject_id'] == subject_id]
            # Get h2o if available
            if 'h2o' in self.df_master:
                h2o = self.df_master[
                    self.df_master['subject_id'] == subject_id]['h2o'].iloc[0]
            else:
                h2o = None

            trace = go.Scattergl(
                x=df_subject['session'],
                y=[n] * len(df_subject),
                mode='markers',
                marker=dict(
                    size=10,
                    line=dict(width=1, color='black'),
                    color=df_subject['current_stage_suggested'].map(
                        stage_to_numeric),
                    # colorbar=dict(title='Training Stage'),
                ),
                name=f'Mouse {subject_id}',
                hovertemplate=(f"<b>Subject {subject_id} ({h2o})"
                               "<br>Session %{x}"
                               "<br>%{customdata[0]}</b>"
                               "<br>foraging_eff = %{customdata[1]}"
                               "<br>finished_trials = %{customdata[2]}"
                               "<extra></extra>"),
                customdata=np.stack(
                    (df_subject.current_stage_suggested,
                     df_subject.foraging_efficiency,
                     df_subject.finished_trials), axis=-1),
                showlegend=False
            )
            traces.append(trace)

        # Create the figur  e
        fig = go.Figure(data=traces)
        fig.update_layout(
            title='Training Progress of Mice',
            xaxis_title='Session',
            yaxis_title='Mouse'
        )

        # Show the plot
        if if_show_fig:
            fig.show()
        return fig


if __name__ == "__main__":
    manager = CurriculumManager()
    manager.df_manager
    manager.update()
    manager.plot_all_progress()

# %%
