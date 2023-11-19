"""
Get data from the behavior master table and give suggestions
"""
# %%
import logging

import pandas as pd

from dynamic_foraging_curriculum.schema.curriculum import TrainingStage, Metrics
from dynamic_foraging_curriculum.curriculums.coupled_baiting import coupled_baiting_curriculum

logger = logging.getLogger(__name__)

# %%
class CurriculumManager:

    def __init__(self, df_manager_path: str = None, df_master_path: str = None):
        """load df_curriculum_manager and df_master from s3"""
        pass

    def upload_df(self):
        """Upload the df_manager_path"""

        pass

    def update(self):
        """update each mouse's training stage"""
        # Loop over all mouse


def update(self):
    ''' Update all mice in the manager'''
    for mouse_id, mouse_tracker in self.mice.items():
        mouse_tracker.evaluate_transitions()

# Function to update progress for a mouse


def update_progress(mouse_id: str, session_count: int, metrics: Metrics):
    # Update the tracker's metrics
    manager.mice[mouse_id].metrics = metrics
    manager.update()

    logging.info(
        f"Mouse {self.mouse_id} transitioned from {current_stage.name} to {next_stage}")
    logging.info(
        f"Mouse {self.mouse_id} stayed at {self.current_stage.name}")

    # Get the current stage
    current_stage = manager.mice[mouse_id].current_stage

    # Log the progress in the DataFrame
    today = datetime.now().strftime('%Y-%m-%d')
    progress_df.loc[len(progress_df)] = [today, mouse_id,
                                         session_count, current_stage.name]

# Function to save progress to a CSV

# %%

df_curriculum_manager = pd.DataFrame(columns=['subject_id', 'session_date', 'task',
                                              'session_total', 'session_at_current_stage',
                                              'curriculum_version', 'task_schema_version',
                                              'foraging_efficiency', 'finished_trials',
                                              'metrics', 'current_stage', 'decision', 'next_stage'])

foraging_efficiencies = [0.5, 0.6, 0.5, 0.6, 0.6,
                         0.6, 0.7, 0.75, 0.75, 0.8, 0.8, 0.8, 0.8]
finished_trials = [500, 500, 500, 500, 500,
                   500, 500, 500, 500, 200, 200, 1000, 1000]


# TODO: define a class called CurriculumManager that interact with the database (local DataFrame or on S3)

# %%
def _count_session_at_current_stage(df: pd.DataFrame, 
                                    subject_id: str, 
                                    current_stage: str) -> int:
    """ Count the number of sessions at the current stage (reset after rolling back) """
    session_at_current_stage = 1
    for stage in reversed(df.query(f'subject_id == "{subject_id}"')['current_stage']):
        if stage != current_stage:
            break
        else:
            session_at_current_stage += 1
    
    return session_at_current_stage

def add_and_evaluate_session(df,
                             session_meta: dict,  # Meta data
                             current_stage: str,
                             # Only performance of the new session dict(foraging_efficiency: float, finished_trials: int)
                             current_metric: dict,
                             ) -> pd.DataFrame:
    """ Add a session to the curriculum manager and evaluate the transition """
    # Count session_at_current_stage
    session_at_current_stage = _count_session_at_current_stage(df, session_meta['subject_id'], current_stage)

    # Append new performance to metrics
    previous_metrics = df[(df['subject_id'] == session_meta['subject_id']) & (
        df['session_total'] == session_meta['session_total'] - 1)]['metrics'].values[0] if session_meta['session_total'] > 1 else dict(foraging_efficiency=[], finished_trials=[])

    metrics = dict(foraging_efficiency=previous_metrics['foraging_efficiency'] + [current_metric['foraging_efficiency']],
                   finished_trials=previous_metrics['finished_trials'] + [
                       current_metric['finished_trials']],
                   )

    # Evaluate
    # TODO: to use the correct version of curriculum
    decision, next_stage = coupled_baiting_curriculum.evaluate_transitions(current_stage=TrainingStage[current_stage],
                                                                           metrics=Metrics(**metrics,
                                                                                           session_total=session,
                                                                                           session_at_current_stage=session_at_current_stage))

    # Add to the manager
    df.loc[len(df)] = dict(**session_meta,
                           session_at_current_stage=session_at_current_stage,
                           current_stage=current_stage,
                           foraging_efficiency=current_metric['foraging_efficiency'],
                           finished_trials=current_metric['finished_trials'],
                           metrics=metrics,
                           decision=decision.name,
                           next_stage=next_stage.name
                           )

    return df


# Simulate trainings
for session, (foraging_efficiency, finished_trial) in enumerate(zip(foraging_efficiencies, finished_trials)):

    df_curriculum_manager = add_and_evaluate_session(df_curriculum_manager,
                                                     session_meta=dict(
                                                         subject_id='test_01',
                                                         session_date='2023-11-17',
                                                         session_total=session + 1,
                                                         task='Coupled Baiting',
                                                         curriculum_version='0.1',  # Allows changing curriculum during training
                                                         task_schema_version='1.0',  # Allows changing task schema during training
                                                     ),
                                                     current_stage='STAGE_1'
                                                                    if session == 0
                                                                    else df_curriculum_manager[df_curriculum_manager['session_total'] == session]['next_stage'].values[0],
                                                     current_metric={'foraging_efficiency': foraging_efficiency,
                                                                     'finished_trials': finished_trial,
                                                                     }
                                                     )

print(df_curriculum_manager)
