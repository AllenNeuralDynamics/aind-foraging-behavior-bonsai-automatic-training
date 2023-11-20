# %%
import json
import logging
import numpy as np
from enum import Enum, auto
from typing import List, Callable, Dict

from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder

from dynamic_foraging_curriculum.schema.task import (DynamicForagingParas, TrainingStage,
                                                     ForagingTask)

logging.basicConfig(level=logging.INFO,
                    filename='curriculum.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# %%

class Metrics(BaseModel):
    ''' Key metrics for automatic training '''
    foraging_efficiency: List[float]  # Full history of foraging efficiency
    finished_trials: List[int]  # Full history of finished trials
    session_total: int
    session_at_current_stage: int


class TransitionRule(BaseModel):
    '''Individual transition rule'''
    to_stage: TrainingStage
    condition: Callable[[Metrics], bool] = Field(
        exclude=True)  # Exclude from JSON serialization
    condition_description: str = ""


class StageTransitions(BaseModel):
    '''Transition rules for a certain stage'''
    from_stage: TrainingStage
    transition_rules: List[TransitionRule]


class DynamicForagingCurriculum(BaseModel):
    ''' A full curriculum for the dynamic foraging task '''
    task: ForagingTask
    curriculum_version: str = Field("0.1", title="Curriculum version")
    # Corresponding to the GUI version
    schema_version: str = Field("0.1", title="Schema version")

    # Core autoamtic training parameter settings
    parameters: Dict[TrainingStage, DynamicForagingParas]
    # Core automatic training stage transition logic
    curriculum: Dict[TrainingStage, StageTransitions]

    def evaluate_transitions(self,
                             current_stage: TrainingStage,
                             metrics: Metrics) -> TrainingStage:
        ''' Evaluate the transition rules based on the current stage and metrics '''
        # Get transition rules for the current stage
        transition_rules = self.curriculum[current_stage].transition_rules

        # Evaluate the transition rules
        for transition in transition_rules:
            # Check if the condition is met in order
            if transition.condition(metrics):
                return transition.to_stage
        return current_stage  # By default, stay at the current stage

    def save_to_json(self, path: str = ""):
        filename = path + \
            f"/curriculum_{self.task.value}_{self.curriculum_version}.json"

        with open(filename, 'w') as f:
            f.write(self.to_json())

    def to_json(self):
        # Transform the model dict before serialization
        transformed_dict = transform_dict_with_enum_keys(
            self.dict(by_alias=True))
        return json.dumps(transformed_dict, indent=4, default=pydantic_encoder)


# ------------------ Helpers ------------------
# A hack to serialize TrainingStage in the dictionary keys
def transform_dict_with_enum_keys(obj):
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {transform_dict_with_enum_keys(k): transform_dict_with_enum_keys(v) for k, v in obj.items()}
    return obj


# %%
class AutomaticTraining:

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
    def save_progress_to_csv(filepath: str):
        progress_df.to_csv(filepath, index=False)

    # Function to load progress from a CSV
    def load_progress_from_csv(filepath: str):
        return pd.read_csv(filepath)

# %%
