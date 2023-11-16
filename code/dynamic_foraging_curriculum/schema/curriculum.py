# %%
from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder
from enum import Enum, auto
from typing import List, Callable, Dict
import json
import logging

from dynamic_foraging_curriculum.schema.task import (DynamicForagingTaskSchema, TrainingStage,
                  ForagingTask)

logging.basicConfig(level=logging.INFO,
                    filename='curriculum.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# %%
class Metrics(BaseModel):
    ''' Key metrics for automatic training '''
    foraging_efficiency: float
    finished_trials: int
    session_total: int
    session_at_current_stage: int


class TransitionRule(BaseModel):
    '''Individual transition rule'''
    to_stage: TrainingStage
    condition: str = Field(..., title="Condition defined by string")
    description: str = ""


class StageTransitions(BaseModel):
    '''transition_rules for a certain stage'''
    from_stage: TrainingStage
    transition_rules: List[TransitionRule]

# A hack to serialize TrainingStage in the dictionary keys
def transform_dict_with_enum_keys(obj):
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {transform_dict_with_enum_keys(k): transform_dict_with_enum_keys(v) for k, v in obj.items()}
    return obj

class Curriculum(BaseModel):
    ''' A full curriculum for the dynamic foraging task '''
    task: ForagingTask
    curriculum_version: str = Field(
        "0.1", title="Curriculum version", const=True)
    stage_transitions: Dict[TrainingStage, StageTransitions]

    class Config:
        json_encoders = {
            ForagingTask: lambda v: v.value,
            TrainingStage: lambda v: v.value,
        }    
        
    def evaluate_transitions(self, 
                             current_stage: TrainingStage, 
                             metrics: Metrics) -> TrainingStage:
        ''' Evaluate the transition rules based on the current stage and metrics '''
        # Get transition rules for the current stage
        transition_rules = self.stage_transitions[current_stage].transition_rules
        
        # Evaluate the transition rules
        for transition in transition_rules:
            # Check if the condition is met in order
            if eval(transition.condition, {}, {"metrics": metrics}):
                return transition.to_stage
        return current_stage # By default, stay at the current stage
    
    def save_to_json(self, filename: str = None):
        if filename is None:
            filename = f"curriculum_{self.task.value}_{self.curriculum_version}.json"
            
        with open(filename, 'w') as f:
            f.write(self.to_json())
            
    def to_json(self):
        # Transform the model dict before serialization
        transformed_dict = transform_dict_with_enum_keys(self.dict(by_alias=True))
        return json.dumps(transformed_dict, indent=4, default=pydantic_encoder)

#%%
coupled_baiting_curriculum = Curriculum(
    task=ForagingTask.C1B1,
    curriculum_version="0.1",
    
    stage_transitions={
        TrainingStage.STAGE_1: StageTransitions(
            from_stage=TrainingStage.STAGE_1,
            transition_rules=[
                TransitionRule(
                    description="Stage 1 -> Stage 2",
                    to_stage=TrainingStage.STAGE_2,
                    condition="metrics.finished_trials >= 100 and "
                              "metrics.foraging_efficiency > 0.7"
                    )
                ]
        ),

        TrainingStage.STAGE_2: StageTransitions(
            from_stage=TrainingStage.STAGE_2,
            transition_rules=[
                TransitionRule(
                    to_stage=TrainingStage.STAGE_2,
                    condition="metrics.finished_trials >= 100 and "
                              "metrics.foraging_efficiency > 0.7"
                ),
                TransitionRule(
                    to_stage=TrainingStage.STAGE_1,
                    condition="metrics.finished_trials <= 100 or "
                              "metrics.foraging_efficiency < 0.5"
                ),
            ]
        )
    }
    # Add other stage transition_rules as needed
)

#%%
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
        
        logging.info(f"Mouse {self.mouse_id} transitioned from {current_stage.name} to {next_stage}")
        logging.info(f"Mouse {self.mouse_id} stayed at {self.current_stage.name}")


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
