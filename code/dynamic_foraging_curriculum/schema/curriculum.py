# %%
import json
import logging
import os
import numpy as np
from enum import Enum, auto
from typing import List, Callable, Dict

from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder

from dynamic_foraging_curriculum.schema.task import (DynamicForagingParas, TrainingStage,
                                                     ForagingTask)
from dynamic_foraging_curriculum.plot.curriculum import draw_curriculum_diagram

# %%

class Metrics(BaseModel):
    ''' Key metrics for automatic training '''
    foraging_efficiency: List[float]  # Full history of foraging efficiency
    finished_trials: List[int]  # Full history of finished trials
    session_total: int
    session_at_current_stage: int

class Decision(Enum):
    STAY: str = "stay"
    PROGRESS: str = "progress"
    ROLLBACK: str = "rollback"

class TransitionRule(BaseModel):
    '''Individual transition rule'''
    decision: Decision
    to_stage: TrainingStage
    condition: str = ""  # A string for lambda function
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
    task_schema_version: str = Field("0.1", title="Task schema version")

    # Core autoamtic training parameter settings
    parameters: Dict[TrainingStage, DynamicForagingParas]
    # Core automatic training stage transition logic
    curriculum: Dict[TrainingStage, StageTransitions]

    def evaluate_transitions(self,
                             current_stage: TrainingStage,
                             metrics: Metrics) -> TrainingStage:
        ''' Evaluate the transition rules based on the current stage and metrics '''
        # Return if already graduated
        if current_stage == TrainingStage.GRADUATED:
            return Decision.STAY, current_stage
        
        # Get transition rules for the current stage
        transition_rules = self.curriculum[current_stage].transition_rules

        # Evaluate the transition rules
        for transition in transition_rules:
            # Check if the condition is met in order
            func = eval(transition.condition.replace("\n", ""))  # Turn the string into a lambda function 
            if func(metrics):  
                return transition.decision, transition.to_stage
        return Decision.STAY, current_stage  # By default, stay at the current stage

    def save_to_json(self, path: str = ""):
        if path == "":
            path = os.path.dirname(__file__)
        filename = path + \
            f"/curriculum_{self.task.value}_{self.curriculum_version}_{self.task_schema_version}.json"

        with open(filename, 'w') as f:
            f.write(self.to_json())

    def to_json(self):
        # Transform the model dict before serialization
        transformed_dict = transform_dict_with_enum_keys(
            self.dict(by_alias=True))
        return json.dumps(transformed_dict, indent=4, default=pydantic_encoder)

    def draw_curriculum_diagram(self):
        ''' Show the diagram of the curriculum '''
        return draw_curriculum_diagram(self)

# ------------------ Helpers ------------------
# A hack to serialize TrainingStage in the dictionary keys
def transform_dict_with_enum_keys(obj):
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {transform_dict_with_enum_keys(k): transform_dict_with_enum_keys(v) for k, v in obj.items()}
    return obj