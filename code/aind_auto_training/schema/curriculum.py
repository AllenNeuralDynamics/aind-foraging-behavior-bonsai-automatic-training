# %%
import json
import logging
import os
import numpy as np
from enum import Enum, auto
from typing import List, Callable, Dict

from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder

from aind_auto_training.schema.task import (DynamicForagingParas, TrainingStage,
                                                     ForagingTask)
from aind_auto_training.plot.curriculum import draw_diagram_rules, draw_diagram_paras

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
            # Turn the string into a lambda function
            func = eval(transition.condition.replace("\n", ""))
            if func(metrics):
                return transition.decision, transition.to_stage
        return Decision.STAY, current_stage  # By default, stay at the current stage

    def _get_export_file_name(self, path: str = ""):
        if path == "":
            path = os.path.dirname(__file__)
        return path + \
            f"/curriculum_{self.task.value}_{self.curriculum_version}_{self.task_schema_version}"

    def save_to_json(self, path: str = ""):
        with open(self._get_export_file_name(path) + '.json', 'w') as f:
            f.write(self.to_json())

    def to_json(self):
        # Transform the model dict before serialization
        transformed_dict = transform_dict_with_enum_keys(
            self.dict(by_alias=True))
        return json.dumps(transformed_dict, indent=4, default=pydantic_encoder)

    def diagram_rules(self,
                      render_file_format='svg',
                      path=''):
        ''' Show the diagram of the curriculum '''
        dot_rules = draw_diagram_rules(self)
        if render_file_format != '':
            dot_rules.render(self._get_export_file_name(path) + '_rules', 
                             format=render_file_format)
        return dot_rules

    def diagram_paras(self,
                      min_value_width=1,
                      min_var_name_width=2,
                      fontsize=12,
                      render_file_format='svg',
                      path='',
                      ):
        ''' Show the table for all parameters in all stages'''
        dot_paras = draw_diagram_paras(self,
                                       min_value_width=min_value_width,
                                       min_var_name_width=min_var_name_width,
                                       fontsize=fontsize
                                       )
        if render_file_format != '':
            dot_paras.render(self._get_export_file_name(path) + '_paras', 
                             format=render_file_format)

        return dot_paras


# ------------------ Helpers ------------------
# A hack to serialize TrainingStage in the dictionary keys
def transform_dict_with_enum_keys(obj):
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {transform_dict_with_enum_keys(k): transform_dict_with_enum_keys(v) for k, v in obj.items()}
    return obj

# %%
