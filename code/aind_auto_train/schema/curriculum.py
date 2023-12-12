# %%
import json
import logging
import os
import numpy as np
from enum import Enum, auto
from typing import List, Dict, Generic

from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder

from aind_auto_train.schema.task import (Task, TrainingStage,
                                            taskparas_class, DynamicForagingParas,
                                            metrics_class, DynamicForagingMetrics)
from aind_auto_train.plot.curriculum import draw_diagram_rules, draw_diagram_paras

# %%



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
    
    class Config:
        validate_assignment = True        


class StageTransitions(BaseModel):
    '''Transition rules for a certain stage'''
    from_stage: TrainingStage
    transition_rules: List[TransitionRule]
    
    class Config:
        validate_assignment = True        


class BehaviorCurriculum(Generic[taskparas_class, metrics_class], BaseModel):
    ''' A parent curriculum for AIND behavioral task '''
    # Version of this **schema**, hard-coded here only.
    curriculum_schema_version: str = Field(
        "0.1", title="Curriculum schema version", const=True)

    task: Task
    # Version of the task schema (i.e., set of parameters accepted by the GUI)
    task_schema_version: str = Field(...,
                                     title="Task schema version")

    # Version of an instance of the curriculum schema (i.e., one curriculum)
    curriculum_version: str = Field(...,
                                    title="Curriculum version")
    curriculum_description: str = Field(
        "", title="Description of this curriculum")

    # !! Should be overriden by the subclass !!
    parameters: Dict[TrainingStage, taskparas_class]

    # Core automatic training stage transition logic
    curriculum: Dict[TrainingStage, StageTransitions]

    def evaluate_transitions(self,
                             current_stage: TrainingStage,
                             metrics: metrics_class  # Note the dynamical type here
                             ) -> TrainingStage:
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
    
    def get_transition_rule(self, 
                            from_stage: TrainingStage, 
                            to_stage: TrainingStage):
        ''' Get the transition rule between two stages '''
        for rule in self.curriculum[from_stage].transition_rules:
            if rule.to_stage == to_stage:
                return rule
        return None

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

    class Config:
        validate_assignment = True        


class DynamicForagingCurriculum(BehaviorCurriculum[DynamicForagingParas,
                                                   DynamicForagingMetrics]):
    # Override parameters
    parameters: Dict[TrainingStage, DynamicForagingParas]
    
    # Override metrics
    def evaluate_transitions(self,
                             current_stage: TrainingStage,
                             metrics: DynamicForagingMetrics  # Note the dynamical type here
                             ) -> TrainingStage:
        return super().evaluate_transitions(current_stage, metrics)



# ------------------ Helpers ------------------
# A hack to serialize TrainingStage in the dictionary keys
def transform_dict_with_enum_keys(obj):
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {transform_dict_with_enum_keys(k): transform_dict_with_enum_keys(v) for k, v in obj.items()}
    return obj

# %%
