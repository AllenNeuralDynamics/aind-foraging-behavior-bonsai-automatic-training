# %%
import json
import os
import logging
import numpy as np
from enum import Enum
from typing import List, Dict, Generic, Literal

from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder

from aind_auto_train.schema.task import (Task, TrainingStage,
                                         taskparas_class, DynamicForagingParas,
                                         metrics_class, DynamicForagingMetrics)
from aind_auto_train.plot.curriculum import draw_diagram_rules, draw_diagram_paras

# %%
logger = logging.getLogger(__name__)

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


class Curriculum(BaseModel, Generic[taskparas_class, metrics_class]):
    ''' A parent curriculum for AIND behavioral task 
    When adding a new curriculum, please inherit from this class and 
    specify the {taskparas_class} and {metrics_class} in the generic type
    '''
    # Version of this **schema**, hard-coded here only.
    curriculum_schema_version: Literal["0.1"] = Field(
        "0.1", title="Curriculum schema version")

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

    def _get_export_model_name(self):
        return (f"/{self.task}_v{self.task_schema_version}"
                f"_curriculum_v{self.curriculum_version}"
                f"_schema_v{self.curriculum_schema_version}")

    def _get_export_schema_name(self):
        return (f"/{self.__class__.__name__}_v{self.curriculum_schema_version}")

    def save_to_json(self, path: str = ""):
        path = path or os.path.dirname(__file__)
        f_name_model = path + self._get_export_model_name() + '.json'
        # Dump the model
        with open(f_name_model, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Curriculum saved to {f_name_model}")
        
        # Dump the schema as well
        f_name_schema = path + self._get_export_schema_name() + '.json'
        with open(f_name_schema, 'w') as f:
            f.write(self.schema_json(indent=4))
        logger.info(f"Curriculum schema saved to {f_name_schema}")

    def to_json(self):
        # Transform the model dict before serialization
        transformed_dict = transform_dict_with_enum_keys(
            self.dict(by_alias=True))
        
        # Add the schema name (only) when exporting to json
        transformed_dict = {'curriculum_schema_name':self.__class__.__name__,
                            **transformed_dict}
        
        return json.dumps(transformed_dict, 
                          indent=4, 
                          default=pydantic_encoder)

    def diagram_rules(self,
                      render_file_format='svg',
                      path=''):
        ''' Show the diagram of the curriculum '''
        path = path or os.path.dirname(__file__)
        dot_rules = draw_diagram_rules(self)
        
        if render_file_format != '':
            f_name = path + self._get_export_model_name() + '_rules'
            dot_rules.render(f_name,
                             format=render_file_format)
            logger.info(f"Curriculum rules diagram saved to {f_name}")
        return dot_rules

    def diagram_paras(self,
                      min_value_width=1,
                      min_var_name_width=2,
                      fontsize=12,
                      render_file_format='svg',
                      path='',
                      ):
        ''' Show the table for all parameters in all stages'''
        path = path or os.path.dirname(__file__)
        dot_paras = draw_diagram_paras(self,
                                       min_value_width=min_value_width,
                                       min_var_name_width=min_var_name_width,
                                       fontsize=fontsize
                                       )
        if render_file_format != '':
            f_name = path + self._get_export_model_name() + '_paras'
            dot_paras.render(f_name,
                             format=render_file_format)
            logger.info(f"Curriculum parameters diagram saved to {f_name}")

        return dot_paras

    class Config:
        validate_assignment = True


class DynamicForagingCurriculum(Curriculum[DynamicForagingParas,
                                           DynamicForagingMetrics]):
    """ Task-specific curriculum for dynamic foraging task
    Note that the two generic types {taskparas_class} and {metrics_class} are specified here
    """
    # Override parameters
    parameters: Dict[TrainingStage, DynamicForagingParas]

    # Override metrics
    def evaluate_transitions(self,
                             current_stage: TrainingStage,
                             metrics: DynamicForagingMetrics
                             ) -> TrainingStage:
        if not isinstance(metrics, DynamicForagingMetrics):
            raise TypeError(f"Expected metrics to be an instance of DynamicForagingMetrics, got {type(metrics).__name__} instead.")

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
