# %%
from pydantic import Field
from typing import List, TypeVar, Literal
from enum import Enum

from pydantic import BaseModel
from aind_data_schema.base import AindModel

# %%
class Task(str, Enum):
    """Foraging tasks"""
    C1B1 = "Coupled Baiting"
    C0B0 = "Uncoupled Without Baiting"
    C1B0 = "Coupled Without Baiting"
    C0B1 = "Uncoupled Baiting"
    DUMMY = "Dummy task"
    
class AdvancedBlockMode(str, Enum):
    ''' Modes for advanced block '''
    OFF = "off"
    NOW = "now"
    ONCE = "once"
    
class AutoWaterMode(str, Enum):
    ''' Modes for auto water '''
    NATURAL = "Natural"
    BOTH = "Both"
    HIGH_PRO = "High pro"
    
class TrainingStage(str, Enum):
    STAGE_1_WARMUP = "Stage 1 w/warmup"  # Stage 1 with warmup (classical Stage 1.1 + 1.2)
    STAGE_1 = "Stage 1"   # Stage 1 without warmup (classical Stage 1.2)
    STAGE_2 = "Stage 2"
    STAGE_2_noLick = 'Stage 2 noLlick'
    STAGE_3 = "Stage 3"
    STAGE_4 = "Stage 4"
    STAGE_5 = "Stage 5"
    STAGE_FINAL = "Stage final"
    GRADUATED = "Graduated"
    
    
class Metrics(BaseModel):
    """ Parent class for all metrics. All other metrics should inherit from this class
    """
    session_total: int
    session_at_current_stage: int

# Metrics class in Curriculum must be a subclass of Metrics
metrics_class = TypeVar('metrics_class', bound=Metrics)

class DynamicForagingMetrics(Metrics):
    """ Metrics for dynamic foraging
    """
    foraging_efficiency: List[float]  # Full history of foraging efficiency
    finished_trials: List[int]  # Full history of finished trials
    
# For dummy task
class DummyTaskMetrics(Metrics):
    dummy_metric_float: List[float]
    dummy_metric_int: List[int]
    
class TaskParas(AindModel):
    """Parent class for TaskParas. All other task parameters should inherit from this class
    """
    # Metadata
    training_stage: TrainingStage = Field(..., title="Training stage")
    task: Task = Field(..., title="Task name")
    task_url: str = Field("", title="URL to the task description")
    task_schema_version: str = Field(..., title="Schema version")  # Corresponding to the GUI
    description: str = Field("", title='Description of this set of parameters')
    
    class Config:
        validate_assignment = True

    def to_GUI_format(self) -> dict:
        '''Turn to the GUI format, especially convert numbers to strings
        '''        
        return {key: (value if isinstance(value, bool) else  # Boolean --> keep it as it is
                      value.value if isinstance(value, Enum) else  # Enum --> use its name
                      str(value))   # All other type -> str
                for key, value in self.dict().items()
                if not self.__fields__[key].json_schema_extra or 
                'exclude_from_GUI' not in self.__fields__[key].json_schema_extra}  # When generate paras for the GUI, exclude those manually controled fields

    
# Task para class in Curriculum must be a subclass of TaskParas
taskparas_class = TypeVar('taskparas_class', bound=TaskParas)

class DynamicForagingParas(TaskParas):
    ''' Training schema for the dynamic foraging GUI.
    
    
        This fully defines a set of training parameters that could be used in the GUI.
        For simplicity, let's start with a flat structure and use exactly the same names as in the GUI.
    '''    
    # --- Critical training parameters ---
    # Reward probability
    BaseRewardSum: float = Field(..., title="Sum of p_reward")
    RewardFamily: int = Field(..., title="Reward family")  # Should be explicit here
    RewardPairsN: int = Field(..., title="Number of pairs")  # Should be explicit here
    
    UncoupledReward: str = Field("0.1,0.3,0.7", title="Uncoupled reward")  # For uncoupled tasks only
    
    # Randomness
    Randomness: str = Field('Exponential', title="Randomness mode")  # Exponential by default
    
    # Block length
    BlockMin:  int = Field(..., title="Block length (min)")
    BlockMax:  int = Field(..., title="Block length (max)")
    BlockBeta:  int = Field(..., title="Block length (beta)")
    BlockMinReward:  int = Field(1, title="Minimal rewards in a block to switch")
    
    # Delay period
    DelayMin:  float = Field(..., title="Delay period (min) ") 
    DelayMax: float = Field(..., title="Delay period (max) ")
    DelayBeta: float = Field(..., title="Delay period (beta)")
    
    # Reward delay
    RewardDelay: float = Field(..., title="Reward delay (sec)")
    
    # Auto water
    AutoReward: bool = Field(..., title="Auto reward switch")
    AutoWaterType: AutoWaterMode = Field(AutoWaterMode.NATURAL, title="Auto water mode")
    Multiplier: float = Field(..., title="Multiplier for auto reward")
    Unrewarded: int = Field(..., title="Number of unrewarded trials before auto water")
    Ignored: int = Field(..., title="Number of ignored trials before auto water")
        
    # ITI
    ITIMin: float = Field(..., title="ITI (min)")
    ITIMax: float = Field(..., title="ITI (max)")
    ITIBeta: float = Field(..., title="ITI (beta)")
    ITIIncrease: float = Field(0.0, title="ITI increase")   # TODO: not implemented in the GUI??
    
    # Response time
    ResponseTime: float = Field(..., title="Response time")
    RewardConsumeTime: float = Field(..., title="Reward consume time", 
                                     description="Time of the no-lick period before trial end")
    StopIgnores: int = Field(..., title="Number of ignored trials before stop")
    
    # Auto block
    AdvancedBlockAuto: AdvancedBlockMode = Field(..., title="Auto block mode")    
    SwitchThr: float = Field(..., title="Switch threshold for auto block")
    PointsInARow: int = Field(..., title="Points in a row for auto block")
                             
    # Auto stop
    MaxTrial: int = Field(..., title="Maximal number of trials")
    MaxTime: int = Field(..., title="Maximal session time (min)")
                             
    # Reward size
    RightValue_volume: float = Field(3.00, title="Right reward size (uL)")
    LeftValue_volume: float = Field(3.00, title="Left reward size (uL)")
    
    # Warmup
    warmup: str = Field('off', title="Warmup master switch")
    warm_min_trial: int = Field(50, title="Warmup finish criteria: minimal trials")
    warm_max_choice_ratio_bias: float = Field(0.1, title="Warmup finish criteria: maximal choice ratio bias from 0.5")
    warm_min_finish_ratio: float = Field(0.8, title="Warmup finish criteria: minimal finish ratio")
    warm_windowsize: int = Field(20, title="Warmup finish criteria: window size to compute the bias and ratio")
    
        


# For dummy task
class DummyTaskParas(TaskParas):
    dummy_para_bool: bool
    dummy_para_float: float

