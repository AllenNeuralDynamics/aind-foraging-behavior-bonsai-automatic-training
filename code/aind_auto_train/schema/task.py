# %%
from pydantic import Field
from typing import List, TypeVar
from enum import Enum

from pydantic import BaseModel
from aind_data_schema.base import AindModel

# %%
class Task(Enum):
    """Foraging tasks"""
    C1B1 = "Coupled Baiting"
    C0B0 = "Uncoupled Without Baiting"
    C1B0 = "Coupled Without Baiting"
    C0B1 = "Uncoupled Baiting"
    
class AdvancedBlockMode(Enum):
    ''' Modes for advanced block '''
    OFF = "off"
    NOW = "now"
    ONCE = "once"
    
class AutoWaterMode(Enum):
    ''' Modes for auto water '''
    NATURAL = "Natural"
    BOTH = "Both"
    HIGH_PRO = "High pro"
    
class TrainingStage(Enum):
    STAGE_1 = "Stage 1"  # A special first stage that needed to be splitted into 1.1 and 1.2 on the GUI side
    STAGE_2 = "Stage 2"
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
    
class TaskParas(AindModel):
    """Parent class for TaskParas. All other task parameters should inherit from this class
    """
    # Metadata
    task: Task = Field(Task.C1B1, title="Task name")
    task_schema_version: str = Field("0.1", title="Schema version")  # Corresponding to the GUI
    curriculum_version: str = Field("0.1", title="Curriculum version")  # Corresponding to the curriculum
    training_stage: TrainingStage = Field(TrainingStage.STAGE_1, title="Training stage")
    description: str = Field("", title='Description of this set of parameters')
    
    class Config:
        validate_assignment = True
    
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
    RewardParisN: int = Field(..., title="Number of pairs")  # Should be explicit here
    
    UncoupledReward: str = Field("0.1,0.3,0.7", title="Uncoupled reward")  # For uncoupled tasks only
    
    # Block length
    BlockMin:  int = Field(..., title="Block length (min)")
    BlockMax:  int = Field(..., title="Block length (max)")
    BlockBeta:  int = Field(..., title="Block length (beta)")
    BlockMinReward:  int = Field(1, title="Minimal rewards in a block to switch")
    
    # Delay period
    DelayMin:  float = Field(..., title="Delay period (min) ") 
    DelayMax: float = Field(..., title="Delay period (max) ")
    DelayBeta: float = Field(..., title="Delay period (beta)")
    
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
                             
    # Reward size. TODO: which one has higher priority? valve open time or volume?
    RightValue: float = Field(0.05, title="Right reward size (valve open time in sec)", exclude_from_GUI=True)  # exclude_from_GUI means this will not sent to the GUI
    LeftValue: float = Field(0.05, title="Left reward size (valve open time in sec)", exclude_from_GUI=True)
    RightValue_volume: float = Field(5.00, title="Right reward size (volume)", exclude_from_GUI=True)
    LeftValue_volume: float = Field(5.00, title="Left reward size (volume)", exclude_from_GUI=True)
    
    # --- Other GUI fields that will never be changed by the script (only clicked by the user) ---
    NextBlock: bool = Field(False, title="(User clicks) Next block", const=True, exclude_from_GUI=True)
    GiveLeft: bool = Field(False, title="(User clicks) Give left", const=True, exclude_from_GUI=True)
    GiveRight: bool = Field(False, title="(User clicks) Give right", const=True, exclude_from_GUI=True)
    GiveWaterL: float = Field(0.03, title="(User clicks) Size of give water left", const=True, exclude_from_GUI=True)
    GiveWaterR: float = Field(0.03, title="(User clicks) Size of give water right", const=True, exclude_from_GUI=True)
    GiveWaterL_volume: float = Field(3.00, title="(User clicks) Size of give water left (volume)", const=True, exclude_from_GUI=True)
    GiveWaterR_volume: float = Field(3.00, title="(User clicks) Size of give water right (volume)", const=True, exclude_from_GUI=True)
    IncludeAutoReward: bool = Field(False, title="(User clicks) Include auto reward", const=True, exclude_from_GUI=True)
    SaveTraining: bool = Field(True, title="(User clicks) Save training", const=True, exclude_from_GUI=True)
    InitiallyInactiveN: int = Field(2, title="Initially inactive trials", exclude_from_GUI=True)   # TODO: What is this???
    Randomness: str = Field("Exponential", title="Randomness mode", const=True, exclude_from_GUI=True)
    
    qt_spinbox_lineedit: float = Field(5.0, title="qt_spinbox_lineedit??", const=True, exclude_from_GUI=True)  # TODO:What is this???
    
    def to_GUI_format(self) -> dict:
        '''Turn to the GUI format, especially convert numbers to strings
        '''        
        return {key: (value if isinstance(value, bool) else  # Boolean --> keep it as it is
                      value.value if isinstance(value, Enum) else  # Enum --> use its name
                      str(value))   # All other type -> str
                for key, value in self.dict().items()
                if 'exclude_from_GUI' not in self.__fields__[key].field_info.extra}  # When generate paras for the GUI, exclude those manually controled fields


