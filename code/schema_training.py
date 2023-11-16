# %%
from pydantic import Field
from typing import List, NamedTuple, Callable
from enum import Enum

from aind_data_schema.base import AindModel
from aind_data_schema.session import Session
from aind_data_schema.stimulus import BehaviorStimulation

# %%
class ForagingTask(Enum):
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
    GRADUATED = "graduated"

class DynamicForagingTaskSchema(AindModel):
    ''' Training schema for the dynamic foraging GUI.
        This fully defines a set of training parameters that could be used in the GUI.
        For simplicity, let's start with a flat structure and use exactly the same names as in the GUI.
    '''
    # Metadata
    schema_version: str = Field("0.1", title="Schema version", const=True)
    curriculum_version: str = Field("0.1", title="Curriculum version", const=True)
    task: ForagingTask = Field(ForagingTask.C1B1, title="Task name")
    training_stage: float = Field(TrainingStage.STAGE_1, title="Training stage")
    
    # --- Critical training parameters ---
    # Reward probability
    BaseRewardSum: float = Field(1.0, title="Sum of p_reward")
    RewardFamily: int = Field(3, title="Reward family")  # Should be explicit here
    RewardParisN: int = Field(1, title="Number of pairs")  # Should be explicit here
    
    UncoupledReward: str = Field("0.1,0.3,0.7", title="Uncoupled reward")  # For uncoupled tasks only
    
    # Block length
    BlockMin:  int = Field(1, title="Block length (min)")
    BlockMax:  int = Field(1, title="Block length (max)")
    BlockBeta:  int = Field(1, title="Block length (beta)")
    BlockMinReward:  int = Field(1, title="Minimal rewards in a block to switch")
    
    # Delay period
    DelayMin:  float = Field(0.0, title="Delay period (min) ") 
    DelayMax: float = Field(0.0, title="Delay period (max) ")
    DelayBeta: float = Field(0.5, title="Delay period (beta)")
    
    # Auto water
    AutoReward: bool = Field(True, title="Auto reward switch")
    AutoWaterType: AutoWaterMode = Field(AutoWaterMode.NATURAL, title="Auto water mode")
    Multiplier: float = Field(0.5, title="Multiplier for auto reward")
    Unrewarded: int = Field(0, title="Number of unrewarded trials before auto water")
    Ignored: int = Field(0, title="Number of ignored trials before auto water")
        
    # ITI
    ITIMin: float = Field(3, title="ITI (min)")
    ITIMax: float = Field(15, title="ITI (max)")
    ITIBeta: float = Field(5, title="ITI (beta)")
    ITIIncrease: float = Field(0.0, title="ITI increase")   # TODO: not implemented in the GUI??
    
    # Response time
    ResponseTime: float = Field(10, title="Response time")
    RewardConsumeTime: float = Field(3.0, title="Reward consume time", 
                                     description="Time of the no-lick period before trial end")
    StopIgnores: int = Field(20000, title="Number of ignored trials before stop")
    
    # Auto block
    AdvancedBlockAuto: AdvancedBlockMode = Field(AdvancedBlockMode.OFF, title="Auto block mode")    
    SwitchThr: float = Field(0.5, title="Switch threshold for auto block")
    PointsInARow: int = Field(5, title="Points in a row for auto block")
                             
    # Auto stop
    MaxTrial: int = Field(1000, title="Maximal number of trials")
    MaxTime: int = Field(90, title="Maximal session time (min)")
                             
    # Reward size. TODO: which one has higher priority? valve open time or volume?
    RightValue: float = Field(0.05, title="Right reward size (valve open time in sec)")
    LeftValue: float = Field(0.05, title="Left reward size (valve open time in sec)")
    RightValue_volume: float = Field(5.00, title="Right reward size (volume)")
    LeftValue_volume: float = Field(5.00, title="Left reward size (volume)")
    
    # --- Other GUI fields that will never be changed by the script (only clicked by the user) ---
    NextBlock: bool = Field(False, title="(User clicks) Next block", const=True)
    GiveLeft: bool = Field(False, title="(User clicks) Give left", const=True)
    GiveRight: bool = Field(False, title="(User clicks) Give right", const=True)
    GiveWaterL: float = Field(0.03, title="(User clicks) Size of give water left", const=True)
    GiveWaterR: float = Field(0.03, title="(User clicks) Size of give water right", const=True)
    GiveWaterL_volume: float = Field(3.00, title="(User clicks) Size of give water left (volume)", const=True)
    GiveWaterR_volume: float = Field(3.00, title="(User clicks) Size of give water right (volume)", const=True)
    IncludeAutoReward: bool = Field(False, title="(User clicks) Include auto reward", const=True)
    SaveTraining: bool = Field(True, title="(User clicks) Save training", const=True)
    InitiallyInactiveN: int = Field(2, title="Initially inactive trials")   # TODO: What is this???
    Randomness: str = Field("Exponential", title="Randomness mode", const=True)
    
    qt_spinbox_lineedit: float = Field(5.0, title="qt_spinbox_lineedit??", const=True)  # TODO:What is this???
    
    def to_GUI_format(self) -> dict:
        '''Turn to the GUI format, especially convert numbers to strings
        '''        
        return {key: (value if isinstance(value, bool) else  # Boolean --> keep it as it is
                      value.value if isinstance(value, Enum) else  # Enum --> use its name
                      str(value))   # All other type -> str
                for key, value in self.dict().items()}

    class Config:
        validate_assignment = True

