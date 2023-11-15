#%%
from pydantic import BaseModel, Field
from enum import Enum, auto
from typing import List, Callable, Dict

from schema_training import (DynamicForagingTrainingParameters, TrainingStages, 
                             ForagingTasks)

import logging
logging.basicConfig(level=logging.INFO, 
                    filename='curriculum.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

#%%
class BehavioralMetrics(BaseModel):
    foraging_efficiency: float
    finished_trials: int
    # ... other metrics as needed ...

class Transition(BaseModel):
    to_stage: TrainingStages
    condition: Callable[[BehavioralMetrics], bool] = Field(exclude=True)

class StageTransitions(BaseModel):
    from_stage: TrainingStages
    transitions: List[Transition]

class MouseTracker(BaseModel):
    mouse_id: str
    current_stage: TrainingStages
    metrics: BehavioralMetrics
    stage_transitions: Dict[TrainingStages, StageTransitions]

    def evaluate_transitions(self):
        transitions = self.stage_transitions[self.current_stage].transitions
        for transition in transitions:
            if transition.condition(self.metrics):
                self.current_stage = transition.to_stage
                logging.info(f"Mouse {self.mouse_id} transitioned from {self.current_stage.name} to {self.current_stage.name}")
                break
            # By default, stay at the current stage
            logging.info(f"Mouse {self.mouse_id} stayed at {self.current_stage.name}")


class TrainingManager(BaseModel):
    mice: Dict[str, MouseTracker]

    def update(self):
        for mouse_id, mouse_tracker in self.mice.items():
            mouse_tracker.evaluate_transitions()
            

# Define transitions for each stage
# ... more transitions for other stages ...

# Function to add a new mouse tracker
def add_mouse_tracker(mouse_id: str, initial_stage: TrainingStages):
    initial_metrics = BehavioralMetrics(foraging_efficiency=0.0, finished_trials=0)
    tracker = MouseTracker(
        mouse_id=mouse_id,
        current_stage=initial_stage,
        metrics=initial_metrics,
        stage_transitions={
            TrainingStages.STAGE_1: StageTransitions(
                from_stage=TrainingStages.STAGE_1,
                transitions=[
                    Transition(
                        to_stage=TrainingStages.STAGE_2, 
                        condition=lambda metrics: (
                            metrics.finished_trials >= 100
                            and metrics.foraging_efficiency > 0.7
                            )
                        ),
                    ]
                ),
            
            TrainingStages.STAGE_2: StageTransitions(
                from_stage=TrainingStages.STAGE_2,
                transitions=[
                    Transition(
                        to_stage=TrainingStages.STAGE_2, 
                        condition=lambda metrics: (
                            metrics.finished_trials >= 100
                            and metrics.foraging_efficiency > 0.7
                            )
                        ),
                    Transition(
                        to_stage=TrainingStages.STAGE_1, 
                        condition=lambda metrics: (
                            metrics.finished_trials <= 100
                            or metrics.foraging_efficiency < 0.5
                            )
                        ),
                    ]
                )
            }
        # Add other stage transitions as needed
    )
    return tracker

# %%
manager = TrainingManager(mice={})
manager.mice['Mouse1'] = add_mouse_tracker('Mouse1', TrainingStages.STAGE_1)
# %%
