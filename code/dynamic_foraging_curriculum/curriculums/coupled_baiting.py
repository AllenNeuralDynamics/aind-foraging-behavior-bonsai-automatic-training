#%%
from dynamic_foraging_curriculum.schema.curriculum import Curriculum, StageTransitions, TransitionRule, TrainingStage, Metrics, ForagingTask, transform_dict_with_enum_keys
from dynamic_foraging_curriculum.schema.task import ForagingTask, TrainingStage
import numpy as np


# %%
coupled_baiting_curriculum = Curriculum(
    task=ForagingTask.C1B1,
    curriculum_version="0.1",
    
    stage_transitions={
        TrainingStage.STAGE_1: StageTransitions(
            from_stage=TrainingStage.STAGE_1,
            transition_rules=[
                TransitionRule(
                    to_stage=TrainingStage.STAGE_2,
                    condition=lambda metrics: 
                        metrics.finished_trials[-1] >= 100 
                        and
                        metrics.foraging_efficiency[-1] > 0.7,
                    description="Finished trials >= 100 and efficiency >= 0.7",
                    )
                ]
        ),

        TrainingStage.STAGE_2: StageTransitions(
            from_stage=TrainingStage.STAGE_2,
            transition_rules=[
                TransitionRule(
                    to_stage=TrainingStage.STAGE_3,
                    condition=lambda metrics:
                        np.mean(metrics.finished_trials[-np.max([3, metrics.session_at_current_stage]):]) >= 500
                        and
                        np.mean(metrics.foraging_efficiency[-np.max([3, metrics.session_at_current_stage]):]) > 0.7,
                    description="For recent 3 sessions, mean finished trials >= 500 and efficiency >= 0.7",                    
                ),
                TransitionRule(
                    to_stage=TrainingStage.STAGE_1,
                    condition=lambda metrics: 
                        metrics.finished_trials[-1] < 100
                        or
                        metrics.foraging_efficiency[-1] < 0.6,
                    description="Finished trials < 100 or efficiency < 0.6",
                ),
            ]
        )
    }
    # Add other stage transition_rules as needed
)

# %%
