#%%
from schema.curriculum import Curriculum, StageTransitions, TransitionRule, TrainingStage, Metrics, ForagingTask, transform_dict_with_enum_keys
from schema.task import ForagingTask, TrainingStage

# %%
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

# %%
