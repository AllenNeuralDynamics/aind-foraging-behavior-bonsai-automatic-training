'''
Curriculum for Dynamic Foraging - Uncoupled Baiting
Adapted from the Uncoupled Without Baiting curriculum
"draft 2, began using 10/17/23"
https://alleninstitute-my.sharepoint.com/:w:/g/personal/katrina_nguyen_alleninstitute_org/EUGu5FS565pLuHhqFYT9yfEBjF7tIDGVEmbwnmcCYJBoWw?e=wD8fX9

Run the code to generate the curriculum.json and graphs

'''

# %%
from aind_auto_train.curriculum_manager import LOCAL_SAVED_CURRICULUM_ROOT

from aind_auto_train.schema.curriculum import (
    DynamicForagingCurriculum, StageTransitions, TransitionRule,
    Decision
)
from aind_auto_train.schema.task import (
    Task, TrainingStage, DynamicForagingParas,
    AutoWaterMode, AdvancedBlockMode
)

# Note this could be any string, not necessarily one of the Task enums
curriculum_name = "Uncoupled Baiting"
curriculum_version = "0.1"
curriculum_description = '''Base curriculum for the uncoupled-baiting task'''

task_schema_version = "1.0"

# --- Parameters ---
# Notes on the STAGE_1:
# On the very first session, we typically run Left-Right alternating for 30~50 trials (previously called stage 1.1)
# and then, within the same session, proceed to block switch [min=10, max=20, beta=5] (previously called stage 1.2).
# Since the curriculum defined here only runs on a session-by-session basis, we decided to leave any within-session automation to the GUI.
# Therefore, here I define parameters for TrainingStage.STAGE_1 as the old stage 1.2, assuming that the GUI will take care
# of stage 1.1 automatically, whenever it receives a training stage name of TrainingStage.STAGE_1 ("Stage 1") AND session = 1.

paras_stage_1 = DynamicForagingParas(
    # Metainfo
    training_stage=TrainingStage.STAGE_1,
    description="Legendary Coupled Baiting Stage 1.2 (block = [10, 30, 10], p_sum = 0.8, p_ratio = [1:0])",

    # -- Essentials --
    # First session is ** coupled baiting **
    task_schema_version=task_schema_version,
    task=Task.C1B1,

    # p_sum = 0.8, p_ratio = [1:0]
    BaseRewardSum=0.8,
    RewardFamily=3,
    RewardPairsN=1,

    # block = [10, 30, 10]
    BlockMin=10,
    BlockMax=30,
    BlockBeta=10,
    BlockMinReward=0,

    ITIMin=1,
    ITIMax=7,
    ITIBeta=3,

    # Start with no delay
    DelayMin=0,
    DelayMax=0,
    DelayBeta=0.5,

    # Reward delay
    RewardDelay=0,

    # -- Within session automation --
    # Auto water
    AutoReward=True,
    AutoWaterType=AutoWaterMode.NATURAL,
    Unrewarded=3,
    Ignored=3,
    Multiplier=0.5,

    # Auto block
    AdvancedBlockAuto=AdvancedBlockMode.NOW,
    SwitchThr=0.5,
    PointsInARow=5,

    # Auto stop; set StopIgnores to a large number at the beginning
    MaxTrial=1000,
    MaxTime=90,
    StopIgnores=20000,

    # -- Miscs --
    ResponseTime=5,  # Very long response time at the beginning
    RewardConsumeTime=3,
    UncoupledReward="",  # Only valid in uncoupled task
)

transition_from_stage_1 = StageTransitions(
    from_stage=TrainingStage.STAGE_1,
    transition_rules=[
        TransitionRule(
            decision=Decision.PROGRESS,
            to_stage=TrainingStage.STAGE_2,
            condition_description="Finished trials >= 200 and efficiency >= 0.6",
            condition="""lambda metrics:
                        metrics.finished_trials[-1] >= 200
                        and
                        metrics.foraging_efficiency[-1] >= 0.6
                        """,
        )
    ]
)

paras_stage_2 = DynamicForagingParas(
    **{
        **paras_stage_1.model_dump(),
        **dict(
            training_stage=TrainingStage.STAGE_2,
            description="Coupled baiting (block = [20, 35, 20], p_sum = 1.0, p_ratio = [8:1])",

            # --- Only include changes compared to stage_1 ---
            # -- Essentials --

            # Coupled baiting
            task=Task.C1B1,

            # p_sum = 0.8 --> 1.0, p_ratio [1:0] -> [8:1]
            BaseRewardSum=1.0,
            RewardFamily=1,
            RewardPairsN=1,

            # block length [10, 30, 10] --> [20, 35, 20]
            BlockMin=20,
            BlockMax=35,
            BlockBeta=20,

            # ITI [1, 7, 3] --> [1, 25, 3]
            ITIMax=25,

            # -- Within session automation --
            # Miscs
            ResponseTime=2,  # Decrease response time: 5 --> 2
        )
    }
)

transition_from_stage_2 = StageTransitions(
    from_stage=TrainingStage.STAGE_2,
    transition_rules=[
        TransitionRule(
            decision=Decision.PROGRESS,
            to_stage=TrainingStage.STAGE_3,
            condition_description="Finished trials >= 300 and efficiency >= 0.65 and stay for >= 2 days",
            condition="""lambda metrics:
                        metrics.finished_trials[-1] >= 300
                        and
                        metrics.foraging_efficiency[-1] >= 0.65
                        and
                        metrics.session_at_current_stage >= 2
                        """,
        ),
        TransitionRule(
            decision=Decision.ROLLBACK,
            to_stage=TrainingStage.STAGE_1,
            condition_description="Finished trials < 200 or efficiency < 0.55",
            condition="""lambda metrics:
                        metrics.finished_trials[-1] < 200
                        or
                        metrics.foraging_efficiency[-1] < 0.55
                        """,
        ),
    ]
)

paras_stage_3 = DynamicForagingParas(
    **{
        **paras_stage_2.model_dump(),
        **dict(
            training_stage=TrainingStage.STAGE_3,
            description="Coupled baiting; remove auto water; add delay",

            # -- Essentials --

            # Coupled baiting
            task=Task.C1B1,

            # Delay 0.0 --> 1.0
            DelayMin=1.0,
            DelayMax=1.0,
            DelayBeta=0.5,

            # Turn off auto water
            AutoReward=False,
        )
    }
)

transition_from_stage_3 = StageTransitions(
    from_stage=TrainingStage.STAGE_3,
    transition_rules=[
        TransitionRule(
            decision=Decision.PROGRESS,
            to_stage=TrainingStage.STAGE_4,
            condition_description="Finished trials >= 400 and efficiency >= 0.7 and stay for >= 2 days",
            condition="""lambda metrics:
                                metrics.finished_trials[-1] >= 400
                                and
                                metrics.foraging_efficiency[-1] >= 0.7
                                and
                                metrics.session_at_current_stage >= 2
                                """,
        ),
        TransitionRule(
            decision=Decision.ROLLBACK,
            to_stage=TrainingStage.STAGE_2,
            condition_description="Finished trials < 200 or efficiency < 0.6",
            condition="""lambda metrics:
                                metrics.finished_trials[-1] < 200
                                or
                                metrics.foraging_efficiency[-1] < 0.6
                                """,
        ),
    ]
)

paras_stage_4 = DynamicForagingParas(
    **{
        **paras_stage_3.model_dump(),
        **dict(
            training_stage=TrainingStage.STAGE_4,
            description="Switch to uncoupled but still baiting; p_rew = [0.1, 0.4, 0.7]; turn on auto water for 1 day",

            # -- Essentials --
            # Coupled baiting
            task=Task.C0B1,
            UncoupledReward="0.1, 0.4, 0.7",

            # Final block length for uncoupled task
            BlockMin=20,
            BlockMax=35,
            BlockBeta=10,

            # Turn on auto water for the first day after switching to uncoupled task
            AutoReward=True,
            Unrewarded=10,
            Ignored=1000,

            # Turn off auto block
            AdvancedBlockAuto=AdvancedBlockMode.OFF,  # Turn off auto block
        )
    }
)

transition_from_stage_4 = StageTransitions(
    from_stage=TrainingStage.STAGE_4,
    transition_rules=[
        TransitionRule(
            decision=Decision.PROGRESS,
            to_stage=TrainingStage.STAGE_FINAL,
            condition_description="Just stay for 1 day",
            condition="""lambda metrics:
                                metrics.session_at_current_stage >= 1
                                """,
        ),
        # Once we reach here (C0B0), maybe we should not roll back to C1B0 or C1B1 anymore?
    ]
)

paras_stage_final = DynamicForagingParas(
    **{
        **paras_stage_4.model_dump(),
        **dict(
            training_stage=TrainingStage.STAGE_FINAL,
            description="Uncoupled baiting; p_rew = [0.1, 0.4, 0.7]; turn off auto water",

            # Essentials
            # Coupled baiting
            task=Task.C0B1,
            UncoupledReward="0.1, 0.4, 0.7",

            BlockMin=20,
            BlockMax=35,
            BlockBeta=10,
            BlockMinReward=0,

            ITIMin=1.0,
            ITIMax=25.0,
            ITIBeta=3.0,

            DelayMin=1.0,
            DelayMax=1.0,
            DelayBeta=0.5,

            RewardDelay=0,

            # Within session automation
            AutoReward=False,  # Turn off auto water
            AdvancedBlockAuto=AdvancedBlockMode.OFF,  # Turn off auto block

            MaxTrial=1000,
            MaxTime=90,
            StopIgnores=20000,

            # Miscs
            ResponseTime=2.0,
            RewardConsumeTime=3.0,
        )
    }
)

transition_from_stage_final = StageTransitions(
    from_stage=TrainingStage.STAGE_FINAL,
    transition_rules=[
        TransitionRule(
            # For graduation, obviously we need more requirements.
            decision=Decision.PROGRESS,
            to_stage=TrainingStage.GRADUATED,
            condition_description=("For recent 5 sessions,"
                                   "mean finished trials >= 500 and mean efficiency >= 0.7 "
                                   "and total sessions >= 10 and sessions at final >= 5"),
            condition="""lambda metrics:
                        metrics.session_total >= 10 
                        and
                        metrics.session_at_current_stage >= 5
                        and
                        np.mean(metrics.finished_trials[-5:]) >= 500
                        and
                        np.mean(metrics.foraging_efficiency[-5:]) >= 0.7
                        """,
        ),
        TransitionRule(
            decision=Decision.ROLLBACK,
            to_stage=TrainingStage.STAGE_4,  # Back to C0B0 with auto water
            condition_description="For recent 2 sessions, mean finished trials < 400 or efficiency < 0.6",
            condition="""lambda metrics:
                        np.mean(metrics.finished_trials[-2:]) < 400
                        or
                        np.mean(metrics.foraging_efficiency[-2:]) < 0.6
                        """,
        ),
    ]
)

# --- Curriculum ---
# %%
curriculum = DynamicForagingCurriculum(
    curriculum_name=curriculum_name,
    curriculum_version=curriculum_version,
    curriculum_description=curriculum_description,

    parameters={
        TrainingStage.STAGE_1: paras_stage_1,
        TrainingStage.STAGE_2: paras_stage_2,
        TrainingStage.STAGE_3: paras_stage_3,
        TrainingStage.STAGE_4: paras_stage_4,
        TrainingStage.STAGE_FINAL: paras_stage_final,
    },

    curriculum={
        TrainingStage.STAGE_1: transition_from_stage_1,
        TrainingStage.STAGE_2: transition_from_stage_2,
        TrainingStage.STAGE_3: transition_from_stage_3,
        TrainingStage.STAGE_4: transition_from_stage_4,
        TrainingStage.STAGE_FINAL: transition_from_stage_final,
    },

)

# %%
if __name__ == '__main__':
    import os

    curriculum_path = LOCAL_SAVED_CURRICULUM_ROOT
    os.makedirs(curriculum_path, exist_ok=True)

    # Save curriculum json and diagrams
    curriculum.save_to_json(path=curriculum_path)
    curriculum.diagram_rules(path=curriculum_path,
                             render_file_format='svg')
    curriculum.diagram_paras(path=curriculum_path,
                             render_file_format='svg',
                             fontsize=12)
