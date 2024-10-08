'''
Curriculum for Dynamic Foraging - Uncoupled without Baiting
Adopted from "draft 2, began using 10/17/23"
https://alleninstitute-my.sharepoint.com/:w:/g/personal/katrina_nguyen_alleninstitute_org/EUGu5FS565pLuHhqFYT9yfEBjF7tIDGVEmbwnmcCYJBoWw?e=wD8fX9

Run the code to generate the curriculum.json and graphs
Added reward delay and shortened no-lick delay to take care of introduction of extra ITI when early licking

'''

#%%
from aind_auto_train.curriculum_manager import LOCAL_SAVED_CURRICULUM_ROOT
from aind_auto_train.schema.curriculum import (
    DynamicForagingCurriculum, StageTransitions, TransitionRule,
    Decision
)
from aind_auto_train.schema.task import (
    Task, TrainingStage, DynamicForagingParas,
    AutoWaterMode, AdvancedBlockMode
)
from aind_auto_train import setup_logging
setup_logging()

# Note this could be any string, not necessarily one of the Task enums
curriculum_name = Task.C0B0
curriculum_version = "2.3.1rwdDelay159"
curriculum_description = '''2024-08-16 max_len = 75 mins; decrease finished trial criterion'''

task_url = "https://github.com/AllenNeuralDynamics/dynamic-foraging-task"
task_schema_version = "1.1.0"

# --- Parameters ---
# Stage 1 with warmup (classical Stage 1.1 + 1.2)

paras_stage_1_warmup = DynamicForagingParas(
    # Metainfo
    training_stage=TrainingStage.STAGE_1_WARMUP,
    description="Warmup, followed by legendary Coupled Baiting Stage 1.2 (block = [10, 30, 10], p_sum = 0.8, p_ratio = [1:0])",

    # -- Essentials --
    # Warmup ON
    warmup='on',
    warm_min_trial=50,
    warm_max_choice_ratio_bias=0.1,
    warm_min_finish_ratio=0.8,
    warm_windowsize=20,

    # First session is ** coupled baiting **
    task_url=task_url,
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

    # Add a (fixed) small delay period at the beginning  # TODO: automate delay period
    DelayMin=0, # almost turned off no lick window
    DelayMax=0,
    DelayBeta=0,

    # Reward size and reward delay
    RewardDelay=0.1,
    RightValue_volume=4.0,
    LeftValue_volume=4.0,

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
    MaxTime=75,
    StopIgnores=20000,

    # -- Miscs --
    ResponseTime=5.0,  # Very long response time at the beginning
    RewardConsumeTime=1.0,  # Shorter RewardConsumeTime to increase the number of trials
    UncoupledReward="",  # Only valid in uncoupled task
)

transition_from_stage_1_warmup = StageTransitions(
    from_stage=TrainingStage.STAGE_1_WARMUP,
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
        ),
        TransitionRule(
            decision=Decision.PROGRESS,
            to_stage=TrainingStage.STAGE_1,
            condition_description="After the first session",
            condition="""lambda metrics:
                        metrics.session_at_current_stage >= 1
                        """,
        )

    ]
)

# Stage 1 without warmup (classical 1.2)
paras_stage_1 = DynamicForagingParas(
    **{
        **paras_stage_1_warmup.model_dump(),
        **dict(
            training_stage=TrainingStage.STAGE_1,
            description="Phase B in Han's slides (block = [10, 30, 10], p_sum = 0.8, p_ratio = [1:0])",

            # -- Essentials --
            # Turn off Warmup from now on
            warmup='off',

            Unrewarded=5,
            Ignored=5,
            
            # Decrease water size to 2.0 from now on
            RightValue_volume=2.0,
            LeftValue_volume=2.0,
        )
    }
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
            description="Coupled without baiting (block = [20, 35, 10], p_sum = 0.8, p_ratio = [8:1])",

            # --- Only include changes compared to stage_1 ---
            # -- Essentials --

            # Coupled no baiting
            task=Task.C1B0,

            # p_ratio [1:0] -> [8:1]
            RewardFamily=1,
            RewardPairsN=1,
            
            # Decrease autowater
            Unrewarded=7,
            Ignored=7,

            # block length [10, 30, 10] --> [20, 35, 20]
            BlockMin=20,
            BlockMax=35,
            BlockBeta=10,

            # ITI [1, 7, 3] --> [1, 10, 3]
            ITIMax=10,
            
            # Delay 0 --> 0.25
            DelayMin=0.25,
            DelayMax=0.25,

            StopIgnores=25,

            # -- Within session automation --
            # Miscs
            ResponseTime=1.5,  # Decrease response time: 5 --> 1.5
        )
    }
)

transition_from_stage_2 = StageTransitions(
    from_stage=TrainingStage.STAGE_2,
    transition_rules=[
        TransitionRule(
            decision=Decision.PROGRESS,
            to_stage=TrainingStage.STAGE_3,
            condition_description="Stay for >= 3 days",
            condition="""lambda metrics:
                        metrics.session_at_current_stage >= 3
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
            description="Coupled without baiting (block = [20, 35, 10], p_sum = 0.8, p_ratio = [8:1]), turn on no lick window",

            # --- Only include changes compared to stage_1 ---
            # -- Essentials --

            # Coupled no baiting
            task=Task.C1B0,

            # p_ratio [1:0] -> [8:1]
            RewardFamily=1,
            RewardPairsN=1,
            
            # Decrease autowater
            Unrewarded=10,
            Ignored=10,

            # block length [10, 30, 10] --> [20, 35, 20]
            BlockMin=20,
            BlockMax=35,
            BlockBeta=10,

            # ITI [1, 7, 3] --> [1, 10, 3]
            ITIMax=10,
            
            # Delay 0.5 --> 1.0
            DelayMin=1.0,
            DelayMax=1.0,

            # -- Within session automation --
            # Miscs
            ResponseTime=1.5,  # Decrease response time: 5 --> 1.5
        )
    }
)

transition_from_stage_3 = StageTransitions(
    from_stage=TrainingStage.STAGE_3,
    transition_rules=[
        TransitionRule(
            decision=Decision.PROGRESS,
            to_stage=TrainingStage.STAGE_4,
            condition_description="Finished trials >= 300 and efficiency >= 0.65 and stay for >= 3 days",
            condition="""lambda metrics:
                        metrics.finished_trials[-1] >= 300
                        and
                        metrics.foraging_efficiency[-1] >= 0.65
                        and
                        metrics.session_at_current_stage >= 3
                        """,
        ),
        TransitionRule(
            decision=Decision.ROLLBACK,
            to_stage=TrainingStage.STAGE_2,
            condition_description="Finished trials < 250 or efficiency < 0.50 after stay for >= 3 days",
            condition="""lambda metrics:
                        (metrics.finished_trials[-1] < 250
                        or
                        metrics.foraging_efficiency[-1] < 0.50)
                        and
                        metrics.session_at_current_stage >= 3
                        """,
        ),
    ]
)

paras_stage_4 = DynamicForagingParas(
    **{
        **paras_stage_3.model_dump(),
        **dict(
            training_stage=TrainingStage.STAGE_4,
            description="Switch to uncoupled; p_rew = [0.1, 0.4, 0.7] or [0.1, 0.5, 0.9]; turn on auto water for 1 days",

            # -- Essentials --
            # Uncoupled no baiting
            task=Task.C0B0,
            UncoupledReward="0.1, 0.5, 0.9",
            
            # reward delay
            RewardDelay=0.15, # increased from 100ms

            # Final block length for uncoupled task
            BlockMin=20,
            BlockMax=35,
            BlockBeta=10,

            # ITI [1, 10, 3] --> [1, 15, 3]
            ITIMax=15,

            # Turn on auto water for the first day after switching to uncoupled task
            AutoReward=True,
            Unrewarded=10,  # almost turned off
            Ignored=10,  # almost turned off

            # Turn off auto block
            AdvancedBlockAuto=AdvancedBlockMode.OFF,  # Turn off auto block
            
            # Miscs
            ResponseTime=1.5,
        )
    }
)

transition_from_stage_4 = StageTransitions(
    from_stage=TrainingStage.STAGE_4,
    transition_rules=[
        TransitionRule(
            decision=Decision.PROGRESS,
            to_stage=TrainingStage.STAGE_FINAL,
            condition_description="Just stay for 2 days",
            condition="""lambda metrics:
                                metrics.session_at_current_stage >= 2
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
            description="Uncoupled without baiting; p_rew = [0.1, 0.5, 0.9]; turn off auto water",

            # Essentials
            # Uncoupled no baiting
            task=Task.C0B0,
            UncoupledReward="0.1, 0.5, 0.9",

            BlockMin=20,
            BlockMax=35,
            BlockBeta=10,
            BlockMinReward=0,

            ITIMin=2.0,
            ITIMax=15.0,
            ITIBeta=3.0,

            DelayMin=1.0,
            DelayMax=1.0,
            DelayBeta=0.0,

            RewardDelay=0.2,

            # Within session automation
            AutoReward=False,  # Turn off auto water
            AdvancedBlockAuto=AdvancedBlockMode.OFF,  # Turn off auto block

            MaxTrial=1000,
            MaxTime=75,
            StopIgnores=25,

            # Miscs
            ResponseTime=1.5,
            RewardConsumeTime=1.0,
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
                                   "mean finished trials >= 400 and mean efficiency >= 0.65 "
                                   "and total sessions >= 10 and sessions at final >= 5"),
            condition="""lambda metrics:
                        metrics.session_total >= 10 
                        and
                        metrics.session_at_current_stage >= 5
                        and 
                        np.mean(metrics.finished_trials[-5:]) >= 400
                        and
                        np.mean(metrics.foraging_efficiency[-5:]) >= 0.65
                        """,
        ),
        TransitionRule(
            decision=Decision.ROLLBACK,
            to_stage=TrainingStage.STAGE_4,  # Back to C0B0 with auto water
            condition_description="For recent 5 sessions, mean finished trials < 250 or efficiency < 0.6",
            condition="""lambda metrics:
                        np.mean(metrics.finished_trials[-5:]) < 250
                        or
                        np.mean(metrics.foraging_efficiency[-5:]) < 0.60
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
        TrainingStage.STAGE_1_WARMUP: paras_stage_1_warmup,
        TrainingStage.STAGE_1: paras_stage_1,
        TrainingStage.STAGE_2: paras_stage_2,
        TrainingStage.STAGE_3: paras_stage_3,
        TrainingStage.STAGE_4: paras_stage_4,
        TrainingStage.STAGE_FINAL: paras_stage_final,
        TrainingStage.GRADUATED: paras_stage_final,        
    },

    curriculum={
        TrainingStage.STAGE_1_WARMUP: transition_from_stage_1_warmup,
        TrainingStage.STAGE_1: transition_from_stage_1,
        TrainingStage.STAGE_2: transition_from_stage_2,
        TrainingStage.STAGE_3: transition_from_stage_3,
        TrainingStage.STAGE_4: transition_from_stage_4,
        TrainingStage.STAGE_FINAL: transition_from_stage_final,
    },

)

# %%
if __name__ == '__main__':
    #%%
    import os

    curriculum_path = LOCAL_SAVED_CURRICULUM_ROOT
    os.makedirs(curriculum_path, exist_ok=True)

    # Save curriculum json and diagrams
    curriculum.save_to_json(path=curriculum_path)
    curriculum.diagram_rules(path=curriculum_path,
                             render_file_format='svg')
    #%%
    curriculum.diagram_paras(path=curriculum_path,
                                render_file_format='svg',
                                fontsize=12)

# %%
