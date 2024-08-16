'''
Curriculum for Dynamic Foraging - Coupled Baiting
https://alleninstitute.sharepoint.com/:p:/s/NeuralDynamics/EQwuU0I4PBtGsU2wilCHklEBDTXYGT3F-QtaN6iDGJLBmg?e=N10dya

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
from aind_auto_train import setup_logging
setup_logging()

curriculum_name = Task.C1B1
curriculum_version = "2.3"
curriculum_description = '''2024-08-16 max_len = 75 mins; decrease finished trial criterion'''

task_url = "https://github.com/AllenNeuralDynamics/dynamic-foraging-task"
task_schema_version = "1.1.0"

# --- Parameters ---
# Stage 1 with warmup (classical Stage 1.1 + 1.2)
paras_stage_1_warmup = DynamicForagingParas(
    # Metainfo
    task_url=task_url,
    task_schema_version=task_schema_version,
    task=Task.C1B1,
    training_stage=TrainingStage.STAGE_1_WARMUP,  # "Phase B" in Han's slides
    description="Warmup, followed by Phase B in Han's slides (block = [10, 20, 5], p_sum = 0.8, p_ratio = [1:0])",

    # -- Essentials --
    # Warmup ON
    warmup='on',
    warm_min_trial=50,
    warm_max_choice_ratio_bias=0.1,
    warm_min_finish_ratio=0.8,
    warm_windowsize=20,
    
    # p_sum = 0.8, p_ratio = [1:0]
    BaseRewardSum=0.8,
    RewardFamily=3,
    RewardPairsN=1,

    # block = [10, 20, 5]
    BlockMin=10,
    BlockMax=20,
    BlockBeta=5,
    BlockMinReward=0,

    # Small ITI at the beginning to better engage the animal
    ITIMin=1,
    ITIMax=7,
    ITIBeta=3,

    # Add a (fixed) small delay period at the beginning  # TODO: automate delay period
    DelayMin=0.1,
    DelayMax=0.1,
    DelayBeta=0,
    
    # Reward size and reward delay
    RewardDelay=0.0,
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
    ResponseTime=5.0,   # Very long response time at the beginning
    RewardConsumeTime=1.0,   # Shorter RewardConsumeTime to increase the number of trials
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
            description="Phase B in Han's slides (block = [10, 20, 5], p_sum = 0.8, p_ratio = [1:0])",

            # -- Essentials --
            # Turn off Warmup from now on
            warmup='off',
            
            Unrewarded=5,
            Ignored=5,      
                  
            # Decrease water size to 3.0 from now on
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

# "Phase C" in Han's slides
paras_stage_2 = DynamicForagingParas(
    **{
        **paras_stage_1.model_dump(),
        **dict(
            training_stage=TrainingStage.STAGE_2,
            description="Phase C in Han's slides (block = [10, 40, 10], p_sum = 0.6, p_ratio = [8:1])",

            # --- Only include changes compared to stage_1 ---
            # -- Essentials --
            # p_sum = 0.8 --> 0.6, p_ratio = [1:0] -> [8:1]
            BaseRewardSum=0.6,
            RewardFamily=1,
            RewardPairsN=1,

            # block length [10, 20, 5] --> [10, 40, 10]
            BlockMin=10,
            BlockMax=40,
            BlockBeta=10,

            # ITI [1, 7, 3] --> [1, 10, 5]
            ITIMin=1,
            ITIMax=10,
            ITIBeta=3,

            DelayMin=0.3,
            DelayMax=0.3,

            # -- Within session automation --
            # Decrease auto water: unrewarded 5 --> 10, ignored 5 --> 10
            Unrewarded=7,
            Ignored=7,

            # Increase auto block switch threshold: 0.5 --> 0.6
            SwitchThr=0.6,
            StopIgnores=25,  # Auto stop on ignores-in-a-row starts to take effect

            # Miscs
            ResponseTime=3,  # Decrease response time: 5 --> 3
        )
    }
)

transition_from_stage_2 = StageTransitions(
    from_stage=TrainingStage.STAGE_2,
    transition_rules=[
        TransitionRule(
            decision=Decision.PROGRESS,
            to_stage=TrainingStage.STAGE_3,
            condition_description="Finished trials >= 300 and efficiency >= 0.65",
            condition="""lambda metrics:
                        metrics.finished_trials[-1] >= 300
                        and
                        metrics.foraging_efficiency[-1] >= 0.65
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

# "Phase D" in Han's slides
paras_stage_3 = DynamicForagingParas(
    **{
        **paras_stage_2.model_dump(),
        **dict(
            training_stage=TrainingStage.STAGE_3,
            description="Phase D in Han's slides (block = [10, 40, 10], p_sum = 0.45, p_ratio = [8:1])",

            # -- Essentials --
            # p_sum = 0.6 --> 0.45, p_ratio still [8:1]
            BaseRewardSum=0.45,

            # block length [10, 40, 10] --> [20, 60, 20]
            BlockMin=20,
            BlockMax=60,
            BlockBeta=20,

            # ITI [2, 10, 5] --> [3, 15, 5]
            ITIMin=1,
            ITIMax=15,
            ITIBeta=3,

            DelayMin=0.5,
            DelayMax=0.5,
            DelayBeta=0.0,

            # Decrease autowater number (almost turned off)
            Unrewarded=10,
            Ignored=10,

            # Miscs
            ResponseTime=2,  # Decrease response time:  3 --> 2
        )
    }
)

transition_from_stage_3 = StageTransitions(
    from_stage=TrainingStage.STAGE_3,
    transition_rules=[
        TransitionRule(
            decision=Decision.PROGRESS,
            to_stage=TrainingStage.STAGE_FINAL,
            condition_description="Finished trials >= 400 and efficiency >= 0.7",
            condition="""lambda metrics:
                        metrics.finished_trials[-1] >= 400
                        and
                        metrics.foraging_efficiency[-1] >= 0.7
                        """,
        ),
        TransitionRule(
            decision=Decision.ROLLBACK,
            to_stage=TrainingStage.STAGE_2,
            condition_description="Finished trials < 300 or efficiency < 0.65",
            condition="""lambda metrics:
                        metrics.finished_trials[-1] < 300
                        or
                        metrics.foraging_efficiency[-1] < 0.65
                        """,
        ),
    ]
)

# "Phase E" in Han's slides
paras_stage_final = DynamicForagingParas(
    **{
        **paras_stage_3.model_dump(),
        **dict(
            training_stage=TrainingStage.STAGE_FINAL,
            description="Phase E in Han's slides (full task: block = [20, 60, 20], p_sum = 0.45, p_ratio = [8:1], [6:1], [3:1], [1:1])",

            # --- Here I explicitly list all parameters again just for clarity ---
            # Essentials

            # Warmup OFF
            warmup='off',

            # p_sum = 0.45, p_ratio = [8:1] --> [8:1], [6:1], [3:1], [1:1]
            BaseRewardSum=0.45,
            RewardFamily=1,
            RewardPairsN=4,

            # block = [10, 20, 5] (mean ~ 33 trials)
            BlockMin=20,
            BlockMax=60,
            BlockBeta=20,
            BlockMinReward=0,

            # ITI [1, 15, 5] --> [1, 30, 5] (mean ~ 6.0 s, not included 1-s no lick window before ITI start)
            ITIMin=1,
            ITIMax=30,
            ITIBeta=3,

            DelayMin=1.0,
            DelayMax=1.0,
            DelayBeta=0.0,

            # Reward size and reward delay
            RewardDelay=0.0,
            RightValue_volume=2.0,
            LeftValue_volume=2.0,
    
            # Within session automation
            AutoReward=False,  # Turn off auto water
            AdvancedBlockAuto=AdvancedBlockMode.OFF,  # Turn off auto block

            MaxTrial=1000,
            MaxTime=75,
            StopIgnores=25,

            # Miscs
            ResponseTime=1.0,
            RewardConsumeTime=3.0,
            UncoupledReward="",  # Only valid in uncoupled task
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
                                   "mean finished trials >= 450 and mean efficiency >= 0.70 "
                                   "and total sessions >= 10 and sessions at final >= 5"),
            condition="""lambda metrics:
                        metrics.session_total >= 10 
                        and
                        metrics.session_at_current_stage >= 5
                        and
                        np.mean(metrics.finished_trials[-5:]) >= 450
                        and
                        np.mean(metrics.foraging_efficiency[-5:]) >= 0.70
                        """,
        ),
        TransitionRule(
            decision=Decision.ROLLBACK,
            to_stage=TrainingStage.STAGE_3,
            condition_description="For recent 5 sessions, mean finished trials < 300 or efficiency < 0.60",
            condition="""lambda metrics:
                        np.mean(metrics.finished_trials[-5:]) < 300
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
        TrainingStage.STAGE_FINAL: paras_stage_final,
        TrainingStage.GRADUATED: paras_stage_final,
    },

    curriculum={
        TrainingStage.STAGE_1_WARMUP: transition_from_stage_1_warmup,
        TrainingStage.STAGE_1: transition_from_stage_1,
        TrainingStage.STAGE_2: transition_from_stage_2,
        TrainingStage.STAGE_3: transition_from_stage_3,
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
