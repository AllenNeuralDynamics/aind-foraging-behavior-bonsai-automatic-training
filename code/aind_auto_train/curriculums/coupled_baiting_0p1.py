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

curriculum_name = Task.C1B1
task=Task.C1B1
task_schema_version = "1.0"
curriculum_version = "0.1"
curriculum_description = '''Base curriculum for the coupled-baiting task'''

# --- Parameters ---
# Notes on the STAGE_1:
# On the very first session, we typically run Left-Right alternating for 30~50 trials (previously called stage 1.1)
# and then, within the same session, proceed to block switch [min=10, max=20, beta=5] (previously called stage 1.2).
# Since the curriculum defined here only runs on a session-by-session basis, we decided to leave any within-session automation to the GUI.
# Therefore, here I define parameters for TrainingStage.STAGE_1 as the old stage 1.2 ("Phase B" in my slides), assuming that the GUI will take care
# of stage 1.1 ("Phase A" in my slides) automatically, whenever it receives a training stage name of TrainingStage.STAGE_1 ("Stage 1").

# "Phase B" in Han's slides
paras_stage_1 = DynamicForagingParas(
    # Metainfo
    task_schema_version=task_schema_version,
    task=task,
    training_stage=TrainingStage.STAGE_1,  # "Phase B" in Han's slides
    description="Phase B in Han's slides (block = [10, 20, 5], p_sum = 0.8, p_ratio = [1:0])",

    # -- Essentials --
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
    DelayMin=0.5,
    DelayMax=0.5,
    DelayBeta=0,
    
    # Reward delay
    RewardDelay=0,

    # -- Within session automation --
    # Auto water
    AutoReward=True,
    AutoWaterType=AutoWaterMode.NATURAL,
    Unrewarded=5,
    Ignored=5,
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
    ResponseTime=5, RewardConsumeTime=3,  # Very long response time at the beginning
    UncoupledReward="",  # Only valid in uncoupled task
)

# "Phase C" in Han's slides
paras_stage_2 = paras_stage_1.model_copy(update=dict(
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

    # ITI [1, 7, 3] --> [2, 10, 5]
    ITIMin=2,
    ITIMax=10,
    ITIBeta=5,

    # Delay 0.5 --> 1.0
    DelayMin=1.0,
    DelayMax=1.0,

    # -- Within session automation --
    # Decrease auto water: unrewarded 5 --> 10, ignored 5 --> 10
    Unrewarded=10,
    Ignored=10,

    # Increase auto block switch threshold: 0.5 --> 0.6
    SwitchThr=0.6,
    StopIgnores=50,  # Auto stop on ignores-in-a-row starts to take effect

    # Miscs
    ResponseTime=3,  # Decrease response time: 5 --> 3
))

# "Phase D" in Han's slides
paras_stage_3 = paras_stage_2.model_copy(update=dict(
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
    ITIMin=3,
    ITIMax=15,
    ITIBeta=5,

    # Delay 1.0 --> 1.5
    DelayMin=1.5,
    DelayMax=1.5,

    # Miscs
    ResponseTime=2,  # Decrease response time:  3 --> 2
))

# "Phase E" in Han's slides
paras_stage_final = paras_stage_3.model_copy(update=dict(
    training_stage=TrainingStage.STAGE_FINAL,
    description="Phase E in Han's slides (full task: block = [20, 60, 20], p_sum = 0.45, p_ratio = [8:1], [6:1], [3:1], [1:1])",

    # --- Here I explicitly list all parameters again just for clarity ---
    # Essentials
    # p_sum = 0.45, p_ratio = [8:1] --> [8:1], [6:1], [3:1], [1:1]
    BaseRewardSum=0.45,
    RewardFamily=1,
    RewardPairsN=4,

    # block = [10, 20, 5] (mean ~ 33 trials)
    BlockMin=20,
    BlockMax=60,
    BlockBeta=20,
    BlockMinReward=0,

    # ITI [3, 15, 5] --> [3, 30, 5] (mean ~ 7.9 s, Bari et al. 2019)
    ITIMin=3,
    ITIMax=30,
    ITIBeta=5,

    # Delay 1.5 --> 2.0 (Bari et al. 2019)
    DelayMin=2.0,
    DelayMax=2.0,
    DelayBeta=0,

    # Within session automation
    AutoReward=False,  # Turn off auto water
    AdvancedBlockAuto=AdvancedBlockMode.OFF,  # Turn off auto block

    MaxTrial=1000,
    MaxTime=90,
    StopIgnores=50,

    # Miscs
    ResponseTime=2,
    RewardConsumeTime=3,
    UncoupledReward="",  # Only valid in uncoupled task
))


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
        TrainingStage.STAGE_FINAL: paras_stage_final,
    },

    curriculum={
        TrainingStage.STAGE_1: StageTransitions(
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
        ),

        TrainingStage.STAGE_2: StageTransitions(
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
        ),

        TrainingStage.STAGE_3: StageTransitions(
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
                    condition_description="Finished trials < 200 or efficiency < 0.6",
                    condition="""lambda metrics:
                        metrics.finished_trials[-1] < 200
                        or
                        metrics.foraging_efficiency[-1] < 0.6
                        """,
                ),
            ]
        ),


        TrainingStage.STAGE_FINAL: StageTransitions(
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
                    to_stage=TrainingStage.STAGE_3,
                    condition_description="For recent 2 sessions, mean finished trials < 400 or efficiency < 0.6",
                    condition="""lambda metrics:
                        np.mean(metrics.finished_trials[-2:]) < 400
                        or
                        np.mean(metrics.foraging_efficiency[-2:]) < 0.6
                        """,
                ),
            ]
        )
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
