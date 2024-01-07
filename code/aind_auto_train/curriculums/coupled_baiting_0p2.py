'''
Curriculum for Dynamic Foraging - Coupled Baiting
https://alleninstitute.sharepoint.com/:p:/s/NeuralDynamics/EQwuU0I4PBtGsU2wilCHklEBDTXYGT3F-QtaN6iDGJLBmg?e=N10dya

Run the code to generate the curriculum.json and graphs

'''

# %%

from aind_auto_train.schema.curriculum import TrainingStage
from aind_auto_train.curriculum_manager import LOCAL_SAVED_CURRICULUM_ROOT


# Reuse the curriculum from 0.1
from aind_auto_train.curriculums.coupled_baiting_0p1 import curriculum as curriculum_0p1

# Override some parameters
curriculum = curriculum_0p1.model_copy()
curriculum.curriculum_version = "0.2"
curriculum.curriculum_description = "More stringent criteria before GRADUATED than 0.1"

curriculum.get_transition_rule(
    TrainingStage.STAGE_FINAL, TrainingStage.GRADUATED).__dict__.update(**dict(
        condition_description=("For recent 7 sessions,"
                               "mean finished trials >= 500 and mean efficiency >= 0.75 "
                               "and total sessions >= 14 and sessions at final >= 7"),
        condition="""lambda metrics:
                    metrics.session_total >= 14 
                    and
                    metrics.session_at_current_stage >= 7
                    and
                    np.mean(metrics.finished_trials[-5:]) >= 500
                    and
                    np.mean(metrics.foraging_efficiency[-5:]) >= 0.75
                    """,
    ))

# curriculum.get_transition_rule(
#     TrainingStage.STAGE_1, TrainingStage.STAGE_2).__dict__.update(**dict(condition_description='feeff'))


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
