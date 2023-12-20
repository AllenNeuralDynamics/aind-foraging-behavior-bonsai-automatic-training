"""A simple demo for adding a curriculum for a dummy task
"""
# %%
from typing import List, Dict

from aind_auto_train.curriculum_manager import LOCAL_SAVED_CURRICULUM_ROOT
from aind_auto_train.schema.task import Task, TrainingStage, DummyTaskParas, DummyTaskMetrics
from aind_auto_train.schema.curriculum import DummyTaskCurriculum, StageTransitions, TransitionRule, Decision


meta = dict(curriculum_version="0.1",
            task_schema_version="0.1",
            task=Task.DUMMY
            )

paras_stage_1 = DummyTaskParas(
    # Metainfo
    **meta,
    training_stage=TrainingStage.STAGE_1,  # "Phase B" in Han's slides
    description="Dummy stage 1",

    dummy_para_bool=False,
    dummy_para_float=0.1
)

paras_stage_2 = DummyTaskParas(
    # Metainfo
    **meta,
    training_stage=TrainingStage.STAGE_FINAL,  # "Phase B" in Han's slides
    description="Dummy stage final",

    dummy_para_bool=False,
    dummy_para_float=0.2
)

curriculum = DummyTaskCurriculum(
    curriculum_task=Task.DUMMY,
    curriculum_version="0.1",

    parameters={
        TrainingStage.STAGE_1: paras_stage_1,
        TrainingStage.STAGE_FINAL: paras_stage_2,
    },

    curriculum={
        TrainingStage.STAGE_1: StageTransitions(
            from_stage=TrainingStage.STAGE_1,
            transition_rules=[
                TransitionRule(
                    decision=Decision.PROGRESS,
                    to_stage=TrainingStage.STAGE_FINAL,
                    condition_description="metric float > 0.5 and metric int > 5",
                    condition="""lambda metrics: 
                                metrics.dummy_metric_float[-1] > 0.5 
                                and 
                                metrics.dummy_metric_int[-1] > 5
                            """
                )
            ]
        ),

        TrainingStage.STAGE_FINAL: StageTransitions(
            from_stage=TrainingStage.STAGE_2,
            transition_rules=[
                TransitionRule(
                    decision=Decision.PROGRESS,
                    to_stage=TrainingStage.GRADUATED,
                    condition_description="for the last 5 sessions, metric float > 0.7 and metric int > 10",
                    condition="""lambda metrics: 
                    metrics.session_total >= 5
                    and
                    np.mean(metrics.dummy_metric_float[-5:]) > 0.7 
                    and 
                    np.mean(metrics.dummy_metric_int[-5:] > 10
                    """
                ),

                TransitionRule(
                    decision=Decision.ROLLBACK,
                    to_stage=TrainingStage.STAGE_1,
                    condition_description="metric float < 0.6 or metric int < 7",
                    condition="""lambda metrics:
                    metrics.dummy_metric_float[-1] < 0.6 
                    and 
                    metrics.dummy_metric_int[-1] < 7                    
                    """
                ),
            ]
        ),
    }
)


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

# %%
