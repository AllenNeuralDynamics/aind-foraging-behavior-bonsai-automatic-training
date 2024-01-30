"""A simple demo for adding a curriculum for a dummy task
"""
# %%
from typing import List, Dict

from aind_auto_train.curriculum_manager import LOCAL_SAVED_CURRICULUM_ROOT
from aind_auto_train.schema.task import Task, TrainingStage, DummyTaskParas, DummyTaskMetrics
from aind_auto_train.schema.curriculum import DummyTaskCurriculum, StageTransitions, TransitionRule, Decision


meta = dict(task_schema_version="0.1",
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
    training_stage=TrainingStage.STAGE_2,  # "Phase B" in Han's slides
    description="Dummy stage interesting",

    dummy_para_bool=True,
    dummy_para_float=0.2
)

paras_stage_3 = DummyTaskParas(
    # Metainfo
    **meta,
    training_stage=TrainingStage.STAGE_3,  # "Phase B" in Han's slides
    description="Dummy stage interesting",

    dummy_para_bool=False,
    dummy_para_float=0.0
)


paras_stage_final = DummyTaskParas(
    # Metainfo
    **meta,
    training_stage=TrainingStage.STAGE_FINAL,  # "Phase B" in Han's slides
    description="Dummy stage final",

    dummy_para_bool=False,
    dummy_para_float=0.2
)



curriculum = DummyTaskCurriculum(
    curriculum_name=Task.DUMMY,
    curriculum_version="0.2",
    curriculum_description = '''A dummy curriculum showing complex transitions and no absorbing GRADUATED stage''',

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
                    to_stage=TrainingStage.STAGE_FINAL,
                    condition_description="metric float > 1.0 and metric int > 10",
                    condition="""lambda metrics: 
                                metrics.dummy_metric_float[-1] > 1.0
                                and 
                                metrics.dummy_metric_int[-1] > 10
                            """
                ),
                TransitionRule(
                    decision=Decision.PROGRESS,
                    to_stage=TrainingStage.STAGE_2,
                    condition_description="metric float > 0.5 and metric int > 5",
                    condition="""lambda metrics: 
                                metrics.dummy_metric_float[-1] > 0.5 
                                and 
                                metrics.dummy_metric_int[-1] > 5
                            """
                )

            ]
        ),


        TrainingStage.STAGE_2: StageTransitions(
            from_stage=TrainingStage.STAGE_2,
            transition_rules=[
                TransitionRule(
                    decision=Decision.PROGRESS,
                    to_stage=TrainingStage.STAGE_FINAL,
                    condition_description="metric float > 1.0 and metric int > 10",
                    condition="""lambda metrics: 
                                metrics.dummy_metric_float[-1] > 1.0
                                and 
                                metrics.dummy_metric_int[-1] > 10
                            """
                ),
                TransitionRule(
                    decision=Decision.PROGRESS,
                    to_stage=TrainingStage.STAGE_3,
                    condition_description="metric float > 0.6 and metric int > 7",
                    condition="""lambda metrics: 
                                metrics.dummy_metric_float[-1] > 0.6 
                                and 
                                metrics.dummy_metric_int[-1] > 7
                            """
                )
            ]
        ),
        
        TrainingStage.STAGE_3: StageTransitions(
            from_stage=TrainingStage.STAGE_3,
            transition_rules=[
                TransitionRule(
                    decision=Decision.PROGRESS,
                    to_stage=TrainingStage.STAGE_FINAL,
                    condition_description="metric float > 1.0 and metric int > 10",
                    condition="""lambda metrics: 
                                metrics.dummy_metric_float[-1] > 1.0
                                and 
                                metrics.dummy_metric_int[-1] > 10
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


        TrainingStage.STAGE_FINAL: StageTransitions(
            from_stage=TrainingStage.STAGE_2,
            transition_rules=[

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
