""" Generate schema diagrams using erdantic
Install erdantic here https://erdantic.drivendata.org/v0.6/
"""

import erdantic as erd
from aind_auto_train.schema.curriculum import Curriculum, DynamicForagingCurriculum, DummyTaskCurriculum

erd.draw(DynamicForagingCurriculum,
         out="./code/aind_auto_train/schema/schema_diagram_DynamicForagingCurriculum.png")

erd.draw(DummyTaskCurriculum,
         out="./code/aind_auto_train/schema/schema_diagram_DummyTaskCurriculum.png")

erd.draw(Curriculum,
         out="./code/aind_auto_train/schema/schema_diagram_Curriculum.png")


