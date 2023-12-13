"""
Get data from the behavior master table and give suggestions
"""
# %%
import os
import logging
import re
import glob

import numpy as np
import pandas as pd
from typing import Any, Generic

from aind_auto_train.schema.task import Task
from aind_auto_train.util.aws_util import download_dir_from_s3, upload_dir_to_s3

logger = logging.getLogger(__name__)

# Directory for caching df_maseter tables
LOCAL_SAVED_CURRICULUM_ROOT = os.path.expanduser(
    '~/capsule/scratch/saved_curriculums/')


# %%
class CurriculumManager:
    """ A class to interact with aind-auto-train curriculums
    """

    def __init__(self,
                 saved_curriculums_on_s3: dict = dict(bucket='aind-behavior-data',
                                                      root='foraging_auto_training/saved_curriculums/'),
                 saved_curriculums_local=LOCAL_SAVED_CURRICULUM_ROOT
                 ):

        self.saved_curriculums_on_s3 = saved_curriculums_on_s3
        self.saved_curriculums_local = saved_curriculums_local

    def get_df_curriculums(self) -> pd.DataFrame:
        """ Return the master table of all curriculums
        """
        # Get
        pass

    def generate_curriculums(self):
        pass

    def download_curriculums(self):
        download_dir_from_s3(bucket=self.saved_curriculums_on_s3['bucket'],
                             s3_dir=self.saved_curriculums_on_s3['root'],
                             local_dir=self.saved_curriculums_local)

    def upload_curriculums(self):
        upload_dir_to_s3(bucket=self.saved_curriculums_on_s3['bucket'],
                         s3_dir=self.saved_curriculums_on_s3['root'],
                         local_dir=self.saved_curriculums_local)


if __name__ == "__main__":
    curriculum_manager = CurriculumManager()
    curriculum_manager.download_curriculums()
    curriculum_manager.upload_curriculums()
# %%
