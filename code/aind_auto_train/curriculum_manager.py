"""
Get data from the behavior master table and give suggestions
"""
# %%
import os
import logging
import re
import glob
import json
import importlib
import inspect

import numpy as np
import pandas as pd
from typing import Any, Generic

from aind_auto_train.schema.task import Task
from aind_auto_train.util.aws_util import download_dir_from_s3, upload_dir_to_s3
import aind_auto_train.schema.curriculum as curriculum_schemas
import aind_auto_train.schema.task as task_schemas

logger = logging.getLogger(__name__)

# # Directory for caching df_maseter tables
# LOCAL_SAVED_CURRICULUM_ROOT = os.path.expanduser(
#     '~\aind-foraging-behavior-bonsai-automatic-training\scratch\saved_curriculums/')
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
        self.download_curriculums()

    def df_curriculums(self) -> pd.DataFrame:
        """ Return the master table of all curriculums
        """

        df_curriculums = pd.DataFrame(columns=[
                                      'curriculum_name',
                                      'curriculum_version',
                                      'curriculum_schema_version',
                                      'curriculum_description'])
        
        schema_version_code_base = curriculum_schemas.Curriculum.model_fields['curriculum_schema_version'].default
        
        for f in self.json_files:
            match = re.search(r'(.+)_curriculum_v(.+)_schema_v(.+)\.json',
                              os.path.basename(f))
            if match is None:
                logger.warning(
                    f"Could not parse {os.path.basename(f)} as a curriculum json file.")
                continue
            curriculum_name, curriculum_version, curriculum_schema_version = match.groups()
            
            # Only show curriculums whose curriculum_schema_version matches the current codebase
            if schema_version_code_base != curriculum_schema_version:
                continue
            
            df_curriculums = pd.concat(
                [df_curriculums,
                 pd.DataFrame.from_records([dict(curriculum_name=curriculum_name,
                                                 curriculum_version=curriculum_version,
                                                 curriculum_schema_version=curriculum_schema_version,
                                                 curriculum_description=json.load(
                                                     open(f, 'r'))['curriculum_description']
                                                 )]
                                           )
                 ], ignore_index=True)

        return df_curriculums

    def get_curriculum(self,
                       curriculum_name: Task,
                       curriculum_schema_version: str,
                       curriculum_version: str
                       ) -> dict:
        """ Get a curriculum from the saved_curriculums directory"""

        json_name = (f"{curriculum_name}_"
                     f"curriculum_v{curriculum_version}_"
                     f"schema_v{curriculum_schema_version}.json")

        # Load json
        try:
            with open(self.saved_curriculums_local + json_name, 'r') as f:
                loaded_json = json.load(f)
        except FileNotFoundError:
            logger.error(
                f"Could not find {json_name} in {self.saved_curriculums_local}")
            return None

        # Sanity check
        assert loaded_json[
            'curriculum_name'] == curriculum_name, f"curriculum_name in json ({loaded_json['curriculum_name']}) does not match file name ({curriculum_name})!"
        assert loaded_json['curriculum_schema_version'] == curriculum_schema_version, \
            f"curriculum_schema_version in json ({loaded_json['curriculum_schema_version']}) does not match file name ({curriculum_schema_version})!"
        assert loaded_json['curriculum_version'] == curriculum_version, \
            f"curriculum_version in json ({loaded_json['curriculum_version']}) does not match file name ({curriculum_version})!"

        # Retrieve the curriculum schema
        curriculum_schema_name = loaded_json['curriculum_schema_name']

        # Check whether the required curriculum schema is available
        assert hasattr(curriculum_schemas, curriculum_schema_name), \
            f"'{curriculum_schema_name}' not found in aind_auto_train.schema.curriculum"

        curriculum_schema = getattr(curriculum_schemas, curriculum_schema_name)

        # Check the schema version
        schema_version = curriculum_schema.model_fields['curriculum_schema_version'].default
        
        if loaded_json['curriculum_schema_version'] != schema_version:
            logger.error(f"Schema version in the loaded json ({loaded_json['curriculum_schema_version']}) does not match the loaded schema ({schema_version})! "
                         f"You're either using an outdated `aind_auto_train` repo or loading an outdated curriculum!")
            return None

        # Create the curriculum object
        curriculum = curriculum_schema(**loaded_json)
        logger.info(
            f"Loaded a {curriculum_schema_name} model from '{json_name}'.")

        # Get the metrics_class name from the typehint of curriculum.evaluate_transitions
        metrics_schema = inspect.getfullargspec(
            curriculum.evaluate_transitions
        ).annotations.get('metrics', None)

        metrics_schema_name = metrics_schema.__name__ if metrics_schema else None

        # Check whether the required metrics schema is available
        assert hasattr(task_schemas, metrics_schema_name), \
            f"'{metrics_schema_name}' not found in aind_auto_train.schema.task"

        metrics = getattr(task_schemas, metrics_schema_name)

        return {'curriculum': curriculum,
                'metrics': metrics,
                'curriculum_json_name': self.saved_curriculums_local + json_name,
                'diagram_paras_name':  self.saved_curriculums_local + json_name.replace('.json', '_paras.svg'),
                'diagram_rules_name': self.saved_curriculums_local + json_name.replace('.json', '_rules.svg'),
                }

    def download_curriculums(self):
        download_dir_from_s3(bucket=self.saved_curriculums_on_s3['bucket'],
                             s3_dir=self.saved_curriculums_on_s3['root'],
                             local_dir=self.saved_curriculums_local)

        self.json_files = glob.glob(
            self.saved_curriculums_local + '/*_curriculum_*.json')

        logger.info(
            f"Found {len(self.json_files)} curriculums in {self.saved_curriculums_local}")

    def upload_curriculums(self):
        upload_dir_to_s3(bucket=self.saved_curriculums_on_s3['bucket'],
                         s3_dir=self.saved_curriculums_on_s3['root'],
                         local_dir=self.saved_curriculums_local)


if __name__ == "__main__":
    from aind_auto_train import setup_logging
    setup_logging()

    curriculum_manager = CurriculumManager()
    
    logger.info(curriculum_manager.df_curriculums())
    _curr = curriculum_manager.get_curriculum(
        curriculum_name='Coupled Baiting',
        curriculum_version='1.0',
        curriculum_schema_version='1.0',
    )

    print(_curr)
    # _curr['curriculum'].diagram_rules(render_file_format='svg')
    curriculum_manager.upload_curriculums()
# %%
