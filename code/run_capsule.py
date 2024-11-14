""" top level run script """
import threading
import time
import logging

from aind_auto_train import setup_logging, __version__
from aind_auto_train.auto_train_manager import DynamicForagingAutoTrainManager
from aind_auto_train.curriculum_manager import CurriculumManager, LOCAL_SAVED_CURRICULUM_ROOT

setup_logging()
logger = logging.getLogger(__name__)

def update_auto_train_database(managers, curriculum_manager, interval=3600):
    # Jon helped me trigger this capsule 10 PM each day by airflow.
    # So now there is no need to keep this capsule running.
    # But since the processing pipeline takes some time, we should run this one longer.
    # Starting from 10 PM, let's update the curriculum manager once per 60 min for 10 times (until 7 AM in the morning)

    RUN_TIMES = 10  # Max pipeline run time for each run
    for i in range(RUN_TIMES):
        logger.info(f'\n\n --- v{__version__} ---')
        try:
            logger.info(f'-- Update curriculums --')
            curriculum_manager.download_curriculums()
            logger.info(f'-- Update training manager --')
            for manager_name, manager in managers.items():
                manager.update()
        except Exception as e:
             logger.error(e)
        time.sleep(interval)

def run():
    # Connect to databases
    managers = {}
    managers['447_demo'] = DynamicForagingAutoTrainManager(
        manager_name='447_demo',
        df_behavior_on_s3=dict(bucket='aind-behavior-data',
                               root='foraging_nwb_bonsai_processed/',
                               file_name='df_sessions.pkl'),
        df_manager_root_on_s3=dict(bucket='aind-behavior-data',
                                   root='foraging_auto_training/'),
        if_rerun_all=False
    )

    curriculum_manager = CurriculumManager(
        saved_curriculums_on_s3=dict(bucket='aind-behavior-data',
                                     root='foraging_auto_training/saved_curriculums/'),
        saved_curriculums_local=LOCAL_SAVED_CURRICULUM_ROOT
    )

    # create a thread, and run update_auto_train_database() per 1 hour
    update_auto_train_database(managers, curriculum_manager)



if __name__ == "__main__":
    run()
