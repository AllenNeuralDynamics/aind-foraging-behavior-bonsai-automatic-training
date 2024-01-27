""" top level run script """
import threading
import time
import logging

from aind_auto_train import setup_logging
from aind_auto_train.auto_train_manager import DynamicForagingAutoTrainManager
setup_logging()
logger = logging.getLogger(__name__)

def update_auto_train_database(managers, interval=3600):
    while True:
        try:
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

    # create a thread, and run update_auto_train_database() per 1 hour
    update_auto_train_database(managers)
    thread = threading.Thread(update_auto_train_database, args=[managers])
    thread.start()



if __name__ == "__main__":
    run()
