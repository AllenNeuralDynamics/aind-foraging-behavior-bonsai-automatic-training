__version__ = "0.2.0"

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s|%(levelname)s|%(name)s|%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler('debug_auto_train_manager.log'),
                              logging.StreamHandler()],
                    )


