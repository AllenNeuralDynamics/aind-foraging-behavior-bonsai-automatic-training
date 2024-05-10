__version__ = "1.1.0"

import logging

def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level,
                        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler('debug_auto_train_manager.log'),
                                logging.StreamHandler()],
                        )


