__version__ = "0.1"

import logging

logging.basicConfig(level=logging.INFO,
                    # format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[logging.FileHandler("debug_curriculum_manager.log"), logging.StreamHandler()]
                    )