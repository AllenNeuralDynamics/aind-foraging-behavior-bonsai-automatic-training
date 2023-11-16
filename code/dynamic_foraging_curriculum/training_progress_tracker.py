#%%
import pandas as pd
from datetime import datetime

from dynamic_foraging_curriculum.schema.curriculum import add_mouse_tracker, TrainingManager, TrainingStage, Metrics

# Initialize DataFrame
progress_df = pd.DataFrame(columns=['Date', 'MouseID', 'SessionCount', 'TrainingStage'])

# Instantiate TrainingManager
manager = TrainingManager(mice={})

# Add mouse trackers to the manager
manager.mice['Mouse1'] = add_mouse_tracker('Mouse1', TrainingStage.STAGE_1)
# ... Add more mice as needed ...



#%% Example usage
new_metrics = Metrics(foraging_efficiency=0.7, finished_trials=300)
update_progress('Mouse1', 3, new_metrics)
save_progress_to_csv('mouse_training_progress.csv')

# Load the DataFrame for analysis
loaded_progress_df = load_progress_from_csv('mouse_training_progress.csv')
print(loaded_progress_df)
# %%
