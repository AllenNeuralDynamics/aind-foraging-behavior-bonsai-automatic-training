#%%
import pandas as pd
from curriculum import add_mouse_tracker, TrainingManager, TrainingStages, BehavioralMetrics
from datetime import datetime

# Initialize DataFrame
progress_df = pd.DataFrame(columns=['Date', 'MouseID', 'SessionCount', 'TrainingStage'])

# Instantiate TrainingManager
manager = TrainingManager(mice={})

# Add mouse trackers to the manager
manager.mice['Mouse1'] = add_mouse_tracker('Mouse1', TrainingStages.STAGE_1)
# ... Add more mice as needed ...

# Function to update progress for a mouse
def update_progress(mouse_id: str, session_count: int, metrics: BehavioralMetrics):
    # Update the tracker's metrics
    manager.mice[mouse_id].metrics = metrics
    manager.update()
    
    # Get the current stage
    current_stage = manager.mice[mouse_id].current_stage
    
    # Log the progress in the DataFrame
    today = datetime.now().strftime('%Y-%m-%d')
    progress_df.loc[len(progress_df)] = [today, mouse_id, session_count, current_stage.name]

# Function to save progress to a CSV
def save_progress_to_csv(filepath: str):
    progress_df.to_csv(filepath, index=False)

# Function to load progress from a CSV
def load_progress_from_csv(filepath: str):
    return pd.read_csv(filepath)

#%% Example usage
new_metrics = BehavioralMetrics(foraging_efficiency=0.7, finished_trials=300)
update_progress('Mouse1', 3, new_metrics)
save_progress_to_csv('mouse_training_progress.csv')

# Load the DataFrame for analysis
loaded_progress_df = load_progress_from_csv('mouse_training_progress.csv')
print(loaded_progress_df)
# %%
