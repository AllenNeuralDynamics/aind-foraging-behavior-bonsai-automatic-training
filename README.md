# Automatic training for the dynamic foraging task

## Overview

Here is a diagram of the [automatic training system](https://github.com/AllenNeuralDynamics/aind-behavior-blog/issues/73) we've been devloping for the dynamic foraging task (those blue arrows on top of our existing [foraging behavior pipeline](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-trigger-pipeline))

<img src="https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/assets/24734299/1a0aded6-c211-4e5d-95e6-f3fa988c2424" width="800">

This repo is the red circle in the diagram. It will be running on Code Ocean and do the following things (very similar to mTrack):
1. Define the training curriculum (mTrack's "Regimen")
2. Retrieve data for each mouse from the behavioral master table (`df_behavior` on S3 that stores session-wise metrics)
3. Evaluate daily performance based on the curriculum and make decisions of the next training stage (stored in `df_manager`)
4. Push the decisions back to S3 by uploading `df_manager`, from which our [python GUI](https://github.com/AllenNeuralDynamics/dynamic-foraging-task) can access and automatically set the training parameters on the next day.

## Key elements
- [Task schema](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/dynamic_foraging_curriculum/schema/task.py) defines schema for training parameters
- [Curriculum schema](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/dynamic_foraging_curriculum/schema/curriculum.py) defines schema for the curriculum, especially the `evaluate_transitions` method
- [Curriculum manager](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/dynamic_foraging_curriculum/automation.py)  (WIP) fetches data from `df_behavior` and updates `df_manager`.

## Usage
1. Design the curriculum, i.e., all [training stages](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/ae4692e523f0dfa14448bdf5693c1338287b4fb1/code/dynamic_foraging_curriculum/curriculums/coupled_baiting.py#L198-L203) and [transition rules](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/ae4692e523f0dfa14448bdf5693c1338287b4fb1/code/dynamic_foraging_curriculum/curriculums/coupled_baiting.py#L222-L246).
   - Here is an [example curriculum](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/dynamic_foraging_curriculum/curriculums/coupled_baiting.py) and automatically generated [json file](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/dynamic_foraging_curriculum/curriculums/curriculum_Coupled%20Baiting_0.1.json) for the dynamic foraging task.
3. Feed all necessary metrics to `Curriculum manager` and let it run.
4. To add other tasks, users should add their own task and curriculum schemas.

## Demo

See [this notebook](https://nbviewer.org/github/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/dynamic_foraging_curriculum/demo.ipynb)

## Compared with mTrack
See [this thread](https://github.com/AllenNeuralDynamics/aind-behavior-blog/discussions/124)
