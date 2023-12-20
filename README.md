# Automatic training for the dynamic foraging task

## Overview
<!-- markdown-link-check-disable-next-line -->
Here is a diagram of the [automatic training system](https://github.com/AllenNeuralDynamics/aind-behavior-blog/issues/73) we've been devloping for the dynamic foraging task (those blue arrows on top of our existing [foraging behavior pipeline](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-trigger-pipeline))

<img width="1115" alt="Screenshot 2023-12-04 at 11 11 32" src="https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/assets/24734299/ccea0ea1-9eb0-47dd-b974-4137879b721f">


This repo is the red circle in the diagram. It will be running on Code Ocean and do the following things (very similar to mTrack):
1. Define the training curriculum (mTrack's "Regimen")
2. Retrieve data for each mouse from the behavioral master table (`df_behavior` on S3 that stores session-wise metrics)
3. Evaluate daily performance based on the curriculum and make decisions of the next training stage (stored in `df_manager`)
4. Push the decisions back to S3 by uploading `df_manager`, from which our [python GUI](https://github.com/AllenNeuralDynamics/dynamic-foraging-task) can access and automatically set the training parameters on the next day.

## Key elements
- [Task schema](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/aind_auto_train/schema/task.py) defines schema for training parameters
- [Curriculum schema](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/aind_auto_train/schema/curriculum.py) defines schema for the curriculum, especially the `evaluate_transitions` method
- [Auto train manager](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/aind_auto_train/auto_train_manager.py) fetches data from `df_behavior` and updates `df_manager` (or "tables" on any other database)
- [Curriculum manager](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/aind_auto_train/curriculum_manager.py) manages all available pre-generated curriculums (on any S3 bucket).

## Usage
1. Design the curriculum, i.e., all [training stages](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/aind_auto_train/curriculums/coupled_baiting.py#L199-L204) and [transition rules](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/aind_auto_train/curriculums/coupled_baiting.py#L223-L247).
   - Here is an [example curriculum](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/aind_auto_train/curriculums/coupled_baiting.py) for the dynamic foraging task.
   - Here is automatically generated [json file](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/aind_auto_train/curriculums/Coupled%20Baiting_curriculum_v0.1_schema_v0.2.json) 
   - Here are automatically rendered diagrams for stage transitions rules and parameters (click the images to try the hover feature :blush:)

| rules | parameters |
|--|--|
|<img width="700" src="https://raw.githubusercontent.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/main/code/aind_auto_train/curriculums/Coupled%20Baiting_curriculum_v0.1_schema_v0.2_rules.svg">|<img width="500" src="https://raw.githubusercontent.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/main/code/aind_auto_train/curriculums/Coupled%20Baiting_curriculum_v0.1_schema_v0.2_paras.svg">|

2. Create `AutoTrainManager` and connect it to the behavior database `df_behavior`.
3. Feed all necessary metrics to `Auto train manager` and let it run.
   - Here is an open-loop simulation with our old mice<br>
     <img width="500" src="https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/assets/24734299/885ce4eb-33e9-471b-94e3-bb4fec4d24a8">

4. To add other tasks, users should add their own task and curriculum schemas.

## Demo
- Demo notebook for the [curriculum schema](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/aind_auto_train/demo_schema.ipynb).
- Demo notebook for [a full automation workflow (auto train manager)](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/aind_auto_train/demo_auto_train_manager.ipynb)
- Demo notebook for the [curriculum manager](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/aind_auto_train/demo_curriculum_manager.ipynb)

## Compared with SLIMS/mTrack
<!-- markdown-link-check-disable-next-line -->
See [this thread](https://github.com/AllenNeuralDynamics/aind-behavior-blog/discussions/124)
