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
1. Design the curriculum, i.e., all [training stages](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/e04c3206d2d1ed2b59ff768d17b50d8bdc0b6d14/code/aind_auto_train/curriculums/coupled_baiting_1p0.py#L378-L385) and [transition rules](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/e04c3206d2d1ed2b59ff768d17b50d8bdc0b6d14/code/aind_auto_train/curriculums/coupled_baiting_1p0.py#L387-L393).
   - Here is an [example curriculum](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/aind_auto_train/curriculums/coupled_baiting_1p0.py#L387-L393) for the dynamic foraging task.
   - Here is automatically generated [json file](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/e04c3206d2d1ed2b59ff768d17b50d8bdc0b6d14/code/aind_auto_train/curriculums/Uncoupled%20Baiting_curriculum_v1.0_schema_v1.0.json) 
   - Here are automatically rendered diagrams for stage transitions rules and parameters (click the images to try the hover feature :blush:)

| rules | parameters |
|--|--|
|<img width="400" src="https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/e04c3206d2d1ed2b59ff768d17b50d8bdc0b6d14/code/aind_auto_train/curriculums/Uncoupled%20Baiting_curriculum_v1.0_schema_v1.0_rules.svg">|<img width="500" src="https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/e04c3206d2d1ed2b59ff768d17b50d8bdc0b6d14/code/aind_auto_train/curriculums/Uncoupled%20Baiting_curriculum_v1.0_schema_v1.0_paras.svg">|

2. Create `AutoTrainManager` and connect it to the behavior database `df_behavior`.
3. Feed all necessary metrics to `Auto train manager` and let it run.
   - Here is an open-loop simulation with our old mice<br>
     <img width="500" src="https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/assets/24734299/885ce4eb-33e9-471b-94e3-bb4fec4d24a8">

4. To add other tasks, users should add their own task and curriculum schemas.

## Demo
- Demo notebook for the [curriculum schema](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/aind_auto_train/demo_schema.ipynb).
- Demo notebook for [a full automation workflow (auto train manager)](https://nbviewer.org/github/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/aind_auto_train/demo_auto_train_manager.ipynb)
- Demo notebook for the [curriculum manager](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training/blob/main/code/aind_auto_train/demo_curriculum_manager.ipynb)

## Notes
System upgrade checklist
- Upgrade all training rigs
- Upgrade CO capsule (terminate all running workers and start a new one)
- Upgrade streamlit app

## Compared with SLIMS/mTrack
<!-- markdown-link-check-disable-next-line -->
See [this thread](https://github.com/AllenNeuralDynamics/aind-behavior-blog/discussions/124)

## Instructions for adding new curriculums
### Do it in Code Ocean (recommended)
1. Duplicate the capsule [`foraging-behavior-bonsai-automatic-training_debug`](https://codeocean.allenneuraldynamics.org/capsule/3812636/tree)
2. Start the Code Server (VS Code)
3. Test the environment by running a demo script `code/aind_auto_train/curriculums/dummy_task.py`
   <img width="400" height="715" alt="image" src="https://github.com/user-attachments/assets/10f93340-77e8-4192-8d20-4f69c400d1a3" />
   
   You should see outputs like this
   ```
   (base) root@ffc34097cd8a:~/capsule# /opt/conda/bin/python /root/capsule/code/aind_auto_train/curriculums/dummy_task.py
   2025-12-11 23:23:44 | INFO | aind_auto_train.schema.curriculum | Curriculum saved to /root/capsule/scratch/saved_curriculums//Dummy task_curriculum_v0.1_schema_v1.0.json
   2025-12-11 23:23:44 | INFO | aind_auto_train.schema.curriculum | Curriculum schema saved to /root/capsule/scratch/saved_curriculums//DummyTaskCurriculum_v1.0.json
   2025-12-11 23:23:44 | INFO | aind_auto_train.schema.curriculum | Curriculum rules diagram saved to /root/capsule/scratch/saved_curriculums//Dummy task_curriculum_v0.1_schema_v1.0_rules
   2025-12-11 23:23:44 | INFO | aind_auto_train.schema.curriculum | Curriculum parameters diagram saved to /root/capsule/scratch/saved_curriculums//Dummy task_curriculum_v0.1_schema_v1.0_paras
   ```
   and six new files under `scratch/saved_curriculums`
   
   <img width="300" height="197" alt="image" src="https://github.com/user-attachments/assets/dd3c88d6-6876-41a8-aee9-e6b315456abc" />
   
   Choose one of the svg file and click the "Preview svg" <img width="150" height="80" alt="image" src="https://github.com/user-attachments/assets/037a2790-10a2-4912-9aa9-a3b4e217577b" />, you should see the curriculum diagram
   
   <img width="800" height="911" alt="image" src="https://github.com/user-attachments/assets/2ae0d54d-82a4-4e28-82fe-e3531de44320" />

4. On your own branch, create you curriculum based on existing curriculums in `code/aind_auto_train/curriculums`
     - If there are new task parameters, you should add them to the `DynamicForagingParas` class in `schema/task.py` with correct default values for backward compatibility.
6. Run your new .py file and check the generated diagrams under `scratch/saved_curriculums`
7. Run the script `code/aind_auto_train/curriculum_manager.py` to upload new curriculum files to AWS S3 bucket `s3://aind-behavior-data/foraging_auto_training/saved_curriculums/`
     - If this fails due to credential issues, just send the six generated files to Han.
8. Issue a PR to the this repo for review
    - Remember to manually bump the version of the repo in `aind_auto_train/__init__.py`
9. Check the new curriculum on the [Streamlit app](https://foraging-behavior-browser.allenneuraldynamics-test.org/?tab_id=tab_auto_train_curriculum).
    - Han may need to update the aind_auto_train in Streamlit
11. On the GUI side, check if you can load the new curriculum and trigger a new session
    - You may need to update the aind_auto_train version as well
12. Let Han know and he will update the version in the auto_train service capsule.

---

### Do it locally
Since `graphviz` in Code Ocean has some unsolved bug, it is recommended to install the library locally in a conda environment to create new curriculums.
1. Clone this repo to your local computer
2. Create conda environment by
   ```python
   conda create -n autotrain python=3.8
   conda activate autotrain
   ```
3. Install the library in editable mode
   ```shell
   pip install -e .
   ```
4. Install `graphviz` via conda (which installs necessary `dot` .bin files for you)
   ```shell
   conda install python-graphviz
   ```
5. Set up AWS credential. (ask Han for this step)
6. Test the installation by running demo script `code\aind_auto_train\curriculums\dummy_task.py`
7. Create you own curriculum based on existing curriculums in `code\aind_auto_train\curriculums\`
8. Run your new .py file and check the generated diagrams under the default folder `{your windows user folder}/capsule/scratch/saved_curriculums/`
9. Run the script `code\aind_auto_train\curriculum_manager.py` to upload new curriculum files to AWS S3 bucket `s3://aind-behavior-data/foraging_auto_training/saved_curriculums/`
10. Check the new curriculum on the [Streamlit app](https://foraging-behavior-browser.allenneuraldynamics-test.org/?tab_id=tab_auto_train_curriculum).
