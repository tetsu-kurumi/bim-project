# CPSC-459/559 Shuttographer README

## Contributions

- Tetsu Kurumisawa

Worked on the OpenCV approach in imitation learning, identifying problems in the plan we had for the and updated the plan/testing the Shutter environment accordingly. Once we decied not to use the RL stream, focused on implementing the imitation learning model. Collected data of demonstrations by humans to frame a subject, processed data to construct state action pairs, and trained the imitation learning model. Helped work on the implementation of the pipeline onto Shutter in the lab. Filmed and edited the demo video. Sent video & survey form to friends to qualitatively assess project. 

- James Rosen

Worked on Shuttographer's core interactivity, including detecting subjects in the Kinect camera frame, asking participants for consent and processing their responses. Also worked on extracting frames from Shutter's camera for processing by the portrait evaluation model, and implemented the Stable Diffusion pipeline. Integrated all components of Shuttographer together. Also acted as the photography subject for data collection in the imitation learning phase. 

- Eason Ding

Worked on implementing face tracking algorithm, extracting body joints information from Kinect camera, designing the finite state machine for face tracking component, creating the static Transformation broadcaster from Kinect to Shutter, writing the launch file that runs the whole pipline.

- Bhavya Kasera

Worked on the Portrait Evaluation Model - did literature review and picked new dataset after we decided the original dataset would not work for the project. Setup and preprocessed dataset to prep for training. Trained and tested the portrait evaluation model, and created interface for integration with the rest of the pipeline. Also worked with Eason on integrating the imitation learning model with the rest of the pipeline.

## Dependencies
- Main Driver

    [Azure Kinect ROS Driver](https://github.com/microsoft/Azure_Kinect_ROS_Driver/tree/melodic)

    [Shutter-ROS](https://gitlab.com/interactive-machines/shutter/shutter-ros)

- Imitation Learning

    [numpy](https://numpy.org/)

    [Tensorflow](https://github.com/tensorflow/tensorflow)

    [Keras](https://github.com/keras-team/keras)

    [scikit-learn](https://github.com/scikit-learn/scikit-learn)

- Portrait Evaluation

    [numpy](https://numpy.org/)

    [torch](https://pytorch.org/docs/stable/torch.html)

    [torchvision](https://pytorch.org/vision/stable/index.html)

    [pandas](https://pandas.pydata.org/)

    [tqdm](https://github.com/tqdm/tqdm)

    [gdown](https://github.com/wkentaro/gdown)

## How to run the code -Main Driver-
- This runs the entire system to take portrait photographs. The individual components can also be run/trained/tested in isolation, as described in the next few sections.

```bash
$ roslaunch shuttographer shuttographer.launch
```

## File Descriptions

`shuttographer.launch` is the launch file to launch all the nodes. 

`audio_files` contains the audio for shutter's responses

`portrait_evaluation` contains codes for training and using protrait evaluation model (See below for details on how to train and test the portrait evaluation model)

`imitation_learning` contains codes for training imitation learning model (See below for details on how to run the imitation learning stream)

`transform_msg.py` creates a Static Transform Broadcaster that publishing the transformation from Kinect `camera_base` to Shutter `base_footprint`

`face_tracking.py` recieves information from topics `/body_tracking_data` and `/joint_states`, and publishes final Shutter position to topic `/joint_group_controller/command`. The final Shutter position is calculated by geometry, and the formatting of the file is greatly inspired by the code provided in [assignment-2](https://github.com/Yale-BIM/f23-assignments/blob/master/assignment-2/shutter_behavior_cloning/src/expert_opt.py)

`new_person_detector.py` recieves information from topic `/body_tracking_data` and publishes a Bool logic to topic `/new_person` that indicates if a new person is detected by Kinect camera.

`consent.py` recieves information from topic `/new_person` and publishes a Bool logic to `/record` that start the recoding of user's feedback.

`record_consent.py` recieves information from topic `/record` and `/audio/audio` and publishes a Bool logic to topic `/consent_confirmed` that indicates if the user wants Shutter to take a picture.

`record_prompt.py` recieves information from topic `/consent_confirmed` and `/audio/audio` and publishes a String to topic `/prompt` that contains the user's prompt to edit the photo.

`photo_editing.py` recieves information from topic `/prompt` and takes 10 photos over a period of time. Then, it evaluates the photos by the `portrait_evaluation` interface and chooses the best photo to be edited by stable diffusion according to user's prompt. All the photos and edited photo are saved into dictionary `IMG_DIRECTORY`.

`helpers.py` contains helper functions to convert speech to text, access chat gpt to check if user consents to take photos, use stable diffusion to edit the chosen photo.

`model` is the trained behavior cloning model for face tracking. Due to limited GPU memory, we cannot run this model along with all other models.
 
## How to run the code -Imitation Learning-

[Data Collection]

- Run Kinect body tracker (Main Driver dependencies need to be installed)
     ```bash
    $ roslaunch azure_kinect_ros_driver driver_with_bodytracking.launch
    ```

- Run Shutter teleoperation
     ```bash
    $ roslaunch shutter_teleop shutter_controller.launch simulation:=false
    ```

- Run Shutter realsense2 camera
     ```bash
    $ roslaunch realsense2_camera rs_camera.launch
    ```

- To collect joint state data into an output file "joint_states.txt" while demonstration

    ```bash
    $ rostopic echo /joint_states >> joint_states.txt
    ```

- To collect Kinect's body tracking data into an output file "body_tracking_data.txt" while demonstration

    ```bash
    $ rostopic echo /body_tracking_data >> body_tracking_data.txt
    ```

[Data Processing]
- To convert the txt files obtained in the [Data Collection] phase into a state action pair dictionary
    ```bash
    $ python3 construct_state_action_pair.py <joint positions txt file> <body tracking txt file> <output json file>
    ```
- The preprocessed dataset can be found and downloaded at the following link - https://drive.google.com/file/d/1A03Kv65b3XdSukEFiKSMytd55Q86Ytni/view?usp=sharing

[Train model and save]
- To train a deep neural network model and save it (change the file paths in behavior_cloning.py to appropriate paths when running)

    ```bash
    $ python3 behavior_cloning.py
    ```

[Trained model]
- The trained model can be downloaded from the following link - https://drive.google.com/file/d/1oDx2ZI67yGIw8qYUnqswkJpYlmQLVEJS/view?usp=sharing

## How to run the code -Portrait Evaluation-

[Dataset]
- The preprocessed dataset from the PIQ23 dataset can be found and downloaded at the following link - https://drive.google.com/drive/folders/1Y--YarAXQ50SZWDG647ub-XLYVC2B-p_?usp=drive_link

[Training]
- To train & save the portrait evaluation model, in train.py and utils.py change the path as directed by comments beginning with 'CHANGE PATH' and run the following -
 
    ```bash
    $ python3 train.py
    ```
[Testing]
- To test the portrait evaluation model, in testing.py change the path as directed by 'CHANGE PATH' comments and run the following -

    ```bash
    $ python3 testing.py
    ```
[Trained model]
- The trained model can be downloaded from the following link - https://drive.google.com/file/d/1oQKuKN5N33KfSMptlNuNHWqmkrub1SAs/view?usp=drive_link
