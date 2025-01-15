# Vision-Based Autonomous Drone Racing

## Description

This project focuses on developing and testing a machine learning solution for vision-based autonomous drone racing. It was completed as part of the *Machine Learning in Robotics* course at ITMO University. The project involves creating policies that enable a quadrotor to navigate a race track using onboard sensors and vision data.

The following code is included in the project:

- `feature_extractor_training.ipynb`: Trains the feature extractor, which processes images from the quadrotor’s camera into informative embeddings for control tasks.

- `train_RL.py`: Trains a state-based policy using reinforcement learning (RL), which serves as the expert model. The RL policy takes full state information, including the quadrotor's position and orientation, to output control actions.

- `train_imitation.py`: Trains a vision- and IMU-based policy using imitation learning. The policy learns to mimic the state-based expert by mapping image embeddings and IMU data to control actions.

- `test_RL.py`: Tests the expert policy trained via reinforcement learning.

- `test_imitation.py`: Tests the vision- and IMU-based policy trained via imitation learning.

Additionally, in `/src`, a PyBullet-based simulation environment for the quadrotor is implemented. This environment is used to train and test both the state-based expert and the vision-based student models. The simulation includes a race track with gates that the quadrotor must navigate while optimizing its trajectory and speed.

## Installation

### Using Docker

A `Dockerfile` is provided for easy setup of the project environment.

1. **Build the Docker Image**:
   ```bash
   docker build -t drone-racing .
   ```

2. **Run the Container**:
   ```bash
   docker run -it --rm -v $(pwd):/workspace drone-racing
   ```
### Download Required Files

1. Download the pre-trained model weights and simulation files from the provided link:
   [Download Here](https://drive.google.com/drive/folders/1omX9G3uqKUYzehZbuZeSXJUrRmWLwa3V?usp=sharing)
2. Extract the downloaded files.
3. Place the extracted model weights in the `results/models` folder.
4. Place the simulation files a folder named `world`

### Testing

To test the final results, run the `test_imitation.py` script with the following options inside the container:
- `-ls` to specify the student model weights
- `-lt` to specify the teacher (expert) model weights

## Process Overview

1. **Feature Extraction**: The feature extractor processes images from the quadrotor’s onboard camera to generate embeddings that represent its position relative to the race track. These embeddings are learned using a contrastive learning approach to ensure they are both informative and robust to environmental variations.

2. **Reinforcement Learning (RL) Expert**: A state-based policy is trained using RL to act as an expert. This policy has access to the quadrotor’s full state and optimizes its trajectory based on predefined reward functions.

3. **Imitation Learning (IL) Student**: A vision- and IMU-based policy is trained to imitate the expert. This policy uses image embeddings and IMU measurements as inputs to predict control actions.


This project demonstrates the integration of reinforcement learning, contrastive learning, and imitation learning for autonomous drone racing using vision and sensor data.
