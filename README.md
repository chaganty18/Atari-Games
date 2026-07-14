# Deep Q-Learning with Atari Environments

This project implements a Deep Q-Learning (DQN) agent to play Atari games using the Gymnasium environment. The agent is trained using experience replay, frame stacking, and a convolutional neural network (CNN) architecture for approximating Q-values. The project provides logs of training performance, evaluations, and model saving. Additionally, the best episodes are visualized and saved as videos.

## Installation

### 1. Install Python 3.11.12

#### On macOS (via Homebrew):
In the terminal running the below command:
```
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"


# Install Python 3.11
brew install python@3.11

# Link it so python3.11 is available in your shell
brew link --force --overwrite python@3.11

# Confirm installation
python3.11 --version  # Should return Python 3.11.12
```

### 2. Download the code:
Download the code and make sure you are inside the root folder for your project where `atarigames.py` and `requirements.txt` files are located.

### 3. Install virtualenv
```python3.11 -m venv dqn_env```

### Activate the environment
```source dqn_env/bin/activate   # On macOS/Linux```

### Command for deactivating the environment:
```deactivate```

To run this project, you need to install the following dependencies:

1. **gymnasium** - For the Atari environments.
2. **tensorflow** - For building and training the neural network model.
3. **numpy** - For numerical operations.
4. **opencv** - For preprocessing frames.
5. **matplotlib** - For plotting results.
6. **moviepy** - For saving episode frames as a video.

Use the following command to install the necessary dependencies(make sure that you are installing the dependencies in virtualenv):

```
pip install -r requirements.txt
```

## Prerequisites:
Python version 3.11.12 or greater should be installed


### 4. Dataset: Atari Games from ALE
This project uses Atari 2600 games as environments for training the reinforcement learning agent. The environments are provided via:

Gymnasium – the successor of OpenAI Gym

Arcade Learning Environment (ALE) – the backend emulator used by Gymnasium for Atari environments

Key Notes:
No external dataset download is required. Gymnasium will automatically download and cache the ROMs the first time you run the environment.

The license for Atari ROMs is accepted using the accept-rom-license option in the requirements.

### 5. Configuration:
The agent is configured with the following default hyperparameters:

| Hyperparameter        | Value         |
|----------------------|---------------|
| gamma              | 0.99          |
| epsilon_start      | 1.0           |
| epsilon_min        | 0.1           |
| epsilon_decay      | 0.999         |
| learning_rate      | 0.00025       |
| batch_size         | 64            |
| memory_size        | 100,000       |
| target_update_freq | 10,000 steps  |
| eval_freq          | 100,000 steps |
| frame_stack_size   | 4             |
| episodes           | 100           |

Configuration can be modified directly in `atarigames.py`.

### 6. Run code:
```
python atarigames.py
```

We can run the list of all available Atari 2600 games environmets by running `environments.py` script

### 7. Experimental Results:
The experimental results & plots are included in `training_logs/ALE/<env_name>` folder

### The project includes a .ipynb notebook version of the script to:

- Run and debug the DQN training interactively

- View training graphs inline

- Document outputs and observations as a report

The license for Atari ROMs is accepted using the accept-rom-license option in the requirements.
