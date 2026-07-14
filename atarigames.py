# -*- coding: utf-8 -*-

import gymnasium as gym
import ale_py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Input
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import time
import csv
import cv2
import imageio
from PIL import Image, ImageDraw
from moviepy import ImageSequenceClip

# Hyperparameters
gamma = 0.99
epsilon = 1.0  # Exploration factor
epsilon_min = 0.1
epsilon_decay = 0.999  # Slower decay for more exploration
learning_rate = 0.00025
batch_size = 64
max_memory_size = 100000
target_update_freq = 10000  # Update target network every 10,000 steps
n_episodes = 100
frame_stack_size = 4  # Number of frames to stack
eval_freq = 100000  # Evaluate every 100,000 frames

def save_logs(log_file, episode, total_reward, episode_loss, epsilon, avg_q_value_episode):
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode, total_reward, episode_loss, epsilon, avg_q_value_episode])

# Function to save the model weights
def save_model(model, save_path, env_name):
    model.save_weights(save_path)
    print(f"Model saved to {save_path} for environment {env_name}")

# Reset the environment and prepare for visualization
def reset_environment(env):
    """
    Resets the environment and returns the initial observation.
    """
    obs, _ = env.reset()  # Reset environment to start a new episode
    return obs

# Neural Network (Q-network)
def create_q_model(input_shape, n_actions):
    model = tf.keras.Sequential([
        Input(shape=input_shape),
        layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
        layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(n_actions)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)      # Convert to grayscale
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)  # Resize to 84x84
    return frame / 255.0  # Normalize to [0, 1]

# Frame stacking (to capture temporal dynamics)
class FrameStack:
    def __init__(self, stack_size=4):
        self.stack = deque(maxlen=stack_size)

    def add(self, frame):
        self.stack.append(frame)

    def get_state(self):
        return np.stack(self.stack, axis=-1)  # Shape: (80, 80, 4)

# Rolling average for rewards to track improvements
def compute_rolling_average(values, window_size=10):
    return np.convolve(values, np.ones(window_size)/window_size, mode='valid')

# Experience replay buffer and frame stack
replay_buffer = ReplayBuffer(max_memory_size)
frame_stack = FrameStack(frame_stack_size)

# Performance tracking
episode_rewards = []  # List to track rewards for plotting
losses = []  # Track the loss during training
epsilon_values = []  # Track epsilon decay

# Function to evaluate the agent
def evaluate_agent(env, model, num_episodes=10):
    total_reward = 0
    for _ in range(num_episodes):
        state, _ = env.reset()  # Reset the environment
        state = preprocess_frame(state)  # Preprocess frame
        for _ in range(frame_stack_size):
            frame_stack.add(state)
        state = frame_stack.get_state()
        done = False
        while not done:
            # Choose action without exploration (epsilon=0)
            q_values = model.predict(np.expand_dims(state, axis=0))
            action = np.argmax(q_values[0])

            # Take action and observe next state
            next_state, reward, done, _, _ = env.step(action)
            next_state = preprocess_frame(next_state)
            frame_stack.add(next_state)
            next_state = frame_stack.get_state()

            state = next_state
            total_reward += reward

    average_reward = total_reward / num_episodes
    return average_reward

def train_agent(env, q_model, target_model, replay_buffer, frame_stack, log_dir,
                n_episodes=50, save_interval=1000, log_file=None):
    global epsilon

    # Use the environment's name for the log file
    if log_file is None:
        log_file = os.path.join(log_dir, f'{env.spec.id}_training_logs.csv')

    # Ensure the directory exists (create any missing directories)
    log_dir_path = os.path.dirname(log_file)
    os.makedirs(log_dir_path, exist_ok=True)

    # Create log file and write header if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Total Reward', 'Episode Loss', 'Epsilon', 'Avg Q Value'])

    total_frames = 0
    best_reward = -float('inf')
    best_episode_path = []

    episode_rewards, losses, epsilon_values = [], [], []
    avg_q_values_per_episode = []
    frame_q_values, frame_numbers = [], []

    n_actions = env.action_space.n
    print(f"[{env.spec.id}] Number of Actions: {n_actions}")

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        state = preprocess_frame(state)
        for _ in range(frame_stack_size):
            frame_stack.add(state)
        state = frame_stack.get_state()

        total_reward = 0
        done = False
        steps = 0
        episode_loss = 0
        q_values_list = []
        current_episode_path = []

        while not done:
            if np.random.rand() <= epsilon:
                action = np.random.choice(n_actions)
            else:
                q_values = q_model.predict(np.expand_dims(state, axis=0), verbose=0)
                action = np.argmax(q_values[0])

            next_state, reward, done, _, _ = env.step(action)
            next_frame = preprocess_frame(next_state)
            frame_stack.add(next_frame)
            next_state = frame_stack.get_state()

            # Store current raw frame and action (for visualization)
            current_episode_path.append((next_frame.copy(), action))

            replay_buffer.add((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            steps += 1
            total_frames += 1

            if replay_buffer.size() >= batch_size:
                minibatch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*minibatch)

                states = np.array(states)
                next_states = np.array(next_states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                dones = np.array(dones)

                target_q_values = target_model.predict(next_states, verbose=0)
                max_next_q_values = np.max(target_q_values, axis=1)
                target = rewards + (1 - dones) * gamma * max_next_q_values

                q_values = q_model.predict(states, verbose=0)
                for i in range(batch_size):
                    if actions[i] < n_actions:
                        q_values[i][actions[i]] = target[i]
                    else:
                        print(f"Warning: Action {actions[i]} is out of bounds for q_values shape {q_values.shape}")
                        q_values[i][0] = target[i]

                loss = q_model.train_on_batch(states, q_values)
                episode_loss += loss

            # Track Q-value stats
            q_values = q_model.predict(np.expand_dims(state, axis=0), verbose=0)
            avg_q = np.mean(q_values)
            q_values_list.append(avg_q)
            frame_q_values.append(avg_q)
            frame_numbers.append(total_frames)

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

        # Save best episode path
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode_path = current_episode_path.copy()

        # Logging
        episode_rewards.append(total_reward)
        avg_loss = episode_loss / steps if steps > 0 else 0  # Handle divide by zero
        losses.append(avg_loss)
        epsilon_values.append(epsilon)
        avg_q_episode = np.mean(q_values_list)
        avg_q_values_per_episode.append(avg_q_episode)

        # Save model periodically
        if total_frames % save_interval == 0:
            save_path = f"{log_dir}/{env.spec.id}_model_{total_frames}.h5"
            print(f"Saving model at frame {total_frames} to {save_path}")
            save_model(q_model, save_path, env.spec.id)

        # Save logs to CSV file
        save_logs(log_file, episode, total_reward, avg_loss, epsilon, avg_q_episode)

        # Print episode summary
        print(f"\nEpisode: {episode} | Game: {env.spec.id} | Total Reward: {total_reward} | "
              f"Loss: {avg_loss:.4f} | Epsilon: {epsilon:.4f}\n")

        # Update target model periodically
        if episode % target_update_freq == 0:
            target_model.set_weights(q_model.get_weights())

        # Evaluate agent periodically
        if eval_freq and total_frames % eval_freq == 0 and total_frames != 0:
            avg_reward = evaluate_agent(env, q_model)
            print(f"Evaluation - Frames: {total_frames}, Average Reward: {avg_reward}")

    rolling_avg_rewards = compute_rolling_average(episode_rewards)
    rolling_avg_q_values = compute_rolling_average(avg_q_values_per_episode)

    return (episode_rewards, losses, epsilon_values,
            rolling_avg_rewards, rolling_avg_q_values,
            frame_q_values, frame_numbers, best_episode_path)

def setup_environment_and_logging(env_name):
    """
    This function sets up the gym environment and creates a logging directory.

    Parameters:
    - env_name (str): The name of the environment to create.

    Returns:
    - env (gym.Env): The initialized environment.
    - log_dir (str): The path to the directory where logs will be saved.
    - n_actions (int): The number of actions available in the environment.
    """
    # Create the environment
    env = gym.make(env_name, render_mode='rgb_array')

    # Get the number of possible actions
    n_actions = env.action_space.n

    # Create a directory for logging results specific to the environment
    log_dir = os.path.join('./training_logs', env_name)
    print("log dir", log_dir)
    os.makedirs(log_dir, exist_ok=True)

    return env, log_dir, n_actions

def visualize_best_path(best_path):
    import matplotlib.pyplot as plt

    for i, (frame, action) in enumerate(best_path):
        plt.imshow(frame, cmap='gray')  # remove cmap if frame is RGB
        plt.title(f"Step {i+1}, Action: {action}")
        plt.axis('off')
        plt.pause(0.1)
        plt.clf()
    plt.close()

def plot_rewards(episode_rewards, log_dir, env_name, filename="reward_plot.png"):
    save_dir = os.path.join(log_dir, f"rewards")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    print(save_path, filename)

    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Reward plot saved at {save_path}")

def plot_losses(losses, log_dir, env_name, filename="loss_plot.png"):
    save_dir = os.path.join(log_dir, f"loss_plot")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Episode Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss per Episode')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Loss plot saved at {save_path}")

def plot_epsilon_decay(epsilon_values, log_dir, env_name, filename="epsilon_decay_plot.png"):
    save_dir = os.path.join(log_dir, f"epsilon_decay")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(10, 5))
    plt.plot(epsilon_values, label='Epsilon')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay per Episode')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Epsilon decay plot saved at {save_path}")

def plot_rolling_avg_rewards(rolling_avg_rewards, log_dir, env_name, filename="rolling_avg_reward_plot.png"):
    save_dir = os.path.join(log_dir, f"avg_rolling_reward")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(10, 5))
    plt.plot(rolling_avg_rewards, label='Rolling Avg Reward', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Rolling Average Reward')
    plt.title('Rolling Average Reward per Episode')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Rolling average reward plot saved at {save_path}")

def plot_rolling_avg_q_values(rolling_avg_q_values, log_dir, env_name, filename="rolling_avg_q_value_plot.png"):
    save_dir = os.path.join(log_dir, f"rolling_avg_q_value")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(10, 5))
    plt.plot(rolling_avg_q_values, label='Rolling Avg Q-value', color='purple')
    plt.xlabel('Episode')
    plt.ylabel('Rolling Average Q-value')
    plt.title('Rolling Average Q-value per Episode')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Rolling average Q-value plot saved at {save_path}")

def plot_frame_q_values(frame_numbers, frame_q_values, log_dir, env_name, filename="frame_q_values_plot.png"):
    save_dir = os.path.join(log_dir, f"frame_q_values")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(10, 5))
    plt.plot(frame_numbers, frame_q_values, label='Frame Q-values', color='teal')
    plt.xlabel('Frame Number')
    plt.ylabel('Q-value')
    plt.title('Q-values per Frame')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Frame Q-values plot saved at {save_path}")

def save_best_path_video_matplotlib(best_path, save_path, fps=10):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    filenames = []

    for i, (frame, action) in enumerate(best_path):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(frame, cmap='gray')
        ax.set_title(f"Step {i+1}, Action: {action}")
        ax.axis('off')

        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path)
        plt.close(fig)
        filenames.append(frame_path)

    clip = ImageSequenceClip(filenames, fps=fps)
    clip.write_videofile(save_path, fps=fps)

    # Cleanup temp
    for fname in filenames:
        os.remove(fname)
    os.rmdir(temp_dir)

    print(f"Saved video to {save_path}")

input_shape = (80, 80, frame_stack_size)
atari_envs_data = {}

env_names = ['ALE/Breakout-v5', 'ALE/Pong-v5', 'ALE/SpaceInvaders-v5', 'ALE/Seaquest-v5', 'ALE/BeamRider-v5']

for env_name in env_names:
    print(f"Running for env: {env_name}")

    env, log_dir, n_actions = setup_environment_and_logging(env_name)
    q_model = create_q_model(input_shape=(84, 84, frame_stack_size), n_actions=n_actions)
    target_model = create_q_model(input_shape=(84, 84, frame_stack_size), n_actions=n_actions)
    target_model.set_weights(q_model.get_weights())

    replay_buffer = ReplayBuffer(max_memory_size)
    frame_stack = FrameStack(frame_stack_size)

    start_time = time.time()
    print(f"Training for env {env_name} has started at {start_time}")
    episode_rewards, losses, epsilon_values, rolling_avg_rewards, rolling_avg_q_values, frame_q_values, frame_numbers, best_episode_path = train_agent(
        env=env,
        q_model=q_model,
        target_model=target_model,
        replay_buffer=replay_buffer,
        frame_stack=frame_stack,
        log_dir=log_dir,
        n_episodes=10,
        save_interval=10000
    )
    env.close()
    end_time = time.time()

    print(f"Total Training Time: {end_time - start_time} seconds")
    atari_envs_data[env_name] = {
        "episode_rewards": episode_rewards,
        "losses": losses,
        "epsilon_values": epsilon_values,
        "rolling_avg_rewards": rolling_avg_rewards,
        "rolling_avg_q_values": rolling_avg_q_values,
        "frame_q_values": frame_q_values,
        "frame_numbers": frame_numbers,
        "best_episode_path": best_episode_path
    }

    save_best_path_video_matplotlib(best_episode_path, save_path=f"{log_dir}/{env_name}/best_episode.mp4", fps=10)

    visualize_best_path(best_episode_path)

    plot_rewards(episode_rewards, log_dir, env_name, filename="reward_plot.png")
    plot_losses(losses, log_dir, env_name)
    plot_epsilon_decay(epsilon_values, log_dir, env_name)
    # plot_rolling_avg_rewards(rolling_avg_rewards, log_dir, env_name=env_name)
    # plot_rolling_avg_q_values(rolling_avg_q_values, log_dir, env_name)
    plot_frame_q_values(frame_numbers, frame_q_values, log_dir, env_name)