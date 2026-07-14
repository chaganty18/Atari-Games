import gymnasium as gym
import ale_py

# Check available Atari environments
atari_envs = [env for env in gym.envs.registry.keys() if "ALE/" in env]
print(f"Found {len(atari_envs)} Atari environments:")
for env in sorted(atari_envs):
    print(env)
