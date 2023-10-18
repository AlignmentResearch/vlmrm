from stable_baselines3.common.base_class import BaseAlgorithm

from vlmrm.contrib.sb3.clip_rewarded_dqn import CLIPRewardedDQN
from vlmrm.contrib.sb3.clip_rewarded_sac import CLIPRewardedSAC


def get_clip_rewarded_rl_algorithm_class(env_name: str) -> BaseAlgorithm:
    if env_name in ["Humanoid-v4", "MountainCarContinuous-v0"]:
        return CLIPRewardedSAC
    elif env_name in ["CartPole-v1"]:
        return CLIPRewardedDQN
    else:
        raise ValueError(f"Unknown environment: {env_name}")
