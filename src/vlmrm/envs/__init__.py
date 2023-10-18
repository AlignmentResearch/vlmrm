import gymnasium

from vlmrm.envs.base import *  # noqa F401
from vlmrm.envs.base import get_clip_rewarded_env_name

gymnasium.register(
    get_clip_rewarded_env_name("MountainCarContinuous-v0"),
    "vlmrm.envs.classic_control.clip_rewarded_mountain_car_continuous:CLIPRewardedContinuousMountainCarEnv",  # noqa: E501
)

gymnasium.register(
    get_clip_rewarded_env_name("CartPole-v1"),
    "vlmrm.envs.classic_control.clip_rewarded_cart_pole:CLIPRewardedCartPoleEnv",
)


gymnasium.register(
    get_clip_rewarded_env_name("Humanoid-v4"),
    "vlmrm.envs.mujoco.clip_rewarded_humanoid:CLIPRewardedHumanoidEnv",
)
