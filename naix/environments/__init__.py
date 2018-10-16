from gym.envs.registration import register
from naix.environments.app import AppTrainEnv


register(
    id='app-train-v0',
    entry_point='naix.environments.app:AppTrainEnv',
)
