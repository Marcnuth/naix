from naix.interfaces.android import AndroidAdbInterface
import gym
import logging
from skimage import transform, color
from naix.interfaces.android import AndroidAdbInterface
from naix.models.networks.dqn import DQN
from naix.environments.app import AppTrainEnv


def test_android():

    d = AndroidAdbInterface(device_id='78cb185f')
    d._execute_back()
    print(d._operations)


def test_reset():
    interface = AndroidAdbInterface(device_id='78cb185f')
    env = AppTrainEnv().initialize(
        interface=interface, package_name='com.tencent.mtt'
    )
    env.reset()