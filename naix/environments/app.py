import gym
import random
import numbers
import logging
import redis
import shutil
import arrow
import time
import json
import cv2
import numpy as np
from skimage import io
from skimage import measure
from pathlib import Path
from itertools import product
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering
from collections import deque
from naix.settings import DIR_ASSETS
from naix.models.images import TemplateContainsModel
from naix.models.bugs import WhiteScreenDetector
from naix.actions.grids import minimal_actions as minimal_grid_actions


logger = logging.getLogger(__name__)


class AppTrainEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.action_set = minimal_grid_actions(self.seed()[0], (1080, 1920), (120, 80), 400)
        self.observation_space = spaces.Box(low=0.0, high=1.0, dtype=np.float64, shape=(100, 100))
        self.action_space = spaces.Discrete(len(self.action_set))
        self.reward_range = (-1.0, 1.0)

        self._package_name = None
        self._interface = None
        self._rewards = deque()
        self._latest_states = deque(maxlen=10)
        self._bug_detectors = [
            WhiteScreenDetector()
        ]

    def initialize(self, **kwargs):
        self._interface = kwargs.get('interface', self._interface)
        self._package_name = kwargs.get('package_name', self._package_name)
        return self

    def step(self, action):
        # 1. act and wait
        state_before_action = self._interface.screenshot()
        self._interface.execute(self.action_set[action])

        # 2. state and reward
        state_after_action = self._interface.screenshot()
        done = self._is_done(state_after_action)
        reward = self._calculate_reward(state_before_action, state_after_action, done)

        # 3. add records
        self._rewards.append(reward)
        self._latest_states.append(state_after_action)

        return state_after_action, reward, done, None

    def seed(self, seed=None):
        return [666, 0]

    def reset(self):
        self._latest_states.clear()
        self._rewards.clear()
        self._reset_app_ui()
        return self._interface.screenshot()

    def render(self, mode='human', close=False):
        return self._interface.screenshot()

    def reward_sum(self):
        return sum(self._rewards)

    def _reset_app_ui(self):
        self._interface.execute(action=('startapp', self._package_name))
        back_cnt = 0
        while self._interface.is_running_page(self._package_name):
            self._interface.execute(action=('back', ), interval_seconds=0.5)
            back_cnt += 1

        logger.info('takes %s backs to exit app', back_cnt)
        self._interface.execute(action=('startapp', self._package_name))
        time.sleep(10)

    def _calculate_reward(self, state_before_action, state_after_action, done_flag):
        """
        -1 for exiting, -0.75 for no change, -0.5 for no changes in latest page
        0.5 for partial changes, 0.75 for totally changes, 1 for totally changes and bug find
        """

        # 1. if state changes even no actions is done, return 0 as reward
        if measure.compare_mse(state_before_action, state_after_action) >= 50.0:
            return 0.0

        # 2. calculate rewards based on change ratio
        any_bug = np.any([detector.is_bug(state_after_action) for detector in self._bug_detectors])
        if done_flag and not any_bug:
            return -1.0

        for i, item in enumerate(reversed(self._latest_states)):
            mse = measure.compare_mse(state_after_action, item)
            if mse < 50.0:
                return -0.5 - 0.25 * bool(i == 0)

        return 0.75 + 0.25 * bool(any_bug)

    def _is_done(self, state):
        """Only when the app is running foreground and the total rewards are larger than -5.0, the app is not done"""
        if not self._interface.is_running_page(self._package_name):
            return True

        return self.reward_sum() <= -5.0
