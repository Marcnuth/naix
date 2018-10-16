import logging
import random
import time
import json
import numpy as np
from pathlib import Path
from collections import deque
from naix.models.networks import cnn


logger = logging.getLogger(__name__)


class DQN:
    def __init__(self, env, input_shape):
        self.env = env
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = cnn.build(input_shape, self.env.action_space.n)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(batch_size, len(self.memory)))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, outpath):
        outpath = Path(outpath)
        outpath.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        self.model.save((outpath / 'model_{}.h5'.format(timestamp)).absolute().as_posix())

        with open((outpath / 'action_set.json').absolute().as_posix(), 'w+') as f:
            f.write(json.dumps(self.env.action_set))
