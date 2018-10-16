import gym
import logging
from skimage import transform, color
from naix.interfaces.android import AndroidAdbInterface
from naix.models.networks.dqn import DQN


logger = logging.getLogger(__name__)


def run():

    interface = AndroidAdbInterface(device_id='78cb185f')

    env = gym.make('app-train-v0').initialize(
        interface=interface, package_name='com.tencent.mtt'
    )
    agent = DQN(env=env, input_shape=(100, 100, 1))
    agent.save('e:/tmp/dqn_app_agent/')


    for e in range(5000):
        reward_sum = 0.0
        state = adapt_state(env.reset())

        for t in range(500):

            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = adapt_state(next_state)
            logger.info('action: %s, reward: %s, done:%s, epsilon:%s', env.action_set[action], reward, done, agent.epsilon)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            reward_sum += reward

            if done:
                print('done when eposide: %s/5000, steps:%s, reward sum:%s' % (e, t, reward_sum))
                break


        agent.replay(32)

    agent.save('e:/tmp/dqn_app_agent/')


def adapt_state(state):
    return transform.resize(color.rgb2gray(state), (1, 100, 100, 1))