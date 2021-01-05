import gym
import argparse
from gym import wrappers, logger

from CartPoleAgent import CartPoleAgent
from DQN import replay
from DQN import Transition
from affichage import Affichage


if __name__ == "__main__":
    logger.set_level(logger.INFO)

    env = gym.make('CartPole-v1')
    outdir = 'log/random-agent-results'
    #env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = CartPoleAgent(env, 0.01, 500, 0.005, True)
    BUFFER_SIZE = 50000
    episode_count = 600
    reward = 0
    end = False
    reword_recorder = Affichage()
    BATCH_SIZE = 128 
    GAMMA = 0.9
    render = False

    buffer = replay(BUFFER_SIZE)
    for i in range(episode_count):
        ob = env.reset()
        reword_recorder.start_episode()
        step = 0
        while True:
            action = agent.act(ob, reward, end)
            transition = Transition(ob, action, None, None, None)
            ob, reward, end, _ = env.step(action)
            reword_recorder.add_value(reward)
            transition = Transition(transition.state, transition.action, ob, reward, end)
            buffer.push(transition)

            if len(buffer) >= BATCH_SIZE:
                sample = buffer.sample(BATCH_SIZE)
                agent.train(sample, GAMMA)

            if render:  
                env.render()
            step += 1
            if end:
                # if i % 100 == 0:
                print('episode {} done took {} steps '.format(i, step))
                reword_recorder.recorde_episode()
                break
    env.close()
    agent.save_param()
    reword_recorder.show()
