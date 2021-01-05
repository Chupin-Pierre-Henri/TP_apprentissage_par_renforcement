import gym
import vizdoomgym
import numpy as np
from skimage.viewer import ImageViewer
from VizdoomAgent import VizDoomAgent
from DQN import replay
from DQN import Transition
from affichage import Affichage

import gym
import argparse
from gym import wrappers, logger



if __name__ == "__main__":
    logger.set_level(logger.INFO)

    env = gym.make('VizdoomBasic-v0', depth=True, labels=True, position=True, health=True)
    outdir = 'log/viz_doom-agent-results'
    #env = wrappers.Monitor(env, directory=outdir, force=True)
    agent = VizDoomAgent(env, 0.01, 500, 0.005, True)
    episode_count = 500
    reward = 0
    done = False
    reword_recorder = Affichage()
    BATCH_SIZE = 128
    GAMMA = 0.9
    render = False
    buffer = replay(50000)
    for i in range(episode_count):
        ob = env.reset()
        reword_recorder.start_episode()
        step = 0
        while True:
            action = agent.act(agent.preprocess(ob), reward, done)
            transition = Transition(agent.preprocess(ob), action, None, None, None)
            if step % 100 == 0:
                print(step)
            ob, reward, done, _ = env.step(action)
            transition = Transition(transition.state, transition.action, agent.preprocess(ob), reward, done)            
            reword_recorder.add_value(reward)
            buffer.push(transition)

            if len(buffer) > BATCH_SIZE:
                sample = buffer.sample(BATCH_SIZE)
                agent.train(sample, GAMMA,True)
            if render:  
                env.render()
            step += 1
            if done:
                reword_recorder.recorde_episode()
                print('episode {} reward {} '.format(i, reword_recorder.recompense[-1]))
                break
    env.close()
    reword_recorder.show()
    agent.save_param()



