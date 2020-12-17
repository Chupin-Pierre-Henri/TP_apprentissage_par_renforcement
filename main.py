import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import wrappers, logger


def afficher_resultat(reward, episode):
    plt.plot(episode, reward)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor filesstate
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False
    somme_reward = 0
    y = np.array([])
    x = np.array([])
    for i in range(episode_count):
        ob = env.reset()
        somme_reward = 0
        nb_interaction = 0
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            somme_reward = somme_reward + reward
            nb_interaction += 1
            env.render()
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        y = np.append(y,(somme_reward/nb_interaction))
        x = np.append(x, i)
    afficher_resultat(y,x)    

    # Close the env and write monitor result info to disk
    env.close()