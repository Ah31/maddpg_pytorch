import torch
import torch.nn
from  maddpg_agent_changed import Agent
import numpy as np
import make_env as gym_env
from utils import visdom_line_plotter
import time
import os
import pickle


def create_agents(env,obs_shape_n):
    agents = []
    for i in range(0,env.n):
        agents.append(Agent("agent_%d"% i,state_size=obs_shape_n,action_size=5,agent_index=i,random_seed=0)) #change after starts working
    return agents

def train(display):
    env = gym_env.make_env(scenario_name="simple_spread", benchmark=True)
    obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
    agents = create_agents(env,obs_shape_n)
    plotter = visdom_line_plotter.VisdomLinePlotter()

    plots_dir = './learning_curves/'
    benchmark_iters = 100000
    benchmark_dir = './benchmark_files_'
    exp_name= 'maddpg_pytorch'

    benchmark = True
    TAU = 1.0 - 1e-2
    max_episode_len = 25

    if display:
        for i in range(len(agents)):
            actor_local = agents[i].actor_local
            actor_local_ckpt = torch.load('./checkpoints/checkpoint_actor_{}.pth'.format(i), map_location='cpu')
            actor_local.load_state_dict(actor_local_ckpt)
            actor_target = agents[i].actor_target
            actor_target_ckpt = torch.load('./checkpoints/checkpoint_actor_target_{}.pth'.format(i), map_location ='cpu')
            actor_target.load_state_dict(actor_target_ckpt)
            critic_local = agents[i].critic_local
            critic_local_ckpt = torch.load('./checkpoints/checkpoint_critic_{}.pth'.format(i), map_location='cpu')
            critic_local.load_state_dict(critic_local_ckpt)
            critic_target = agents[i].critic_target
            critic_target_ckpt = torch.load('./checkpoints/checkpoint_critic_target_{}.pth'.format(i), map_location='cpu')
            critic_target.load_state_dict(critic_target_ckpt)


    final_ep_rewards = []
    final_ep_ag_rewards= []
    episode_rewards = [0.0]
    agent_rewards = [[0.0] for _ in range(env.n)]
    agent_info = [[[]]]
    obs_n = env.reset()
    episode_step = 0
    train_step = 0
    t_start = time.time()

    print('Starting iterations...')
    while True:
        # get action
        for agent in agents:
            agent.reset()
        action_n = [agent.act(obs,add_noise=True) for agent, obs in zip(agents, obs_n)]

        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        episode_step += 1
        done = all(done_n)
        terminal = (episode_step >= max_episode_len)
        # collect experience
        for i, agent in enumerate(agents):
            agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
        obs_n = new_obs_n
        # print(rew_n)
        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        if done or terminal:
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)
            # print(episode_rewards)
            for a in agent_rewards:
                a.append(0)
            agent_info.append([[]])



        # increment global step counter
        train_step += 1
        # for benchmarking learned policies
        if benchmark:
            for i, info in enumerate(info_n):
                agent_info[-1][i].append(info_n['n'])
            if train_step > benchmark_iters and (done or terminal) and (len(episode_rewards) % 1000 == 0):

                file_name = benchmark_dir + exp_name + '.pkl'
                print('Finished benchmarking, now saving...')
                with open(file_name, 'wb') as fp:
                    pickle.dump(agent_info[:-1], fp)
                # break
            # continue

        # for displaying learned policies
        if display:
            time.sleep(0.1)
            env.render()
            continue

        # update all trainers, if not in display or benchmark mode
        loss = None
        for agent in agents:
            agent.preupdate()
        for agent in agents:
            loss = agent.step(agents, train_step,terminal)


        # save model, display training output
        if terminal and (len(episode_rewards) % 1000 == 0):  # 25 and 1000

            print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-1000:]), round(time.time() - t_start, 3)))
            plotter.plot('Episode Rewards', 'Rewards', 'Training', len(episode_rewards), np.mean(episode_rewards[-1000:]))
            i=0
            for agt in agents:
                torch.save(agt.actor_local.state_dict(), './checkpoints/checkpoint_actor_{}.pth'.format(i))
                torch.save(agt.actor_target.state_dict(), './checkpoints/checkpoint_actor_target_{}.pth'.format(i))
                torch.save(agt.critic_local.state_dict(), './checkpoints/checkpoint_critic_{}.pth'.format(i))
                torch.save(agt.critic_target.state_dict(), './checkpoints/checkpoint_critic_target_{}.pth'.format(i))

                i+=1

            t_start = time.time()
            # Keep track of final episode reward
            final_ep_rewards.append(np.mean(episode_rewards[-1000:]))
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-1000:]))

        # if len(episode_rewards) > 60000:
        #     break

        # saves final episode reward for plotting training curve later
        if len(episode_rewards) > 10000:
            rew_file_name = plots_dir + exp_name + '_rewards.pkl'
            os.makedirs(os.path.dirname(rew_file_name), exist_ok=True)
            with open(rew_file_name, 'wb') as fp:
                pickle.dump(final_ep_rewards, fp)
            agrew_file_name = plots_dir + exp_name + '_agrewards.pkl'
            os.makedirs(os.path.dirname(agrew_file_name), exist_ok=True)
            with open(agrew_file_name, 'wb') as fp:
                pickle.dump(final_ep_ag_rewards, fp)
            print('...Finished total of {} episodes.'.format(len(episode_rewards)))
            break

if __name__ == '__main__':
    train(display=False)