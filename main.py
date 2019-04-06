
# coding: utf-8

# # Curiosity-Driven Exploration
import gym

import numpy as np
import torch
import torch.nn as nn
import torch.optim

import collections

from config import ConfigArgs as args
from models import Actor, Critic, InverseModel, ForwardModel, FeatureExtractor
from losses import PGLoss

def select_action(policy):
    return np.random.choice(len(policy), 1, p=policy)[0]

def to_tensor(x, dtype=None):
    return torch.tensor(x, dtype=dtype).unsqueeze(0)

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    # Actor Critic
    actor = Actor(n_actions=env.action_space.n, space_dims=4, hidden_dims=32)
    critic = Critic(space_dims=4, hidden_dims=32)

    # ICM
    feature_extractor = FeatureExtractor(env.observation_space.shape[0], 32)
    forward_model = ForwardModel(env.action_space.n, 32)
    inverse_model = InverseModel(env.action_space.n, 32)

    # Actor Critic
    a_optim = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    c_optim = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    # ICM
    icm_params = list(feature_extractor.parameters()) + list(forward_model.parameters()) + list(inverse_model.parameters())
    icm_optim = torch.optim.Adam(icm_params, lr=args.lr_icm)

    pg_loss = PGLoss()
    mse_loss = nn.MSELoss()
    xe_loss = nn.CrossEntropyLoss()

    global_step = 0
    n_eps = 0
    reward_lst = []
    mva_lst = []
    mva = 0.
    avg_ireward_lst = []

    while n_eps < args.max_eps:
        n_eps += 1
        next_obs = to_tensor(env.reset(), dtype=torch.float)
        done = False
        score = 0
        ireward_lst = []
        
        while not done:
            obs = next_obs
            a_optim.zero_grad()
            c_optim.zero_grad()
            icm_optim.zero_grad()
            
            # estimate action with policy network
            policy = actor(obs)
            action = select_action(policy.detach().numpy()[0])
            
            # interaction with environment
            next_obs, reward, done, info = env.step(action)
            next_obs = to_tensor(next_obs, dtype=torch.float)
            advantages = torch.zeros_like(policy)
            extrinsic_reward = to_tensor([0.], dtype=torch.float) if args.sparse_mode else to_tensor([reward], dtype=torch.float)
            t_action = to_tensor(action)
            
            v = critic(obs)[0]
            next_v = critic(next_obs)[0]
            
            # ICM
            obs_cat = torch.cat([obs, next_obs], dim=0)
            features = feature_extractor(obs_cat) # (2, hidden_dims)
            inverse_action_prob = inverse_model(features) # (n_actions)
            est_next_features = forward_model(t_action, features[0:1])

            # Loss - ICM
            forward_loss = mse_loss(est_next_features, features[1])
            inverse_loss = xe_loss(inverse_action_prob, t_action.view(-1))
            icm_loss = (1-args.beta)*inverse_loss + args.beta*forward_loss
            
            # Reward
            intrinsic_reward = args.eta*forward_loss.detach()
            if done:
                total_reward = -100 + intrinsic_reward if score < 499 else intrinsic_reward
                advantages[0, action] = total_reward - v
                c_target = total_reward
            else:
                total_reward = extrinsic_reward + intrinsic_reward
                advantages[0, action] = total_reward + args.discounted_factor*next_v - v
                c_target = total_reward + args.discounted_factor*next_v
            
            # Loss - Actor Critic
            actor_loss = pg_loss(policy, advantages.detach())
            critic_loss = mse_loss(v, c_target.detach())
            ac_loss = actor_loss + critic_loss
            
            # Update
            loss = args.lamda*ac_loss + icm_loss
            loss.backward()
            icm_optim.step()
            a_optim.step()
            c_optim.step()
            
            if not done:
                score += reward
            
            ireward_lst.append(intrinsic_reward.item())
            
            global_step += 1
        avg_intrinsic_reward = sum(ireward_lst) / len(ireward_lst)
        mva = 0.95*mva + 0.05*score
        reward_lst.append(score)
        avg_ireward_lst.append(avg_intrinsic_reward)
        mva_lst.append(mva)
        print('Episodes: {}, AVG Score: {:.3f}, Score: {}, AVG reward i: {:.6f}'.format(n_eps, mva, score, avg_intrinsic_reward))

    np.save('curiosity-mva.npy', np.array(mva_lst))
    print('Complete')

