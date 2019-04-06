
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.optim


## Actor Critic architecture

class Actor(nn.Module):
    def __init__(self, n_actions, space_dims, hidden_dims):
        super(Actor, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(space_dims, hidden_dims),
            nn.ReLU(True),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dims, n_actions),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        policy = self.actor(features)
        return policy
    
class Critic(nn.Module):
    def __init__(self, space_dims, hidden_dims):
        super(Critic, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(space_dims, hidden_dims),
            nn.ReLU(True),
        )
        self.critic = nn.Linear(hidden_dims, 1)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        est_reward = self.critic(features)
        return est_reward

## ICM Architecture

class InverseModel(nn.Module):
    def __init__(self, n_actions, hidden_dims):
        super(InverseModel, self).__init__()
        self.fc = nn.Linear(hidden_dims*2, n_actions)
        
    def forward(self, features):
        features = features.view(1, -1) # (1, hidden_dims)
        action = self.fc(features) # (1, n_actions)
        return action

class ForwardModel(nn.Module):
    def __init__(self, n_actions, hidden_dims):
        super(ForwardModel, self).__init__()
        self.fc = nn.Linear(hidden_dims+n_actions, hidden_dims)
        self.eye = torch.eye(n_actions)
        
    def forward(self, action, features):
        x = torch.cat([self.eye[action], features], dim=-1) # (1, n_actions+hidden_dims)
        features = self.fc(x) # (1, hidden_dims)
        return features

class FeatureExtractor(nn.Module):
    def __init__(self, space_dims, hidden_dims):
        super(FeatureExtractor, self).__init__()
        self.fc = nn.Linear(space_dims, hidden_dims)
        
    def forward(self, x):
        y = torch.tanh(self.fc(x))
        return y
