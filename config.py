
# coding: utf-8

class ConfigArgs:
    beta = 0.2
    lamda = 0.1
    eta = 100.0 # scale factor for intrinsic reward
    discounted_factor = 0.99
    lr_critic = 0.005
    lr_actor = 0.001
    lr_icm = 0.001
    max_eps = 1000
    sparse_mode = True
