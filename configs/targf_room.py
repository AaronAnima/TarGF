import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.env_type = 'Room'
    
    ''' Train GF '''
    # file
    config.data_name = 'UnshuffledRoomsMeta'
    # env
    config.seed = 0
    # train
    config.n_epoches = 10000
    config.batch_size_gf = 64
    config.lr = 2e-4
    config.t0 = 1e-1
    config.beta1 = 0.9
    config.workers = 8
    config.hidden_dim = 128
    config.embed_dim = 64
    config.sigma = 25.
    config.full_train = False
    config.base_noise_scale = 0.01
    # eval
    config.vis_freq_gf = 20
    config.test_ratio = 0.1 # for splitting the training set
    config.vis_col = 2
    config.sampling_steps = 500

    ''' Train RL '''
    # file 
    config.score_exp = 'Room_Score_rebuild'
    # env
    config.horizon = 250
    config.max_vel = 4. 
    # train 
    config.max_timesteps = 1e6
    config.batch_size_rl = 256
    config.discount = 0.95
    config.tau = 0.005
    config.policy_freq = 1
    config.start_timesteps = 1e2
    config.residual_t0 = 0.01
    config.buffer_size = 1e6
    config.is_residual = True 
    # reward
    config.reward_mode = 'densityIncre'
    config.reward_freq = 1
    config.lambda_col = 1.0
    config.lambda_sim = 5.0
    config.reward_t0 = 0.01
    config.normalize_reward = True
    # eval
    config.eval_freq_rl = 1
    config.eval_num = 4

    return config
