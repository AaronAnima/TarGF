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
    config.repeat_loss = 1
    config.lr = 2e-4
    config.ode_t0 = 1e-1
    config.beta1 = 0.9
    config.workers = 8
    config.hidden_dim_gf = 128
    config.embed_dim_gf = 64
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
    config.score_exp = 'Room_Score'
    # env
    config.horizon = 250
    config.max_vel = 4. 
    # train 
    config.max_timesteps = 1e6
    config.batch_size_rl = 256
    config.discount = 0.95
    config.tau = 0.005
    config.policy_freq = 1
    config.start_timesteps = 25e2
    config.residual_t0 = 0.01
    config.buffer_size = 1e6
    config.is_residual = True 
    config.hidden_dim_actor = 128
    config.embed_dim_actor = 64
    config.hidden_dim_critic = 128
    config.embed_dim_critic = 64
    # reward
    config.reward_mode = 'densityIncre'
    config.reward_freq = 1
    config.lambda_col = 1.0
    config.lambda_sim = 5.0
    config.reward_t0 = 0.01
    config.normalize_reward = True
    # eval
    config.eval_freq_rl = 100
    config.eval_num = 4

    ''' Eval Policy '''
    config.policy_type = 'targf_sac' # for room: ['targf_sac']
    config.recover = True
    config.test_num = 2 # test set contains 83 different room-conditions
    config.policy_exp = 'Room_SAC'
    config.calc_metrics = True
    config.save_videos = True
    config.test_seeds = [5, 10, 15, 20, 25]

    return config
