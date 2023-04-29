import ml_collections

def get_default_config():
    config = ml_collections.ConfigDict()
    config.env_type = 'Ball'
    
    ''' Train GF '''
    # file
    config.data_name = 'CircleCluster_Examples'
    # env
    config.seed = 0
    config.pattern = 'CircleCluster'
    config.num_per_class = 7
    config.num_classes = 3
    config.num_objs = config.num_per_class * config.num_classes
    # train
    config.n_epoches = 10000
    config.n_samples = 10000
    config.batch_size_gf = 2048
    config.repeat_loss = 1
    config.lr = 2e-4
    config.ode_t0 = 1.0
    config.beta1 = 0.5
    config.workers = 8
    config.hidden_dim_gf = 64
    config.embed_dim_gf = 32
    config.sigma = 25.
    config.full_train = False
    # eval
    config.vis_freq_gf = 100
    config.video_freq_gf = 1000
    config.test_ratio = 0.1 # for splitting the training set
    config.vis_col = 4
    config.sampling_steps = 2000

    ''' Train SAC '''
    # file 
    config.score_exp = 'None'
    # env
    config.action_type = 'vel'
    # model
    config.knn_actor = 20
    config.knn_critic = 20
    config.hidden_dim_actor = 128
    config.embed_dim_actor = 64
    config.hidden_dim_critic = 128
    config.embed_dim_critic = 64
    config.is_residual = True
    config.residual_t0 = 0.01
    # train 
    config.discount = 0.95
    config.start_timesteps = 2500
    config.max_timesteps = 5e5
    config.batch_size_rl = 256
    config.tau = 0.005
    config.policy_freq = 1
    config.buffer_size = 1e6
    # reward
    config.reward_t0 = 0.01
    config.reward_mode = 'densityIncre'
    config.lambda_col = 5.0
    config.lambda_sim = 1.0
    config.reward_freq = 1
    config.normalize_reward = True
    # eval
    config.eval_freq_rl = 100
    config.eval_col = 5
    config.eval_num = config.eval_col**2

    ''' ORCA '''
    config.is_decay_t0_orca = False
    config.orca_t0 = 0.01
    config.knn_orca = 2

    ''' Eval Policy '''
    config.policy_type = 'targf_sac' # ['targf_orca', 'targf_sac']
    config.policy_exp = 'CircleCluster_SAC_rebuild'
    config.calc_metrics = True
    config.save_videos = True
    config.test_seeds = [5, 10, 15, 20, 25]
    config.test_num = 4
    config.recover = False 


    return config
