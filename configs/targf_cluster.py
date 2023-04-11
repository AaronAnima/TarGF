from configs.targf_ball_base import get_default_config

def get_config():
    config = get_default_config()
    
    ''' Train GF '''
    # file
    config.data_name = 'Cluster_Examples'
    # env
    config.pattern = 'Cluster'

    ''' Train SAC '''
    # file 
    config.score_exp = 'Cluster_Score'
    # reward
    config.lambda_col = 5.0
    config.lambda_sim = 1.0

    ''' Eval Policy '''
    config.policy_type = 'targf_sac' # ['targf_orca', 'targf_sac']
    config.policy_exp = 'Cluster_SAC'
    config.calc_metrics = True
    config.save_videos = True
    config.test_seeds = [5, 10, 15, 20, 25]
    config.test_num = 4

    return config
