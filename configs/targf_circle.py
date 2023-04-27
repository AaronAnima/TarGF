from configs.targf_ball_base import get_default_config

def get_config():
    config = get_default_config()
    
    ''' Train GF '''
    # file
    config.data_name = 'Circle_Examples'
    # env
    config.pattern = 'Circle'

    ''' Train SAC '''
    # file 
    config.score_exp = 'Circle_Score_fixbug_1e4'
    # reward
    config.lambda_col = 3.0
    config.lambda_sim = 1.0

    ''' Eval Policy '''
    config.policy_type = 'targf_orca' # ['targf_orca', 'targf_sac']
    config.policy_exp = 'Circle_SAC'
    config.calc_metrics = True
    config.save_videos = True
    config.test_seeds = [5, 10]
    config.test_num = 5

    return config
