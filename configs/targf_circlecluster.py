from configs.targf_ball_base import get_default_config

def get_config():
    config = get_default_config()
    
    ''' Train GF '''
    # file
    config.data_name = 'CircleCluster_Examples'
    # env
    config.pattern = 'CircleCluster'

    ''' Train SAC '''
    # file 
    config.score_exp = 'CircleCluster_Score_rebuild'
    # reward
    config.lambda_col = 5.0
    config.lambda_sim = 1.0

    ''' ORCA '''


    ''' Eval Policy '''

    return config
