import gym
import ebor

from envs.Room.RoomArrangement import RLEnvDynamic

def get_env(configs):
    if configs.env_type == 'Room': 
        tar_data = 'UnshuffledRoomsMeta'
        exp_kwconfigs = {
            'max_vel': configs.max_vel,
            'pos_rate': 1,
            'ori_rate': 1,
            'max_episode_len': configs.horizon,
        }
        env = RLEnvDynamic(
            tar_data,
            exp_kwconfigs,
            meta_name='ShuffledRoomsMeta',
            is_gui=False,
            fix_num=None, 
            split='train', # determine initialising on which split of rooms
        )
        max_action = configs.max_vel
    elif configs.env_type == 'Ball':
        env_name = '{}-{}Ball{}Class-v0'.format(configs.pattern, configs.num_objs, configs.num_classes)
        env = gym.make(env_name)
        env.seed(configs.seed)
        env.reset()
        max_action = env.action_space['obj1']['linear_vel'].high[0]
    else:
        raise ValueError(f"Mode {configs.env_type} not recognized.")
    return env, max_action