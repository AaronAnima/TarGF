# Modified from Yang Song's repo: https://github.com/yang-song/score_sde_pytorch
import os
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
from torch.utils.tensorboard import SummaryWriter
from ipdb import set_trace

from runners.train_gf import gf_trainer
from runners.train_rl import rl_trainer
# from runners.eval_policy import evaluate
from utils.misc import exists_or_mkdir

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train_gf", "train_rl", "eval_targf_sac", "eval_targf_orca"], "Running mode: train modules or eval policies")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
    # create log dirs
    exists_or_mkdir('./logs')
    exists_or_mkdir(os.path.join('./logs', FLAGS.workdir))
    if FLAGS.mode == "train_gf":
        exists_or_mkdir(os.path.join('./logs', FLAGS.workdir, 'test_videos'))
        tb_path = os.path.join('./logs', FLAGS.workdir, 'tb')
        exists_or_mkdir(tb_path)
        writer = SummaryWriter(tb_path)
        # Run the training pipeline
        gf_trainer(FLAGS.config, FLAGS.workdir, writer)
    elif FLAGS.mode == "train_rl":
        tb_path = os.path.join('./logs', FLAGS.workdir, 'tb')
        exists_or_mkdir(tb_path)
        writer = SummaryWriter(tb_path)
        # Run the training pipeline
        rl_trainer(FLAGS.config, FLAGS.workdir, writer)
    elif 'eval' in FLAGS.mode: # FLAGS.mode in ['eval_targf_sac', 'eval_targf_orca']
        policy_type = (FLAGS.mode).split('_')[-1]
        # Run the evaluation pipeline
        evaluate(FLAGS.config, FLAGS.workdir, policy_type=policy_type)
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
