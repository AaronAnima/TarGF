#------------------------------- 3090-8 -------------------------------

tensorboard --logdir ../logs/SDE_circlerect_1e5_1e4epoches_8balls/tb --port 10020
tensorboard --logdir ../logs/SDE_sorting6_1e5_1e4epoches_7balls/tb --port 10021 
tensorboard --logdir ../logs/test_1sample_sigma25_gnn_bs10000_pc/tb --port 10022
tensorboard --logdir ../logs/Sorting_SAC_col3_sim1_001_001_gamma095_7balls_knn20_horizon100_Image641e5_residual/tb --port 10023 
tensorboard --logdir ../logs/M4D26_bedroom_target_128_64_dataM12D25/tb --port 10024
tensorboard --logdir ../logs/Sorting_SAC_col0_sim1_0001_01_gamma095_7balls_knn20_horizon100_Image641e5/tb --port 10025 
tensorboard --logdir ../logs/Hybrid_GoalSAC_L1Incre_tar1_col3_VAEGoal_ball7/tb --port 10026 
tensorboard --logdir ../logs/Hybrid_SAC_col5_sim1_001_01_gamma095_7balls_knn20_horizon100/tb --port 10027
tensorboard --logdir ../logs/Placing_GoalSAC_L1Incre_tar1_col5_VAEGoal_ball7/tb --port 10028
tensorboard --logdir ../logs/Hybrid_GoalSAC_L1Incre_tar1_col5_VAEGoal_ball7/tb --port 10029
tensorboard --logdir ../logs/Placing_SAC_col2_sim1_001_01_gamma095_7balls_knn20_horizon100/tb --port 10020

#------------------------------- 3090-8-2 -------------------------------

# tensorboard --logdir ../logs/Sorting_SAC_col1_sim1_001_01_gamma095_7balls_knn20_horizon100/tb --port 10030 
# tensorboard --logdir ../logs/Sorting_SAC_col3_sim1_001_01_gamma095_7balls_knn20_horizon100/tb --port 10031 
# tensorboard --logdir ../logs/Sorting_SAC_col5_sim1_001_01_gamma095_7balls_knn20_horizon100/tb --port 10032 
# tensorboard --logdir ../logs/Placing_SAC_col1_sim1_001_01_gamma095_7balls_knn20_horizon100/tb --port 10033 
# tensorboard --logdir ../logs/Placing_SAC_col3_sim1_001_01_gamma095_7balls_knn20_horizon100/tb --port 10034
# tensorboard --logdir ../logs/Placing_SAC_col5_sim1_001_01_gamma095_7balls_knn20_horizon100/tb --port 10035 
# tensorboard --logdir ../logs/Hybrid_SAC_col1_sim1_001_01_gamma095_7balls_knn20_horizon100/tb --port 10036
# tensorboard --logdir ../logs/Hybrid_SAC_col3_sim1_001_01_gamma095_7balls_knn20_horizon100/tb --port 10037