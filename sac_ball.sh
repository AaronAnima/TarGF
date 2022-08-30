#####################################################################################################################################################################
#####################################################--------Training-------#########################################################################################
#####################################################################################################################################################################

#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------- Sorting Paper -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

CUDA_VISIBLE_DEVICES=0 python Runners/BallSAC.py \
--exp_name Sorting_SAC_col5_sim1_001_01_gamma095_7balls_knn20_horizon100 \
--reward_mode densityIncre \
--horizon 100 \
--n_boxes 7 \
--knn_actor 20 \
--knn_critic 20 \
--normalize_reward True \
--env sorting \
--score_exp SDE_sorting_1e5_1e4epoches_7balls \
--reward_freq 1 \
--hidden_dim 128 \
--embed_dim 64 \
--reward_t0 0.01 \
--residual_t0 0.1 \
--lambda_col 5.0 \
--lambda_sim 1.0 \
--is_residual True \
--batch_size 256 \
--eval_num 25 \
--eval_col 5 \
--discount 0.95 \
--start_timesteps 2500 \
--eval_freq 100 \
--max_timesteps 500000 \
--seed 0 \


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------- Placing Paper -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


# CUDA_VISIBLE_DEVICES=0 python Runners/BallSAC.py \
# --exp_name Placing_SAC_col3_sim1_001_01_gamma095_7balls_knn20_horizon100 \
# --reward_mode densityIncre \
# --n_boxes 7 \
# --horizon 100 \
# --knn_actor 20 \
# --knn_critic 20 \
# --normalize_reward True \
# --env placing \
# --score_exp placing \
# --reward_freq 1 \
# --hidden_dim 128 \
# --embed_dim 64 \
# --reward_t0 0.01 \
# --residual_t0 0.1 \
# --lambda_col 3.0 \
# --lambda_sim 1.0 \
# --is_residual True \
# --batch_size 256 \
# --eval_num 25 \
# --eval_col 5 \
# --discount 0.95 \
# --start_timesteps 2500 \
# --eval_freq 100 \
# --max_timesteps 500000 \
# --seed 0 \


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------- Hybrid Paper --------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

# CUDA_VISIBLE_DEVICES=0 python Runners/BallSAC.py \
# --exp_name Hybrid_SAC_col5_sim1_001_01_gamma095_7balls_knn20_horizon100 \
# --n_boxes 7 \
# --horizon 100 \
# --knn_actor 20 \
# --knn_critic 20 \
# --reward_mode densityIncre \
# --normalize_reward True \
# --env hybrid \
# --score_exp hybrid \
# --reward_freq 1 \
# --hidden_dim 128 \
# --embed_dim 64 \
# --reward_t0 0.01 \
# --residual_t0 0.1 \
# --lambda_col 5.0 \
# --lambda_sim 1.0 \
# --is_residual True \
# --batch_size 256 \
# --eval_num 25 \
# --eval_col 5 \
# --discount 0.95 \
# --start_timesteps 2500 \
# --eval_freq 100 \
# --max_timesteps 500000 \
# --seed 0 \


#####################################################################################################################################################################
#####################################################--------Evaluation-------#######################################################################################
#####################################################################################################################################################################


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------- Sorting Paper -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

# CUDA_VISIBLE_DEVICES=0 python Runners/EvalMASAC.py \
# --exp_name Sorting_SAC_col5_sim1_001_01_gamma095_7balls_knn20_horizon100 \
# --action_type vel \
# --inp_mode state \
# --recover False \
# --is_best False \
# --eval_mode analysis \
# --eval_num 100 \
# --horizon 100 \
# --n_boxes 7 \
# --env sorting \
# --score_exp sorting \
# --is_residual True \
# --seed 0 \
# --hidden_dim 128 \
# --embed_dim 64 \
# --reward_t0 0.01 \
# --residual_t0 0.1 \
# --knn_actor 20 \
# --knn_critic 20 \


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------- Placing Paper -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

# CUDA_VISIBLE_DEVICES=0 python Runners/EvalMASAC.py \
# --exp_name Placing_SAC_col3_sim1_001_01_gamma095_7balls_knn20_horizon100 \
# --eval_num 100 \
# --recover False \
# --is_best False \
# --inp_mode state \
# --action_type vel \
# --horizon 100 \
# --n_boxes 7 \
# --env placing \
# --score_exp placing \
# --is_residual True \
# --eval_mode fullmetric \
# --seed 0 \
# --hidden_dim 128 \
# --embed_dim 64 \
# --reward_t0 0.01 \
# --residual_t0 0.1 \
# --knn_actor 20 \
# --knn_critic 20 \

#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------- Hybrid Paper --------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

# CUDA_VISIBLE_DEVICES=0 python Runners/EvalMASAC.py \
# --exp_name Hybrid_SAC_col5_sim1_001_01_gamma095_7balls_knn20_horizon100 \
# --action_type vel \
# --n_boxes 7 \
# --eval_num 100 \
# --is_best False \
# --horizon 100 \
# --recover False \
# --env hybrid \
# --score_exp hybrid \
# --is_residual True \
# --eval_mode fullmetric \
# --seed 0 \
# --hidden_dim 128 \
# --embed_dim 64 \
# --reward_t0 0.01 \
# --residual_t0 0.1 \
# --knn_actor 20 \
# --knn_critic 20 \




