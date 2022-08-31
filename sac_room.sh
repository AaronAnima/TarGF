#####################################################################################################################################################################
#####################################################--------Training-------#########################################################################################
#####################################################################################################################################################################

# CUDA_VISIBLE_DEVICES=0 python Runners/RoomSAC.py \
# --exp_name Arrangement_SAC_multi_normr_col1_sim5_vel10_brown_10_100_flip_rotate_756rooms \
# --is_single_room False \
# --max_vel 10.0 \
# --reward_mode densityIncre \
# --normalize_reward True \
# --score_exp M4D26_bedroom_target_128_64_dataM12D25 \
# --reward_freq 1 \
# --hidden_dim 128 \
# --embed_dim 64 \
# --reward_t0 0.01 \
# --residual_t0 0.01 \
# --lambda_col 1.0 \
# --lambda_sim 5.0 \
# --is_residual True \
# --batch_size 256 \
# --eval_num 4 \
# --horizon 250 \
# --discount 0.95 \
# --start_timesteps 2500 \
# --eval_freq 1 \
# --max_timesteps 1000000 \
# --seed 0 \
# --sigma 25.0 \


# CUDA_VISIBLE_DEVICES=1 python Runners/RoomSAC.py \
# --exp_name Arrangement_SAC_multi_normr_col1_sim5_vel10_brown_10_100_flip_rotate \
# --is_single_room False \
# --max_vel 10.0 \
# --reward_mode densityIncre \
# --normalize_reward True \
# --score_mode room \
# --reward_freq 1 \
# --hidden_dim 128 \
# --embed_dim 64 \
# --reward_t0 0.01 \
# --residual_t0 0.01 \
# --lambda_col 1.0 \
# --lambda_sim 5.0 \
# --is_residual True \
# --batch_size 256 \
# --eval_num 4 \
# --horizon 250 \
# --discount 0.95 \
# --start_timesteps 2500 \
# --eval_freq 50 \
# --max_timesteps 1000000 \
# --seed 0 \
# --sigma 25.0 \



#####################################################################################################################################################################
#####################################################--------Evaluation-------#######################################################################################
#####################################################################################################################################################################


# CUDA_VISIBLE_DEVICES=1 python Runners/RoomEvalSAC.py \
# --exp_name Arrangement_SAC_multi_normr_col1_sim5_vel10_brown_10_100_flip_rotate_756rooms \
# --eval_num 83 \
# --save_video False \
# --is_single_room False \
# --score_exp M4D26_bedroom_target_128_64_dataM12D25 \
# --hidden_dim 128 \
# --embed_dim 64 \
# --residual_t0 0.01 \
# --horizon 250 \
# --sigma 25.0 \


# CUDA_VISIBLE_DEVICES=1 python Runners/RoomEvalSAC.py \
# --exp_name Arrangement_SAC_multi_normr_col1_sim5_vel10_brown_10_100_flip_rotate \
# --eval_num 83 \
# --save_video False \
# --is_single_room False \
# --score_exp M4D26_bedroom_target_128_64_dataM12D25 \
# --hidden_dim 128 \
# --embed_dim 64 \
# --residual_t0 0.01 \
# --horizon 250 \
# --sigma 25.0 \



