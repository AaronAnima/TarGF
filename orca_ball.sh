#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------- CircleRect Paper -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------



# CUDA_VISIBLE_DEVICES=0 python Runners/Planning/ORCA.py \
# --exp_name CircleRect_ORCA_score_knn2_001_8balls_vel \
# --is_onebyone False \
# --is_pid False \
# --is_decay False \
# --recover False \
# --action_type vel \
# --num_objs 8 \
# --knn_orca 2 \
# --horizon 100 \
# --mode debug \
# --eval_num 100 \
# --env circlerect \
# --score_mode circlerect \
# --orca_mode score \
# --target_t0 0.01 \
# --seed 0 \

# CUDA_VISIBLE_DEVICES=3 python Runners/Planning/ORCA.py \
# --exp_name CircleRect_ORCA_score_knn2_001_6balls_vel \
# --is_onebyone False \
# --is_pid False \
# --is_decay False \
# --recover False \
# --action_type vel \
# --num_objs 6 \
# --knn_orca 2 \
# --horizon 100 \
# --mode debug \
# --eval_num 100 \
# --env circlerect \
# --score_mode circlerect \
# --orca_mode score \
# --target_t0 0.01 \
# --seed 0 \

#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------- Sorting Paper -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

################################
#---------- ORCA-Score ---------
################################

# CUDA_VISIBLE_DEVICES=1 python Runners/Planning/ORCA.py \
# --exp_name Sorting_ORCA_score_knn2_01decay_7balls \
# --is_decay True \
# --recover False \
# --num_objs 7 \
# --knn_orca 2 \
# --horizon 100 \
# --mode eval \
# --eval_num 100 \
# --env sorting \
# --score_mode sorting \
# --orca_mode score \
# --target_t0 0.1 \
# --seed 0 \

# CUDA_VISIBLE_DEVICES=0 python Runners/Planning/ORCA.py \
# --exp_name Sorting6_ORCA_score_knn2_001_7balls \
# --is_onebyone False \
# --is_pid False \
# --is_decay False \
# --recover False \
# --action_type vel \
# --num_objs 7 \
# --knn_orca 2 \
# --horizon 200 \
# --mode eval \
# --eval_num 100 \
# --env sorting6 \
# --score_mode sorting6 \
# --orca_mode score \
# --target_t0 0.01 \
# --seed 0 \

# CUDA_VISIBLE_DEVICES=2 python Runners/Planning/ORCA.py \
# --exp_name Sorting_ORCA_score_knn2_001_5balls_onebyone \
# --is_onebyone True \
# --is_pid False \
# --is_decay False \
# --recover True \
# --action_type vel \
# --num_objs 5 \
# --knn_orca 2 \
# --horizon 300 \
# --mode debug \
# --eval_num 100 \
# --env sorting \
# --score_mode sorting \
# --orca_mode score \
# --target_t0 0.01 \
# --seed 0 \


################################
#---------- ORCA-Goal ----------
################################

# CUDA_VISIBLE_DEVICES=2 python Runners/Planning/ORCA.py \
# --exp_name Sorting_ORCA_VAEgoal_knn2_7balls \
# --recover False \
# --num_objs 7 \
# --horizon 100 \
# --eval_num 100 \
# --score_mode sorting \
# --goal_mode Score \
# --mode eval \
# --env sorting \
# --orca_mode goal \
# --knn_orca 2 \
# --seed 0 \

# CUDA_VISIBLE_DEVICES=3 python Runners/Planning/ORCA.py \
# --exp_name Sorting_ORCA_VAEgoal_knn2_001_5balls_onebyone \
# --is_onebyone True \
# --is_pid False \
# --is_decay False \
# --recover False \
# --action_type vel \
# --num_objs 5 \
# --knn_orca 2 \
# --horizon 300 \
# --mode debug \
# --eval_num 100 \
# --env sorting \
# --score_mode sorting \
# --orca_mode goal \
# --target_t0 0.01 \
# --seed 0 \

#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------- Placing Paper -------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

################################
#---------- ORCA-Score ---------
################################

# CUDA_VISIBLE_DEVICES=3 python Runners/Planning/ORCA.py \
# --exp_name Placing_ORCA_score_knn2_01decay_7balls \
# --is_decay True \
# --recover False \
# --eval_num 100 \
# --horizon 100 \
# --mode eval \
# --num_objs 7 \
# --env placing \
# --score_mode placing \
# --orca_mode score \
# --knn_orca 2 \
# --target_t0 0.1 \
# --seed 1 \

################################
#---------- ORCA-Goal ----------
################################

# CUDA_VISIBLE_DEVICES=0 python Runners/Planning/ORCA.py \
# --exp_name Placing_ORCA_VAEgoal_knn2_7balls \
# --eval_num 100 \
# --horizon 100 \
# --recover False \
# --score_mode placing \
# --num_objs 7 \
# --mode eval \
# --goal_mode Score \
# --env placing \
# --orca_mode goal \
# --knn_orca 2 \
# --seed 0 \

#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------- Hybrid Paper --------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

################################
#---------- ORCA-Score ---------
################################

# CUDA_VISIBLE_DEVICES=1 python Runners/Planning/ORCA.py \
# --exp_name Hybrid_ORCA_score_knn2_01decay_7balls \
# --is_decay True \
# --recover False \
# --eval_num 100 \
# --horizon 100 \
# --num_objs 7 \
# --mode eval \
# --score_mode hybrid \
# --env hybrid \
# --orca_mode score \
# --knn_orca 2 \
# --target_t0 0.1 \
# --seed 1 \

# CUDA_VISIBLE_DEVICES=2 python Runners/Planning/ORCA.py \
# --exp_name Hybrid_ORCA_score_knn2_01decay_7balls_force \
# --is_decay True \
# --is_onebyone False \
# --is_pid True \
# --recover False \
# --action_type force \
# --eval_num 100 \
# --horizon 300 \
# --num_objs 7 \
# --mode eval \
# --score_mode hybrid \
# --env hybrid \
# --orca_mode score \
# --knn_orca 2 \
# --target_t0 0.1 \
# --seed 1 \

# CUDA_VISIBLE_DEVICES=1 python Runners/Planning/ORCA.py \
# --exp_name Hybrid_ORCA_score_knn2_001_7balls \
# --is_decay False \
# --recover False \
# --eval_num 100 \
# --horizon 200 \
# --num_objs 7 \
# --mode debug \
# --score_mode hybrid \
# --env hybrid \
# --orca_mode score \
# --knn_orca 2 \
# --target_t0 0.01 \
# --seed 1 \

# CUDA_VISIBLE_DEVICES=3 python Runners/Planning/ORCA.py \
# --exp_name Hybrid_ORCA_score_knn2_001_7balls_force \
# --is_decay False \
# --is_onebyone False \
# --is_pid True \
# --recover False \
# --action_type force \
# --eval_num 100 \
# --horizon 300 \
# --num_objs 7 \
# --mode debug \
# --score_mode hybrid \
# --env hybrid \
# --orca_mode score \
# --knn_orca 2 \
# --target_t0 0.01 \
# --seed 1 \

# CUDA_VISIBLE_DEVICES=1 python Runners/Planning/ORCA.py \
# --exp_name Hybrid_ORCA_score_knn2_001_5balls_onebyone \
# --is_onebyone True \
# --is_pid False \
# --is_decay False \
# --recover False \
# --action_type vel \
# --num_objs 5 \
# --knn_orca 2 \
# --horizon 300 \
# --mode debug \
# --eval_num 100 \
# --env sorting \
# --score_mode sorting \
# --orca_mode score \
# --target_t0 0.01 \
# --seed 0 \


################################
#---------- ORCA-Goal ----------
################################

# CUDA_VISIBLE_DEVICES=2 python Runners/Planning/ORCA.py \
# --exp_name Hybrid_ORCA_VAEgoal_knn2_7balls \
# --recover True \
# --eval_num 100 \
# --horizon 100 \
# --mode eval \
# --num_objs 7 \
# --score_mode hybrid \
# --goal_mode Score \
# --env hybrid \
# --orca_mode goal \
# --knn_orca 2 \
# --seed 1 \
