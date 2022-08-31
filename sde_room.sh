CUDA_VISIBLE_DEVICES=0 python Runners/RoomSDE.py \
--exp_name M4D26_bedroom_target_128_64_dataM12D25 \
--data_name UnshuffledRoomsMeta \
--test_decay False \
--room_type bedroom \
--score_mode target \
--full_train False \
--n_epoches 10000 \
--eval_freq 1 \
--batch_size 256 \
--lr 2e-4 \
--t0 0.1 \
--test_ratio 0.1 \
--hidden_dim 128 \
--embed_dim 64 \
--base_noise_scale 0.01 \

