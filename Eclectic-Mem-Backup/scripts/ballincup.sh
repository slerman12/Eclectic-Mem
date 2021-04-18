CUDA_VISIBLE_DEVICES=2 python train.py \
    --domain_name ball_in_cup \
    --task_name catch \
    --encoder_type pixel \
    --action_repeat 8 \
    --pre_transform_image_size 100 --image_size 84 \
    --agent em_sac --frame_stack 3 \
    --seed -1 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 1000000