CUDA_VISIBLE_DEVICES=1 python train-mp.py\
    --domain_name reacher \
    --task_name easy \
    --encoder_type pixel \
    --action_repeat 4 \
    --pre_transform_image_size 84 --image_size 84 \
    --agent curl_sac --frame_stack 3 \
    --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 500000