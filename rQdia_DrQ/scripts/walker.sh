CUDA_VISIBLE_DEVICES=2 python train.py --expname rQdia-dqr-walker-walk \
    --domain_name walker \
    --task_name walk \
    --encoder_type pixel \
    --action_repeat 2 \
    --pre_transform_image_size 84 --image_size 84 \
    --agent curl_sac --frame_stack 3 \
    --seed ${1:--1} --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 500000