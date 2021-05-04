CUDA_VISIBLE_DEVICES=1 python train.py --expname rQdia-curl-cheetah-run \
    --domain_name cheetah \
    --task_name run \
    --encoder_type pixel \
    --action_repeat 8 \
    --pre_transform_image_size 84 --image_size 84 \
    --agent curl_sac --frame_stack 3 \
    --seed -1 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 1000000