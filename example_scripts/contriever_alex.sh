torchrun --standalone --nnodes=1 --nproc_per_node=3 train.py \
--retriever_model_id bert-base-uncased --pooling average --augmentation delete \
--prob_augmentation 0.1 --train_data "../contriever/data_scripts/encoded-data/bert-base-uncased/en_XX" \
--loading_mode split --ratio_min 0.1 --ratio_max 0.5 --chunk_length 256 \
--momentum 0.9995 --queue_size 63936 --temperature 0.05 --warmup_steps 2000 \
--total_steps 5000 --lr 0.00005 --scheduler linear --optim adamw --per_gpu_batch_size 32 \
--output_dir ./checkpoints --torchrun --reset_params --contrastive_mode simoco