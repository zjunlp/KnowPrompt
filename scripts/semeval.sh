CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs=10  --num_workers=8 \
    --model_name_or_path roberta-large \
    --accumulate_grad_batches 1 \
    --batch_size 16 \
    --data_dir dataset/semeval \
    --check_val_every_n_epoch 1 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --t_lambda 0.001 \
    --wandb \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 3e-5

# 90.2