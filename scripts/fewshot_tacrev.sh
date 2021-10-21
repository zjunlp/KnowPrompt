
# by default t_lambda 0.001, init 1 1, init answer words 0, use 1


#1 normal
CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs=20  --num_workers=8 \
    --model_name_or_path roberta-large \
    --accumulate_grad_batches 1 \
    --batch_size 16 \
    --data_dir dataset/tacrev/k-shot/16-1 \
    --check_val_every_n_epoch 5 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --t_lambda 0.001 \
    --wandb \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 3e-5 \
    --init_answer_words 1 \
    --init_type_words 1 \
    --init_answer_words_by_one_token 0  \
    --use_template_words 1



#2 no init answer words
CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs=20  --num_workers=8 \
    --model_name_or_path roberta-large \
    --accumulate_grad_batches 1 \
    --batch_size 16 \
    --data_dir dataset/tacrev/k-shot/16-1 \
    --check_val_every_n_epoch 5 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --t_lambda 0.001 \
    --wandb \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 3e-5 \
    --init_answer_words 0 \
    --init_type_words 1 \
    --init_answer_words_by_one_token 0  \
    --use_template_words 1

# no ke in answer words only one token
CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs=20  --num_workers=8 \
    --model_name_or_path roberta-large \
    --accumulate_grad_batches 1 \
    --batch_size 16 \
    --data_dir dataset/tacrev/k-shot/16-1 \
    --check_val_every_n_epoch 5 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --t_lambda 0.001 \
    --wandb \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 3e-5 \
    --init_answer_words 1 \
    --init_type_words 1 \
    --init_answer_words_by_one_token 1  \
    --use_template_words 1

# no template words
CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs=20  --num_workers=8 \
    --model_name_or_path roberta-large \
    --accumulate_grad_batches 1 \
    --batch_size 16 \
    --data_dir dataset/tacrev/k-shot/16-1 \
    --check_val_every_n_epoch 5 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --t_lambda 0.001 \
    --wandb \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 3e-5 \
    --init_answer_words 1 \
    --init_type_words 1 \
    --init_answer_words_by_one_token 0  \
    --use_template_words 0


# no type words knowledge
CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs=20  --num_workers=8 \
    --model_name_or_path roberta-large \
    --accumulate_grad_batches 1 \
    --batch_size 16 \
    --data_dir dataset/tacrev/k-shot/16-1 \
    --check_val_every_n_epoch 5 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --t_lambda 0.001 \
    --wandb \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 3e-5 \
    --init_answer_words 1 \
    --init_type_words 0 \
    --init_answer_words_by_one_token 0  \
    --use_template_words 1
    
# no structual constrain
CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs=20  --num_workers=8 \
    --model_name_or_path roberta-large \
    --accumulate_grad_batches 1 \
    --batch_size 16 \
    --data_dir dataset/tacrev/k-shot/16-1 \
    --check_val_every_n_epoch 5 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --t_lambda 0.000 \
    --wandb \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 3e-5 \
    --init_answer_words 1 \
    --init_type_words 1 \
    --init_answer_words_by_one_token 0  \
    --use_template_words 1

# 8-shot 32.2 no ablation
# vitual answer word 26.6
# no knowledge injection 26.2
# no template words 32.5
# no knowledge injection into type words 32.8
# no structual constrain 31.7



# 16 shot

# 32.4
# 31.5
# 33.3
# 33.4
# 32.7


# by default t_lambda 0.001, init 1 1, init answer words 0, use 1


#1 normal
CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs=5  --num_workers=8 \
    --model_name_or_path roberta-large \
    --accumulate_grad_batches 1 \
    --batch_size 16 \
    --data_dir dataset/tacrev \
    --check_val_every_n_epoch 5 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --t_lambda 0.001 \
    --wandb \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 3e-5 \
    --init_answer_words 1 \
    --init_type_words 1 \
    --init_answer_words_by_one_token 0  \
    --use_template_words 1



#2 no init answer words
CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs=5  --num_workers=8 \
    --model_name_or_path roberta-large \
    --accumulate_grad_batches 1 \
    --batch_size 16 \
    --data_dir dataset/tacrev \
    --check_val_every_n_epoch 5 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --t_lambda 0.001 \
    --wandb \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 3e-5 \
    --init_answer_words 0 \
    --init_type_words 1 \
    --init_answer_words_by_one_token 0  \
    --use_template_words 1

# no ke in answer words only one token
CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs=5  --num_workers=8 \
    --model_name_or_path roberta-large \
    --accumulate_grad_batches 1 \
    --batch_size 16 \
    --data_dir dataset/tacrev \
    --check_val_every_n_epoch 5 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --t_lambda 0.001 \
    --wandb \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 3e-5 \
    --init_answer_words 1 \
    --init_type_words 1 \
    --init_answer_words_by_one_token 1  \
    --use_template_words 1

# no template words
CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs=5  --num_workers=8 \
    --model_name_or_path roberta-large \
    --accumulate_grad_batches 1 \
    --batch_size 16 \
    --data_dir dataset/tacrev \
    --check_val_every_n_epoch 5 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --t_lambda 0.001 \
    --wandb \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 3e-5 \
    --init_answer_words 1 \
    --init_type_words 1 \
    --init_answer_words_by_one_token 0  \
    --use_template_words 0


# no type words knowledge
CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs=5  --num_workers=8 \
    --model_name_or_path roberta-large \
    --accumulate_grad_batches 1 \
    --batch_size 16 \
    --data_dir dataset/tacrev \
    --check_val_every_n_epoch 5 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --t_lambda 0.001 \
    --wandb \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 3e-5 \
    --init_answer_words 1 \
    --init_type_words 0 \
    --init_answer_words_by_one_token 0  \
    --use_template_words 1
    
# no structual constrain
CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs=5  --num_workers=8 \
    --model_name_or_path roberta-large \
    --accumulate_grad_batches 1 \
    --batch_size 16 \
    --data_dir dataset/tacrev \
    --check_val_every_n_epoch 5 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --t_lambda 0.000 \
    --wandb \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 3e-5 \
    --init_answer_words 1 \
    --init_type_words 1 \
    --init_answer_words_by_one_token 0  \
    --use_template_words 1

# 8-shot 32.2 no ablation
# vitual answer word 26.6
# no knowledge injection 26.2
# no template words 32.5
# no knowledge injection into type words 32.8
# no structual constrain 31.7



# 16 shot

# 32.4
# 31.5
# 33.3
# 33.4
#