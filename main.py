"""Experiment-running framework."""
import argparse
import importlib
from logging import debug

import numpy as np
from pytorch_lightning.trainer import training_tricks
import torch
import pytorch_lightning as pl
import lit_models
import yaml
import time
from lit_models import TransformerLitModelTwoSteps
from transformers import AutoConfig, AutoModel
from pytorch_lightning.plugins import DDPPlugin
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In order to ensure reproducible experiments, we must set random seeds.


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--litmodel_class", type=str, default="TransformerLitModel")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data_class", type=str, default="DIALOGUE")
    parser.add_argument("--lr_2", type=float, default=3e-5)
    parser.add_argument("--model_class", type=str, default="bert.BertForSequenceClassification")
    parser.add_argument("--two_steps", default=False, action="store_true")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    
    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"data.{temp_args.data_class}")
    model_class = _import_class(f"models.{temp_args.model_class}")
    litmodel_class = _import_class(f"lit_models.{temp_args.litmodel_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    litmodel_class.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser

device = "cuda"
from tqdm import tqdm
def _get_relation_embedding(data):
    train_dataloader = data.train_dataloader()
    #! hard coded
    relation_embedding = [[] for _ in range(36)]
    model = AutoModel.from_pretrained('bert-base-uncased')
    model.eval()
    model = model.to(device)


    cnt = 0
    for batch in tqdm(train_dataloader):
        with torch.no_grad():
            #! why the sample in this case will cause errors
            if cnt == 416:
                continue
            cnt += 1
            input_ids, attention_mask, token_type_ids , labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state.detach().cpu()
            _, mask_idx = (input_ids == 103).nonzero(as_tuple=True)
            bs = input_ids.shape[0]
            mask_output = logits[torch.arange(bs), mask_idx] # [batch_size, hidden_size]
            

            labels = labels.detach().cpu()
            mask_output = mask_output.detach().cpu()
            assert len(labels[0]) == len(relation_embedding)
            for batch_idx, label in enumerate(labels.tolist()):
                for i, x in enumerate(label):
                    if x:
                        relation_embedding[i].append(mask_output[batch_idx])
    
    # get the mean pooling
    for i in range(36):
        if len(relation_embedding[i]):
            relation_embedding[i] = torch.mean(torch.stack(relation_embedding[i]), dim=0)
        else:
            relation_embedding[i] = torch.rand_like(relation_embedding[i-1])

    del model
    return relation_embedding




        
        





def main():
    parser = _setup_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)
    data_class = _import_class(f"data.{args.data_class}")
    model_class = _import_class(f"models.{args.model_class}")
    litmodel_class = _import_class(f"lit_models.{args.litmodel_class}")

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    data = data_class(args, model)
    data_config = data.get_data_config()
    model.resize_token_embeddings(len(data.tokenizer))

    

    # gpt no config?

    # if "gpt" in args.model_name_or_path or "roberta" in args.model_name_or_path:
    #     tokenizer = data.get_tokenizer()
    #     model.resize_token_embeddings(len(tokenizer))
    #     model.update_word_idx(len(tokenizer))
    #     if "Use" in args.model_class:
    #         continous_prompt = [a[0] for a in tokenizer([f"[T{i}]" for i in range(1,3)], add_special_tokens=False)['input_ids']]
    #         continous_label_word = [a[0] for a in tokenizer([f"[class{i}]" for i in range(1, data.num_labels+1)], add_special_tokens=False)['input_ids']]
    #         discrete_prompt = [a[0] for a in tokenizer(['It', 'was'], add_special_tokens=False)['input_ids']]
    #         dataset_name = args.data_dir.split("/")[1]
    #         model.init_unused_weights(continous_prompt, continous_label_word, discrete_prompt, label_path=f"{args.model_name_or_path}_{dataset_name}.pt")
    # data.setup()
    # relation_embedding = _get_relation_embedding(data)
    lit_model = litmodel_class(args=args, model=model, tokenizer=data.tokenizer)
    data.tokenizer.save_pretrained('test')


    logger = pl.loggers.TensorBoardLogger("training/logs")
    dataset_name = args.data_dir.split("/")[-1]
    if args.wandb:
        logger = pl.loggers.WandbLogger(project="dialogue_pl", name=f"{dataset_name}")
        logger.log_hyperparams(vars(args))
    
    # init callbacks
    early_callback = pl.callbacks.EarlyStopping(monitor="Eval/f1", mode="max", patience=5,check_on_train_epoch_end=False)
    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="Eval/f1", mode="max",
        filename='{epoch}-{Eval/f1:.2f}',
        dirpath="output",
        save_weights_only=True
    )
    callbacks = [early_callback, model_checkpoint]

    # args.weights_summary = "full"  # Print full summary of the model
    gpu_count = torch.cuda.device_count()
    accelerator = "ddp" if gpu_count > 1 else None


    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, default_root_dir="training/logs", gpus=gpu_count, accelerator=accelerator,
        plugins=DDPPlugin(find_unused_parameters=False) if gpu_count > 1 else None,
    )

    # trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(lit_model, datamodule=data)




    # two steps

    path = model_checkpoint.best_model_path
    print(f"best model save path {path}")

    if not os.path.exists("config"):
        os.mkdir("config")
    config_file_name = time.strftime("%H:%M:%S", time.localtime()) + ".yaml"
    day_name = time.strftime("%Y-%m-%d")
    if not os.path.exists(os.path.join("config", day_name)):
        os.mkdir(os.path.join("config", time.strftime("%Y-%m-%d")))
    config = vars(args)
    config["path"] = path
    with open(os.path.join(os.path.join("config", day_name), config_file_name), "w") as file:
        file.write(yaml.dump(config))

    # lit_model.load_state_dict(torch.load(path)["state_dict"])


    if not args.two_steps: trainer.test()
    step2_model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="Eval/f1", mode="max",
        filename='{epoch}-{Step2Eval/f1:.2f}',
        dirpath="output",
        save_weights_only=True
    )

    if args.two_steps:
        # we build another trainer and model for the second training
        # use the Step2Eval/f1 

        # lit_model_second = TransformerLitModelTwoSteps(args=args, model=lit_model.model, data_config=data_config)
        step_early_callback = pl.callbacks.EarlyStopping(monitor="Eval/f1", mode="max", patience=6, check_on_train_epoch_end=False)
        callbacks = [step_early_callback, step2_model_checkpoint]
        trainer_2 = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, default_root_dir="training/logs", gpus=gpu_count, accelerator=accelerator,
            plugins=DDPPlugin(find_unused_parameters=False) if gpu_count > 1 else None,
        )
        trainer_2.fit(lit_model, datamodule=data)
        trainer_2.test()
        # result = trainer_2.test(lit_model, datamodule=data)[0]
        # with open("result.txt", "a") as file:
        #     a = result["Step2Test/f1"]
        #     file.write(f"test f1 score: {a}\n")
        #     file.write(config_file_name + '\n')

    # trainer.test(datamodule=data)


if __name__ == "__main__":

    main()
