import logging
import os
import sys
import json
import numpy as np
from typing import Dict

import datasets
import transformers
import torch
from transformers import set_seed, Trainer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from os.path import join
from arguments import get_args
from tasks.utils import *
import warnings
import time

import wandb
os.environ["WANDB_MODE"] = "offline"
 
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def init_wandb(args, model):
    """
    If you are using wandb, please config you wandb keys and project here: replace "your name" and "your key".
    And uncomment the following lines.
    """
    # os.environ["WANDB_API_KEY"] = "your key"
    # if args.wandb_resume:
    #     wandb.init(project=args.wandb_project, entity="your name", resume="must", id=args.wandb_resume)
    # else:
    #     wandb.init(project=args.wandb_project, entity="your name")
    # wandb.config = {
    #         "epochs": args.num_train_epochs,
    #         "batch_size": args.per_device_train_batch_size,
    #         }
    # wandb.watch(model)
    pass # delete this line if you are using wandb

class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()

def train(trainer, resume_from_checkpoint=None, last_checkpoint=None):
    checkpoint = None

    print("start training")
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.log_best_metrics()


def evaluate(trainer, local_rank, to_wandb):
    logger.info("*** Evaluate ***")

    eval_metrics = trainer.evaluate(eval_dataset=trainer.eval_dataset)
    if 'eval_BleuScore' in eval_metrics:
        eval_bleu = eval_metrics.pop('eval_BleuScore')
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    if local_rank == 0 and to_wandb:
        wandb.log({"eval/loss": eval_metrics['eval_loss']})

    test_metrics = trainer.evaluate(eval_dataset=trainer.predict_dataset, metric_key_prefix="test",)
    if 'test_BleuScore' in test_metrics:
        test_bleu = test_metrics.pop('test_BleuScore')
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)


def predict(trainer, predict_dataset=None):
    if predict_dataset is None:
        logger.info("No dataset is available for testing")

    elif isinstance(predict_dataset, dict):

        for dataset_name, d in predict_dataset.items():
            logger.info("*** Predict: %s ***" % dataset_name)
            predictions, labels, metrics = trainer.predict(
                d, metric_key_prefix="predict"
            )
            predictions = predictions.numpy()
            if 'test_BleuScore' in metrics:
                test_bleu = metrics.pop('test_BleuScore')
            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

    else:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(
            predict_dataset, metric_key_prefix="predict"
        )
        if 'predict_BleuScore' in metrics:
            predict_bleu = metrics.pop('predict_BleuScore')
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        
        with open(os.path.join(training_args.output_dir, "predictions.json"), "w") as f:
            json.dump(predictions.tolist(), f, indent=4)
        with open(os.path.join(training_args.output_dir, "labels.json"), "w") as f:
            json.dump(labels.tolist(), f, indent=4)

if __name__ == "__main__":
    args = get_args()

    model_args, data_args, training_args = args
    if training_args.local_rank == 0:
        print('################### initializing ##################')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if not os.path.isdir("checkpoints") or not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    if data_args.dataset_name.lower() in ['vqa', 'flickr']:
        from tasks.vqa.get_trainer_prepared import get_trainer
    else:
        raise NotImplementedError(
            "Task {} is not implemented. Please choose a task from: {}".format(data_args.dataset_name))

    set_seed(training_args.seed)

    if training_args.local_rank == 0:
        print('################### loading ##################')
    model, trainer, predict_dataset = get_trainer(args)
    #for wandb
    if training_args.local_rank == 0 and training_args.to_wandb:
        init_wandb(training_args, model)


    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if training_args.local_rank == 0:
        print('################### training ##################')
    if training_args.do_train:
        train(trainer, training_args.resume_from_checkpoint, last_checkpoint)

    if training_args.do_eval:
        evaluate(trainer, training_args.local_rank, training_args.to_wandb)

    if training_args.do_predict:
        predict(trainer, predict_dataset)

    if training_args.to_wandb and training_args.local_rank == 0:
        wandb.finish(0)
