from enum import Enum
import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments

from tasks.utils import *


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.training_args
    """

    dataset_name: str = field(
        metadata={
            "help": "The name of the dataset to use: " + ", ".join(DATASETS),
            "choices": DATASETS
        }
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    preprared_cond: Optional[bool] = field(
        default = True, metadata={"help": "prepared or not"}
    )
    load_cond_size: Optional[str] = field(
        default = 'xl', metadata={"help": "the size of condition to be loaded, xl-2048, xxl-4096"}
    )
    related_shot: Optional[str] = field(
            default=None, metadata={"help": "whether to use related shots for stage1 data, 'related', 'related_plus'"}
    )
    with_answer: Optional[bool] = field(
            default=False, metadata={"help": "whether to use the condition generated with answer"}
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )

    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    # NOTE 没用到
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    # NOTE 没用到
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    # NOTE 没用到
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default='dialog_version_control/data/ATIS/train.json', metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default='dialog_version_control/data/ATIS/test.json',
        metadata={"help": "A csv or a json file containing the test data."}
    )
    label_file: Optional[str] = field(
        default='dialog_version_control/data/ATIS/label.txt',
        metadata={"help": "A txt file containing the label data."}
    )
    dev_rate: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "For spliting a dev set"
        },
    )
    use_preprocessed: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to use preprocessed data"
        },
    )
    done_preprocess: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether has finished the data preprocess "
        },
    )
    load_datatype: Optional[str] = field(
        default=None,
        metadata={
            "help": "MIC_full or pretrain"
        },
    )
    mix_blip2: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to mix blip2 captions in stage1 training"
        },
    )
    drop_input: Optional[str] = field(
        default=None,
        metadata={
            "help": "whether to drop input contain: 'image', 'text', 'both', None"
        },
    )

    only_evaluate: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to only test the result"
        },
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    unfreeze_llm: Optional[bool] = field(
            default = False, metadata = {'help': "whether to unfreeze llm, for stage 2"}
    )
    unfreeze_qformer: Optional[bool] = field(
            default = False, metadata = {'help': "whether to unfreeze llm, for stage 2"}
    )
    unfreeze_qtoken: Optional[bool] = field(
            default = False, metadata = {'help': "whether to unfreeze llm, for stage 2"}
    )

    global_calculation: Optional[str] = field(
        default = None, metadata = {'help': "how to add condition vector"}
    )
    condition_from: Optional[str] = field(
            default = None, metadata = {"help": "where to get the condition vector, if 'both', use both middle and last"}
    )
    send_condition_to_llm: Optional[bool] = field(
            default = False, metadata = {"help": "whether append a <condition> token as the beginning of LLM input"}
    )

    # NOTE 没用到
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    # NOTE 没用到
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    # NOTE 没用到
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    # NOTE 没用到
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    # NOTE 没用到
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    prefix: bool = field(
        default=False,
        metadata={
            "help": "Will use P-tuning v2 during training"
        }
    )
    prompt: bool = field(
        default=False,
        metadata={
            "help": "Will use prompt tuning during training"
        }
    )
    pre_seq_len: int = field(
        default=6,
        metadata={
            "help": "The length of prompt"
        }
    ) # 可能会引起误会，datasets内也定义了pre_seq_len
    task_type: Optional[str] = field(
        default="language_modeling",
        metadata={
            "help": "Design which head to use."
        }
    )
    eval_type: Optional[str] = field(
        default="eval",
        metadata={
            "help": "Design which head to use."
        }
    )
    prompt_type: Optional[str] = field(
        default="soft",
        metadata={
            "help": "Use hard or soft prompt"
        }
    )
    template_id: Optional[str] = field(
        default="template_0",
        metadata={
            "help": "The specific soft prompt template to use"
        }
    )
    verbalizer_id: Optional[str] = field(
        default="verbalizer_0",
        metadata={
            "help": "The specific verbalizer to use"
        }
    )
    prompt_operation: Optional[str] = field(
        default="mean",
        metadata={
            "help": "Will use max, sum, mean, attention or cross-attention soft prompt tuning during training"
        }
    )
    hidden_dropout_prob: float = field(
        default=0.1,
        metadata={
            "help": "The dropout probability used in the models"
        }
    )
    num_attention_layers: int = field(
        default=1,
        metadata={
            "help": ""
        }
    )
    num_attention_heads: int = field(
        default=8,
        metadata={
            "help": ""
        }
    )
    whether_PositionalEncoding: bool = field(
        default=True,
        metadata={
            "help": ""
        }
    )
    whether_PositionalWiseFeedForward: bool = field(
        default=True,
        metadata={
            "help": ""
        }
    )
    fix_deberta: bool = field(
        default=True,
        metadata={
            "help": ""
        }
    )
    data_augmentation: Optional[str] = field(
        default="none",
        metadata={
            "help": "rdrop, AT, mixup, manifold_mixup"
        }
    )
    model_type: Optional[str] = field(
        default="blip2",
        metadata={
            "help": "blip2, instructblip"
        }
    )
    image_place_holder: Optional[str] = field(
        default=None,
        metadata={
            "help": "placeholder for image, e.g. 图"
        }
    )
    label: Optional[str] = field(
        default="label",
        metadata={
            "help": ""
        }
    )
    experiment_name: Optional[str] = field(
        default="label",
        metadata={
            "help": ""
        }
    )
# Negative Sample
    negative_sample_num: Optional[int] = field(
        default=1,
        metadata={
            "help": ""
        }

    )
    processor_path: Optional[str] = field(
        default=None,
        metadata={
            "help": ""
        }
    )
    backbone_model: Optional[str] = field(
        default="flan-t5",
        metadata={
            "help": "flan-t5,opt,vicuna"
        }
    )


@dataclass
class ExtraTrainingArguments(TrainingArguments):
    # loss for the 1st forward
    dual_loss: Optional[bool] = field(
        default = False, metadata = {"help": "whether to compute loss during the first forward"}
    )
    # random_condition
    random_condition: Optional[bool] = field(
        default = False, metadata = {"help": "replace condition with random vector"}
    )
    dual_none_condition: Optional[bool] = field(
        default = False, metadata = {"help": "replace condition with random vector"}
    )
    none_condition: Optional[bool] = field(
        default = False, metadata = {"help": "replace condition with random vector"}
    )
    # whether to see loss without condition during eval_loop
    eval_wo_cond: Optional[bool] = field(
        default = False, metadata = {"help": "whether to see loss without condition during eval_loop"}
    )

    # for wandb
    to_wandb: Optional[bool] = field(
            default = True, metadata = {"help": "whether to use wandb"})
    wandb_resume: Optional[str] = field(
            default = None, metadata = {"help": "to resume from existing training logs, e.g.: 38td18ni"}
            )
    wandb_project: Optional[str] = field(
            default = "global_condition", metadata = {"help": "wandb project name"})

    # learning rates for different layers 
    qformer_lr: float = field(
        default=1e-5, metadata={"help":""}
    )
    condition_projection_lr: float = field(
        default=1e-5, metadata={"help":""}
    )
    llm_lr: float = field(
        default=1e-5, metadata={"help":""}
    )

    max_label_length: Optional[int] = field(  
        default=32, 
        metadata={
            "help": "max_label_length"
        }
    )

    generation_max_length: Optional[int] = field(
        default=32, 
        metadata={
            "help": "generation_max_length"
        }
    )
    generation_min_length: Optional[int] = field(
        default=1,
        metadata={
            "help": "generation_min_length"
        }
    )
    generation_num_beams: Optional[int] = field(
        default=5,
        metadata={
            "help": "generation_num_beams"
        }
    )
    predict_with_generate: bool = field(
        default=True,
        metadata={
            "help": ""
        }
    )
    multiple_choice : bool = field(
        default=False,
        metadata={
            "help": ""
        }
    )
    few_shot : bool = field(
        default=False,
        metadata={
            "help": ""
        }
    )
    using_instruct_qformer: bool = field(
        default=True,
        metadata={
            "help": ""
        }
    )
    full_bf16_training: bool = field(
        default=False,
        metadata={
            "help": "WHETHER TO USE BF16 full TRAINING"
        }
    )

def get_args():
    """Parse all the args."""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ExtraTrainingArguments))

    args = parser.parse_args_into_dataclasses()

    return args
