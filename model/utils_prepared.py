

from transformers import (
    BertModel,
    RobertaModel,
    AlbertModel,
    DebertaV2Model,
    XLNetModel,
    DebertaV2Model,
    AutoConfig
)
import torch

from model.blip2.modeling_blip_2 import Blip2ForConditionalGeneration
from model.instructblip_ex.modeling_instructblip_prepared_condition import InstructBlipForConditionalGeneration

from transformers.deepspeed import is_deepspeed_zero3_enabled

MODEL_CLASS = {
    "blip-2": Blip2ForConditionalGeneration,
    "instructblip": InstructBlipForConditionalGeneration,

}


def get_model(model_args, local_rank, config: AutoConfig, fix_bert: bool = False):

    model_class = MODEL_CLASS[config.model_type]
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        _fast_init = not is_deepspeed_zero3_enabled(),
    )
    # first freeze all layers
    # keep LM frozen
    for param in model.parameters():
        param.requires_grad = False

    # fine-tune the condition_projection 
    for param in model.qformer.embeddings.condition_projection.parameters():
        param.requires_grad = True
    #for param in model.qformer.embeddings.pre_cond_layernorm.parameters():
    #    param.requires_grad = True

    if model_args.condition_from == 'both':
        if local_rank == 0:
            print("model_args.condition_from", model_args.condition_from)
        for param in model.qformer.embeddings.condition_mid_projection.parameters():
            param.requires_grad = True
    # unfreeze language_projection
    #for param in model.language_projection.parameters():
    #    param.requires_grad = True

    # unfreeze qformer
    if model_args.unfreeze_qformer:
        for param in model.qformer.parameters():
            param.requires_grad = True
    # unfreeze qtoken
    if model_args.unfreeze_qtoken:
        model.query_tokens.requires_grad = True
    # unfreeze llm
    if model_args.unfreeze_llm:
        for param in model.language_projection.parameters():
            param.requires_grad = True
        for block in model.language_model.encoder.block:
            block.layer[0].SelfAttention.q.weight.requires_grad=True
            block.layer[0].SelfAttention.v.requires_grad=True
        for block in model.language_model.decoder.block:
            block.layer[0].SelfAttention.q.weight.requires_grad=True
            block.layer[0].SelfAttention.v.requires_grad=True
            block.layer[1].EncDecAttention.q.requires_grad=True
            block.layer[1].EncDecAttention.v.requires_grad=True

    all_param = 0
    trained_param=0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad ==True:
            trained_param+=param.numel()
    total_param = all_param 

    if local_rank == 0:
        print('***** total param is {} *****'.format(total_param))
        print('***** total trained param is {} *****'.format(trained_param))
    return model
