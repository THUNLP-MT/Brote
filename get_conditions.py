# For T5 based model
import sys, os, datasets, json, re, base64
import pandas as pd
import pyarrow.parquet as pq
import transformers
from io import BytesIO
from collections.abc import Mapping
from tqdm import tqdm

import numpy as np
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from model.instructblip import InstructBlipConfig, InstructBlipModel, InstructBlipPreTrainedModel,InstructBlipForConditionalGeneration,InstructBlipProcessor

from PIL import Image

import argparse

class ConditionGenerator:
    def __init__(self, args):
        if args.model_path is not None:
            model_ckpt = args.model_path
        else:
            model_ckpt = f"BleachNick/MMICL-Instructblip-T5-{args.model_size}"
        config = InstructBlipConfig.from_pretrained(model_ckpt)
    
        if args.processor_path is not None:
            processor_ckpt = args.processor_path
        else:
            processor_ckpt = f"Salesforce/instructblip-flan-t5-{args.model_size}"
        self.processor = InstructBlipProcessor.from_pretrained(processor_ckpt)
    
        self.model = InstructBlipForConditionalGeneration.from_pretrained(model_ckpt,
                config=config).to('cuda:0',dtype=torch.bfloat16) # or torch.float if GPU memory is adequate 
         
        self.image_placeholder="图"
        sp = [self.image_placeholder]+[f"<image{i}>" for i in range(20)]
        sp = sp + self.processor.tokenizer.additional_special_tokens[len(sp):]
        self.processor.tokenizer.add_special_tokens({'additional_special_tokens':sp})
        if self.model.qformer.embeddings.word_embeddings.weight.shape[0] != len(self.processor.qformer_tokenizer):
            self.model.qformer.resize_token_embeddings(len(self.processor.qformer_tokenizer))
    
        self.replace_token="".join(32*[self.image_placeholder])

        parquet_file = pq.ParquetFile(args.input_data)
        df = pd.concat([item.to_pandas() for item in parquet_file.iter_batches(batch_size=600)]).reset_index(drop=True)
        self.dataset = datasets.Dataset.from_pandas(df)

    
    def preprocess_function_batched_base64(self, examples):
        result = self.processor.tokenizer(examples["input_text"].replace("图", "".join(32*["图"])),
                padding="max_length", max_length=512, truncation=True)
        result['label'] = self.processor.tokenizer(examples["output_text"], padding="max_length", max_length=32, truncation=True)["input_ids"]
        result["pixel_values"] = []
    
        if os.path.isfile(examples["input_image"][0]):
            for img_path in examples["input_image"]:
                img = Image.open(img_path)
                result["pixel_values"].append(self.processor(images = img)["pixel_values"][0])
        else:
            for basrr64_img in examples["input_image"]:
                img = load_base64_image(basrr64_img)
                result["pixel_values"].append(self.processor(images = img)["pixel_values"][0])
    
        return result    

    def process_batch(self, features):
        features = [self.preprocess_function_batched_base64(f) for f in features]

        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}

        if "label" in first and first["label"] is not None: # for training & eval batch
            label = first["label"][0].item() if isinstance(first["label"], torch.Tensor) else first["label"][0]
            dtype = torch.long  if isinstance(label, int) else torch.float
            batch["labels"] = torch.stack([torch.Tensor(f["label"]) for f in features]).to(dtype)
        elif "label_ids" in first and first["label_ids"] is not None: # for training & eval batch
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)
        IGNORE_INDEX=-100
        batch["labels"][torch.where(batch["labels"] == 0)]= IGNORE_INDEX # pad_token_id: 0

        ignored_keys = ['input_text', 'input_image', 'output_text', 'output_image','pixel_values', 'images', 'gpt_caption', 'model', 'question_id']
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str) and k not in ignored_keys :
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
            if k == 'pixel_values':
                max_image_length = max([len(f[k]) for f in features])
                image_list=[]
                mask_list= []
                for f in features:
                    image,img_mask = padd_images(f[k],max_image_length)
                    image_list.append(image)
                    mask_list.append(img_mask)
                batch[k] = torch.stack(image_list)
                batch['img_mask'] = torch.stack(mask_list)
        batch['sp_token'] = 32100
        return batch

def padd_images(image, max_length):
    image = torch.tensor(np.array(image))
    mask = torch.zeros(max_length).bool()
    pad_len = max_length - image.shape[0]
    mask[:image.shape[0]] = True
    image = pad(image,(0,0,0,0,0,0,0,pad_len)) # padding behind the first dim
    return image,mask

def load_base64_image(base64_str):
    image = base64.b64decode(base64_str)
    image = BytesIO(image)
    image = Image.open(image)

    return image


def get_conditions(cond_gen, args):
    cond_gen.model.eval()
    dataloader = DataLoader(cond_gen.dataset, batch_size=args.batch_size, collate_fn=cond_gen.process_batch, num_workers=args.num_workers)
    rst = []
    rst_mid = []
    i = 0
    total_fail = 0
    total_gen = 0
    hidden_units = 4096 if args.model_size == 'xxl' else 2048
    with torch.no_grad():
        for inputs in tqdm(dataloader):
            i+=1
            inputs = batch
            try:
                outputs = model.generate(
                    pixel_values = inputs['pixel_values'].to('cuda'),
                    input_ids = inputs['input_ids'].to('cuda'),
                    attention_mask = inputs['attention_mask'].to('cuda'),
                    img_mask = inputs['img_mask'].to('cuda'),
                    do_sample = False,
                    max_length = 50,
                    min_length = 1,
                    set_min_padding_size = False,
                    generate_conditions = True if not args.generate_caption else False, 
                    condition_from = args.condition_from,
                )
                rst.append(outputs.cpu())
                total_gen += args.batch_size
            except:
                print(f'\nfailed at {i}th batch, batch_size={args.batch_size}, continue to the next batch')
                rst.append(torch.zeros(args.batch_size, hidden_units))
                total_fail += args.batch_size
    rst = torch.concat(rst, 0)
    print(f'total_fail: {total_fail}, total_gen: {total_gen}')
    return rst.tolist(), []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", required=True, type=str, help="ends with 'parquet.gzip'")
    parser.add_argument("--output_name", required=True, type=str, help="ends with 'parquet.gzip'")
    parser.add_argument("--output_dir", default='MIC_sample', type=str, help="data dir to place the generated condition context")
    parser.add_argument("--model_name", default=None, type=str, help="MMICL, or InstructBLIP, etc. The base model used to generate condition context.")
    parser.add_argument("--model_size", default="xl", type=str, help="xxl, xl")
    parser.add_argument("--model_path", default="BleachNick/MMICL-Instructblip-T5-xl", type=str)
    parser.add_argument("--processor_path", default="Salesforce/instructblip-flan-t5-xl", type=str)
    parser.add_argument("--with_condition", action="store_true", help="if loading model with condition (Brote ckpts)")
    parser.add_argument("--generate_caption", action="store_true", help='Whether to generate captions(text)')
    parser.add_argument("--batch_size", default=30, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--dbg", action="store_true")
    parser.add_argument("--condition_from", default=-1, type=int, help="the layer to get condition from, default -1, the last encoder layer")
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()

    print('========== loading data ========== ')
    cond_gen = ConditionGenerator(args)    

    print('args.generate_caption', args.generate_caption)
    print(len(cond_gen.dataset), 'data loaded')

    print('========== generating ========== ')
    rst, rst_mid = get_conditions(cond_gen, args)

    print('process batch rst')
    df = pd.DataFrame(cond_gen.dataset)
    if not args.generate_caption:
        df['condition'] = rst
    else:
        df['condition_cap'] = rst

    print('========== saving ========== ')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    df.to_parquet(os.path.join(args.output_dir, args.output_name), compression='gzip')


