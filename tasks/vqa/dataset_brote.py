import base64
from io import BytesIO
import random
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import torch
from PIL import Image
import os
import torch.nn as nn
import numpy as np
import logging
from datasets import load_dataset, load_metric, Dataset, concatenate_datasets
import datasets
from os.path import join
from torch.nn.functional import pad
from typing import Any, Callable, Dict, List, NewType
from torchmetrics import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from tqdm import tqdm

from pycocoevalcap.cider.cider_scorer import CiderScorer
InputDataClass = NewType("InputDataClass", Any)
from collections.abc import Mapping

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)
IGNORE_INDEX=-100
MASK_INDEX =1
datasets.config.IN_MEMORY_MAX_SIZE = 300 *1024 *1024 *1024

class BroteDataset():
    def __init__(self, processor, model_args, data_args, training_args, config):

        self.rng = random.Random(training_args.seed)

        self.processor = processor
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.config = config

        if self.data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            self.padding = "longest"

        if self.data_args.max_seq_length > processor.tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({processor.tokenizer.model_max_length}). Using max_seq_length={processor.tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, processor.tokenizer.model_max_length)

        if self.data_args.load_datatype == 'pretrain': # for pretraining
            df = self.load_stage1_traindata(sample=False)
            self.train_dataset = datasets.Dataset.from_pandas(df[:-data_args.max_eval_samples])
            self.eval_dataset = datasets.Dataset.from_pandas(df[-data_args.max_eval_samples:])
            self.predict_dataset = self.eval_dataset
            self.data_collator = self.collector_without_preprocess

        elif self.data_args.load_datatype.startswith('mic_full'): 
            df = self.load_stage2_traindata(self.data_args.train_file)
            self.train_dataset = datasets.Dataset.from_pandas(df[:-data_args.max_eval_samples])
            self.eval_dataset = datasets.Dataset.from_pandas(df[-data_args.max_eval_samples:])
            self.predict_dataset = self.eval_dataset
            self.data_collator = self.collector_without_preprocess

        else:
            raise NotImplementedError

        self.metric = load_metric('accuracy')
        # if the above does not work well, please specify the path and use the following: 
        # self.metric = load_metric(path='path to cache/accuracy')
        self.test_key = "accuracy"

        self.train_dataset = self.train_dataset.shuffle(training_args.seed)

        datasets.disable_caching()

        # Split train and dev datasets
        self.train_dataset = self.train_dataset.with_format("torch")
        self.eval_dataset = self.eval_dataset.with_format("torch")
        self.predict_dataset = self.predict_dataset.with_format("torch")

        if training_args.do_train:
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))
        if training_args.do_eval:
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))
        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))
        self.special_visual_token_id = self.processor.tokenizer.convert_tokens_to_ids("图") if self.model_args.backbone_model == 'flan-t5' else self.processor.tokenizer.convert_tokens_to_ids("<visual_embedding>")

    def load_stage1_traindata(self, sample=False):
        dataset_file_names_list = ['vqa', 'stvqa', 'vcr', 'nlvr2', 'ivqa', 'vsr', 'iconqa', 'CGD']

        dataset_len = {}
        dfs = None
        for f_name in os.path.listdir(self.data_args.train_file):
            if not self.data_args.train_file.endswith('condition.parquet.gzip'):
                continue
            path = os.path.join(self.data_args.train_file, f_name)
            parquet_file = pq.ParquetFile(path)
            df = pd.concat([item.to_pandas() for item in parquet_file.iter_batches(batch_size=600)]).reset_index(drop=True)

            ori_dataset = df['ori_dataset'][0] 
            dataset_len[ori_dataset] = len(df)
            if dfs is None:
                dfs = df
            else:
                dfs = pd.concat([dfs, df]).reset_index(drop=True)
        df = self.parse_stage1_captions(dfs, dataset_len=dataset_len)
        if sample:
            df = df.sample(frac=1, random_state=self.training_args.seed)
        return df
    

    def load_stage2_traindata(self, train_file, sample=False):
        raise NotImplementedError

    def parse_stage1_captions(self, df, dataset_len=None):
        #caption_flag = 'promptcap' if 'promptcap' in df.keys() else 'gpt_caption'
        caption_flag = 'gpt_caption'
        conditions = []

        data_dict = []
        if dataset_len is not None:
            dataset_blip_cnt = {name:int(num*0.15) for name, num in dataset_len.items()}
            dataset_blip_done = {name:0 for name, num in dataset_len.items()}
        for i in range(len(df)):
            num_img = len(df.iloc[i]['input_image'])
            num_cap = len(df.iloc[i][caption_flag])
            num = min(num_img, num_cap)
            if dataset_len is not None:
                ori_dataset = df.iloc[i]['ori_dataset']
            else:
                ori_dataset = ""
            for j in range(num):
                if df.iloc[i][caption_flag][j] != "" and df.iloc[i][caption_flag][j] is not None:
                    output_text = df.iloc[i][caption_flag][j]
                    _cond = conditions.iloc[i] if caption_flag == 'promptcap' else df.iloc[i]['condition']
                    _from = 'gpt'
                    if self.data_args.mix_blip2:
                        if dataset_blip_done[ori_dataset]<dataset_blip_cnt[ori_dataset]:
                            output_text = df.iloc[i]['blip2_caption'][j]
                            try:
                                _cond = np.zeros(_cond.shape)
                            except:
                                if self.data_args.load_cond_size == 'xxl':
                                    _cond = np.zeros((4096,))
                                else:
                                    _cond = np.zeros((2048,))
                            _from = 'blip2'

                    #if self.model_args.send_condition_to_llm:
                    #    input_text = "Given <cond>. Image 0 is <image0>图. Please describe this image:"
                    #else:
                    #    input_text = "Image 0 is <image0>图. Please describe this image:"
                    img = df.iloc[i]['input_image'][j]
                    input_text = "Image 0 is <image0>图. Please describe this image:"
                    _dict = {"input_text": input_text,
                            "output_text": output_text,
                            "input_image": img, 
                            "ori_dataset": df['ori_dataset'].iloc[i],
                            "condition": _cond,
                            "from": _from}
                    #if self.model_args.condition_from == 'both':
                    #    _dict["condition_12layer"] = df.iloc[i]['condition_12layer']
                    data_dict.append(_dict)
            dataset_blip_done[ori_dataset] += 1
        del df
        return pd.DataFrame(data_dict)

    def prepare_stage2_drop(self, input_text, input_image, ori_dataset):
        if self.data_args.drop_input == 'image':
            if len(input_image) > 1:
                input_text, input_image = self.drop_img(input_text, input_image)
        elif self.data_args.drop_input == 'image_blank':
            if len(input_image) > 1:
                input_image = self.drop_img_blank(input_image)
        elif self.data_args.drop_input == 'image_mix':
            if len(input_image) > 1:
                if random.randint(1, 2) == 1:
                    input_text, input_image = self.drop_img(input_text, input_image)
                else:
                    input_image = self.drop_img_blank(input_image)
        elif self.data_args.drop_input == 'text':
            shots = input_text.split('\n\n')
            len_shot = len(shots)
            if len_shot > 1:
                input_text = self.drop_txt(shots, len_shot)
        elif self.data_args.drop_input == 'both':
            shots = input_text.split('\n\n')
            len_shot = len(shots)
            if len_shot > 1 and ori_dataset != 'nlvr2':
                opt = random.randint(1, 3) # 1: drop img; 2: drop img_blank; 3: drop txt
                if opt == 1:
                    input_text, input_image = self.drop_img(input_text, input_image)
                elif opt == 2:
                    input_image = self.drop_img_blank(input_image)
                else:
                    input_text = self.drop_txt(shots, len_shot)
            else:
                if len(input_image) > 1:
                    if random.randint(1, 2) == 1:
                        input_text, input_image = self.drop_img(input_text, input_image)
                    else:
                        input_image = self.drop_img_blank(input_image)

        return input_text, input_image 

    def _prepare_inputs(self, examples):
        if self.model_args.send_condition_to_llm:
            result= self.processor.tokenizer(examples["input_text"].replace("图", "".join(33*["图"])),
                padding=self.padding, max_length=self.max_seq_length, truncation=True)
        else:
            result= self.processor.tokenizer(examples["input_text"].replace("图", "".join(32*["图"])),
                padding=self.padding, max_length=self.max_seq_length, truncation=True)

        result['label'] = self.processor.tokenizer(examples["output_text"], padding=self.padding, max_length=self.training_args.max_label_length, truncation=True)["input_ids"]
        result["pixel_values"] = []

        flag = isinstance(examples["input_image"],list)
        if flag:
            for img in examples["input_image"]:
                if os.path.isfile(img):
                    img = Image.open(img)
                else: # not img file, load from base64 str
                    img = self.load_base64_image(img)
                result["pixel_values"].append(self.processor(images = img)["pixel_values"][0])
        else:
            img = examples["input_image"]
            if os.path.isfile(img):
                img = Image.open(img)
            else: # not img file, load from base64 str
                img = self.load_base64_image(img)
            result["pixel_values"].append(self.processor(images = img)["pixel_values"][0])

        result['input_text'] = examples["input_text"]
        if not self.data_args.preprared_cond:
            if self.training_args.dual_none_condition:
                result['global_conditions_zero'] = torch.zeros(1,2048)
            return result

        if self.model_args.condition_from is not None:
            raise NotImplementedError
        else:
            condition = examples["condition"]

        if self.training_args.random_condition:
            result['global_conditions'] = torch.rand(1,condition.shape[0]).renorm(2,0,0.9)[0]
        elif self.training_args.none_condition:
            result['global_conditions'] = torch.zeros(condition.shape)
        else:
            result['global_conditions'] = condition
        return result    

    def drop_img(self, input_text, input_image):
        input_text = input_text.replace("图", "", len(input_image)-1) 
        input_image = [input_image[-1]]
        return input_text, input_image

    def drop_img_blank(self, input_image):
        for i in range(len(input_image)-1):
            input_image[i] = os.path.join(self.training_args.train_file, "img_placeholder.png")
        return input_image

    def drop_txt(self, shots, len_shot):
        if len_shot == 2:
            drop_idx = 0
        else:
            drop_idx = random.randint(1,len_shot-1)-1
        shots[drop_idx] = shots[drop_idx].split('图')[0]+'图'
        return '\n\n'.join(shots)

    def _prepare_inputs_drop(self, examples):
        flag = isinstance(examples["input_image"],list)

        if examples.get('stage', '') != 'stage1': # only drop input for stage2 data
            if self.data_args.drop_input == 'image':
                if flag and len(examples["input_image"]) > 1:
                    examples["input_text"], examples["input_image"] = self.drop_img(examples["input_text"], examples["input_image"])
            elif self.data_args.drop_input == 'image_blank':
                if flag and len(examples["input_image"]) > 1:
                    examples["input_image"] = self.drop_img_blank(examples["input_image"])
            elif self.data_args.drop_input == 'image_mix':
                if flag and len(examples["input_image"]) > 1:
                    if random.randint(1, 2) == 1:
                        examples["input_text"], examples["input_image"] = self.drop_img(examples["input_text"], examples["input_image"])
                    else:
                        examples["input_image"] = self.drop_img_blank(examples["input_image"])
            elif self.data_args.drop_input == 'text':
                shots = examples["input_text"].split('\n\n')
                len_shot = len(shots)
                if len_shot > 1:
                    examples["input_text"] = self.drop_txt(shots, len_shot)
            elif self.data_args.drop_input == 'both':
                shots = examples["input_text"].split('\n\n')
                len_shot = len(shots)
                if len_shot > 1 and examples['ori_dataset'] != 'nlvr2':
                    opt = random.randint(1, 3) # 1: drop img; 2: drop img_blank; 3: drop txt
                    #opt = random.randint(1, 2) # 1: drop img; 2: drop txt
                    if opt == 1:
                        examples["input_text"], examples["input_image"] = self.drop_img(examples["input_text"], examples["input_image"]) # drop img
                    elif opt == 2:
                        examples["input_image"] = self.drop_img_blank(examples["input_image"]) # drop img_blank
                    else:
                        examples["input_text"] = self.drop_txt(shots, len_shot)
                else:
                    if flag and len(examples["input_image"]) > 1:
                        if random.randint(1, 2) == 1:
                            examples["input_text"], examples["input_image"] = self.drop_img(examples["input_text"], examples["input_image"])
                        else:
                            examples["input_image"] = self.drop_img_blank(examples["input_image"]) # drop img_blank
    
            if not len(examples["input_image"]) == len(examples["input_text"].split('图'))-1:
                import pdb; pdb.set_trace()
                print('len_shot == 1, drop img_blank')
                print(examples['file'], len(examples["input_image"]), len(examples["input_text"].split('图'))-1)
                
        result= self.processor.tokenizer(examples["input_text"].replace("图", "".join(32*["图"])),
                padding=self.padding, max_length=self.max_seq_length, truncation=True)

        result['label'] = self.processor.tokenizer(examples["output_text"], padding=self.padding, max_length=self.training_args.max_label_length, truncation=True)["input_ids"]
        result["pixel_values"] = []

        if flag:
            for img in examples["input_image"]:
                if os.path.isfile(img):
                    img = Image.open(img)
                else: # is not file, load from base64 str
                    img = self.load_base64_image(img)
                result["pixel_values"].append(self.processor(images = img)["pixel_values"][0])
        else:
            img = examples["input_image"]
            if os.path.isfile(img):
                img = Image.open(img)
            else: # is not file, load from base64 str
                img = self.load_base64_image(img)
            result["pixel_values"].append(self.processor(images = img)["pixel_values"][0])

        result['image_counts_per_instance'] = len(examples["input_image"])

        result['input_text'] = examples["input_text"]
        if not self.data_args.preprared_cond:
            return result

        if self.model_args.condition_from is not None:
            raise NotImplementedError
        else:
            condition = examples["condition"]

        if self.training_args.random_condition:
            result['global_conditions'] = torch.rand(1,condition.shape[0]).renorm(2,0,0.9)[0]
        elif self.training_args.none_condition:
            result['global_conditions'] = torch.zeros(condition.shape)
        else:
            result['global_conditions'] = condition

        return result    

    def load_base64_image(self,base64_str):
        image = base64.b64decode(base64_str)
        image = BytesIO(image)
        image = Image.open(image)

        return image
 
    def padd_images(self, image, max_length):
        image = torch.tensor(image)
        mask = torch.zeros(max_length).bool()
        pad_len = max_length - image.shape[0]
        mask[:image.shape[0]] = True
        image = pad(image,(0,0,0,0,0,0,0,pad_len)) # padding behind the first dim
        return image,mask
    
    def pad_features(self, features,feature_name,dtype=torch.long,pad_token_id=32000, padding= 'pad_2_max_length'):

        # Step 1: Create a list of label tensors
        if isinstance(features[0][feature_name],torch.Tensor):
            padded_labels = [f[feature_name][0] for f in features]
        elif isinstance(features[0][feature_name],np.ndarray):
            padded_labels = [torch.tensor(f[feature_name][0]) for f in features]
        else:
            padded_labels = [torch.tensor(np.array(f[feature_name][0])) for f in features]
        # Step 2: Get the max length of the label tensors
        max_length = max(len(f[feature_name][0]) for idx,f in enumerate(features)) if padding == 'pad_2_max_length' else self.max_seq_length
        if max_length < self.max_seq_length:
            max_length = self.max_seq_length
        # Step 3: Pad the label tensors
        padded_labels = [pad(label, (0, max_length - len(label)), value=pad_token_id) for label in padded_labels]
        padded_labels = torch.stack(padded_labels).to(dtype)
        return padded_labels

    def replicate_values(self,tensor, indices, num_replications):
        new_tensor = tensor.copy()
        for i,index in enumerate(indices):
            value_to_replicate = tensor[index]
            idx = index+i*num_replications
            replicated_values = np.repeat(value_to_replicate, num_replications)
            new_tensor = np.insert(new_tensor, idx+1, replicated_values)
        return new_tensor


    def padding_input_ids(self,feature,sp_token,key='input_ids',num_replications=31,dtype = torch.long):
        pad_input_ids=[]
        length =[]
        diff_length =[]
        for each in feature:
            o_tensor = each[key][0]
            if not isinstance(o_tensor, np.ndarray):
                o_tensor = np.array(o_tensor)
            target_indices = np.where(o_tensor == sp_token)[0]
            new_tensor = self.replicate_values(o_tensor, target_indices, num_replications)
            length.append(len(new_tensor))
            diff_length.append(len(new_tensor)-len(o_tensor))
            pad_input_ids.append(torch.tensor(new_tensor))
        max_length = max(length)
        pad_ids = torch.stack([pad(ids, (0, max_length - length[idx]), value=self.processor.tokenizer.pad_token_id) for idx,ids in enumerate(pad_input_ids)])
        return pad_ids,diff_length

    def collector_without_preprocess(self, features: List[InputDataClass]):

        if self.data_args.drop_input is not None:
            features = [self._prepare_inputs_drop(f) for f in features]
        else:
            features = [self._prepare_inputs(f) for f in features]

        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}

        if "label" in first and first["label"] is not None: # for training & eval batch
            label = first["label"][0].item() if isinstance(first["label"], torch.Tensor) else first["label"][0]
            dtype = torch.long  if isinstance(label, int) else torch.float
            batch["labels"] = torch.Tensor([f["label"] for f in features]).to(dtype)
        elif "label_ids" in first and first["label_ids"] is not None: # for training & eval batch
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)
        batch["labels"][torch.where(batch["labels"]==self.processor.tokenizer.pad_token_id)]= IGNORE_INDEX 

        ignored_keys = ['input_text', 'image', 'output_text', 'output_image','pixel_values', 'gpt_caption', 'images', 'model', 'from']
        for k, v in first.items():
            if k.startswith('_'): continue
            if k == 'condition': continue
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
                    image,img_mask = self.padd_images(f[k],max_image_length)
                    image_list.append(image)
                    mask_list.append(img_mask)
                batch[k] = torch.stack(image_list)
                batch['img_mask'] = torch.stack(mask_list)

        batch['sp_token'] = self.special_visual_token_id
        batch["input_text"] = [f["input_text"] for f in features]# added for icl model

        if self.training_args.full_bf16_training:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and k in ['pixel_values', 'global_conditions', 'global_conditions_zero']:
                    batch[k] = v.to(dtype=torch.bfloat16)
        return batch

    def save_pred_label(self):
        pass

    def compute_metrics(self, p: EvalPrediction):

        preds = p.predictions
        labels = p.label_ids
        preds[preds==IGNORE_INDEX] = 0

        labels[labels==IGNORE_INDEX] = 0
        bleu = BLEUScore()
        rouge = ROUGEScore()
        cider_scorer = CiderScorer(n=4, sigma=6)
        bleu_scorer = BleuScorer(n=4)

        bleu_result = []
        accuracy = 0
        dict_return={}
        
        p_token_batch = self.processor.tokenizer.batch_decode(preds,skip_special_tokens=True)
        label_token_batch = self.processor.tokenizer.batch_decode(labels,skip_special_tokens=True)
        try:
            rouge_mertic = rouge(p_token_batch , label_token_batch )
            dict_return.update(rouge_mertic)
        except:
            if self.training_args.local_rank == 0:
                print("error when compute rouge")
        for i,p_token in enumerate(p_token_batch):

            bleu_result.append(bleu([p_token],[[label_token_batch[i]]]).item()*100)
            cider_scorer+= (p_token,[label_token_batch[i]])
            bleu_scorer+= (p_token,[label_token_batch[i]])
            l = label_token_batch[i].split(' ')[0]
            if p_token == label_token_batch[i] or p_token in label_token_batch[i] or  label_token_batch[i] in p_token:
                accuracy+=1
        cider,_ = cider_scorer.compute_score()
        bleu_score, _ = bleu_scorer.compute_score(option='closest', verbose=1)

        dict_return['bleu_1'] = bleu_score[0]
        dict_return['bleu_2'] = bleu_score[1]
        dict_return['bleu_3'] = bleu_score[2]
        dict_return['bleu_4'] = bleu_score[3]
        dict_return['cider'] = cider
        dict_return['accuracy'] = accuracy/len(preds)
        dict_return['avg_bleuScore'] = np.array(bleu_result).mean()


        return dict_return
