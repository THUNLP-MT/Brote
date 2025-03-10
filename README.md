# 👀 Browse and Concentrate: Comprehending Multimodal Content via prior-LLM Context Fusion (ACL '24 Oral)

[**🌐 Homepage**](https://thunlp-mt.github.io/Brote/) | [**📖 arXiv**](https://arxiv.org/pdf/2402.12195.pdf) | [**🤗 Models**](https://huggingface.co/wangphoebe/Brote-IM-XXL)

This repo includes codes and examples for paper [Browse and Concentrate: Comprehending Multimodal Content via prior-LLM Context Fusion](https://arxiv.org/pdf/2402.12195.pdf). 
## Activities

1. [2024-12-20] Pre-training scripts released.
2. [2024-12-17] Update condition context generation scripts.
3. [2024-12-14] Release the training instructions. The full training scripts will be available soon.
4. [2024-12-08] Pretraining data released. (These data are updated on 20 Dec 2024, please use the newst version.)
5. [2024-05-16] This paper is accepted by ACL 2024 (main conference, oral). Information for our training data is updated.
6. [2024-04-18] Code and cases for data generation released. The generated data are used for pretraining.
7. [2024-03-18] Brote-IM-XXL model released, please download from this [link](https://huggingface.co/wangphoebe/Brote-IM-XXL). 
8. [2024-02-26] Project released.

## Framework
We propose a paradigm **Bro**wse and Concentra**te** (**Brote**) for incorporating multimodal context before feeding features into the LLM, together with two approaches to implement our paradigm, Brote-EX and Brote-IM. The model structures are shown in the following figure.

<img src="./figures/model.png" alt="Image" width="500">

## Instructions For Training and Inference

(Please jump to the [inference section](#Inference) if you want to use our model for inference only.)

### 1. Data
Please refer to the data format described in [MIC](https://github.com/HaozheZhao/MIC).

#### 1.1 Data for pretraining.

We create a dataset of 56k fewshot data samples (each data sample contains one or multiple images), resulting in 191k training instances (one image per instance). These instances are supposed to contain question-aware and cross-image information. The data construction pipeline is illustrated in the following figure.

<img src="./figures/pretrain_data.png" alt="Image" width="400">

Please download our pretraining dataset from the ModelScope [link](https://www.modelscope.cn/datasets/wphoebe/Brote-pretrain/files), or HuggingFace [link](https://huggingface.co/datasets/wangphoebe/Brote-pretrain).

#### 1.2 Data for finetuning.

We sampled about 500k data from MIC for model finetuning.

### 2. Environment
```
pip install -r requirements.txt
```

### 3. Training

The full training scripts will be available soon.

⚠️ Please follow the instructions for training:

#### 3.1. Pretraining 
  - Preparing for training data
     - **Download** the pretraining data from ModelScope [link](https://www.modelscope.cn/datasets/wphoebe/Brote-pretrain/files), or HuggingFace [link](https://huggingface.co/datasets/wangphoebe/Brote-pretrain).
     - **Generate and save** condition contexts using the original InstructBlip or MMICL models.
       - We used encoder_last_hidden_state\[eos_token_index\] in our paper. You can also explore representations from othe layers or positions.
       - The input data of this generate process comes from the 'input_text' and 'input_image' fields in the pretraining dataset.
       - Please modify the required fields in run_script/gen_condition/get_conditions_gpt.sh. Here is an example of using the script:
         ```
         bash run_script/gen_condition/get_conditions_gpt.sh 0 stage1_gpt_v0.parquet.gzip ./pretrain_data stage1_gpt_v0_condion.parquet.gzip
         ```
  - Unfreeze the **parameters for query token and Q-Former** (the others remain frozen), and conduct training targeting at the '**gpt_caption**' field in the pretraining dataset.

  - Command to run:
    ```
    bash run_script/pretrain/train_stage1.sh
    ```

#### 3.2. Finetuning 
  - Brote-EX
    - Download the MIC dataset.
    - Generate and save condition contexts using the original InstructBlip or MMICL models. Note that this refers to the condition contexts of MIC dataset following our data dropping strategies (discussed in section 3.4 in our paper), which is different from the pretrainig data.
    - Unfreeze the **parameters for query token, Q-Former, and query & values** of attention layers of the LLM.

  - Brote-IM
    - Download the MIC dataset.
    - No need to generate condition contexts. You can directly fineutne from the pretrained model following the above instruction, or continue fineutning from Brote-EX (this works better).
    - Unfreeze the **parameters for query token, Q-Former, and query & values** of attention layers of the LLM.

### <a id="Inference"> 4. Inference </a>

To run the test script (ensure the required libraries are properly installed):
```
export CUDAID='please set you cuda id here'
export TASKID='please set the case id (from 1 to 5), or use the string 'all'(lowercase)'
CUDA_VISIBLE_DEVICES=$CUDAID python test.py $TASKID 
```
Please note that the input data format matters. If you cannot obtain similar results as mentioned in our paper, please try to modify the instruction template, especially for those aligning image tokens to the image representation. 

## Example
<img src="./figures/git_showcase.png" alt="Image" width="600">

(🐱 in this figure is a 6-year-old cat, his name is Alan.)

## Models
Please download our model from [**🤗 Models**](https://huggingface.co/wangphoebe/Brote-IM-XXL).

## Reference

📑 If you find our project helpful to your research, please consider citing:
```
@inproceedings{
wang2024browse,
title={Browse and Concentrate: Comprehending Multimodal Content via Prior-{LLM} Context Fusion},
author={Wang, Ziyue and Chen, Chi and Zhu, Yiqi and Luo, Fuwen and Li, Peng and Yan, Ming and Zhang, Ji and Huang, Fei and Sun, Maosong and Liu, Yang},
booktitle={The 62nd Annual Meeting of the Association for Computational Linguistics},
year={2024},
}
```
## Acknowledgement
Our models are build upon [MMICL](https://github.com/HaozheZhao/MIC) and [InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip).
