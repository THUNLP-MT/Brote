# 👀 Browse and Concentrate: Comprehending Multimodal Content via prior-LLM Context Fusion

[**🌐 Homepage**](https://stephenzhuyiqi.github.io/Brote/) | [**📖 arXiv**](https://arxiv.org/pdf/2402.12195.pdf) | [**🤗 Models**](https://huggingface.co/wangphoebe/Brote-IM-XXL)

This repo includes codes and examples for paper [Browse and Concentrate: Comprehending Multimodal Content via prior-LLM Context Fusion](https://arxiv.org/pdf/2402.12195.pdf). 

## Framework
We propose a paradigm **Bro**wse and Concentra**te** (**Brote**) for incorporating multimodal context before feeding features into the LLM, together with two approaches to implement our paradigm, Brote-EX and Brote-IM. The model structures are shown in the following figure.

<img src="./figures/model.png" alt="Image" width="600">

## Instructions For Training and Inference

### Data
Please refer to the data format described in [MIC](https://github.com/HaozheZhao/MIC).

### Environment
```
pip install -r requirements.txt
```

### Training
coming soon

### Inference
Please refer to the test.py file; files under the **model** dir are for test only, and will be updated soon for training.

Instructions for the test script are coming soon. 

## Example
<img src="./figures/git_showcase.png" alt="Image" width="600">

(🐱 in this figure is a 6-year-old cat, his name is Alan.)

## Models
coming soon

## Reference

📑 If you find our project helpful to your research, please consider citing:
```
@article{wang2024browse,
  title={Browse and Concentrate: Comprehending Multimodal Content via prior-LLM Context Fusion},
  author={Wang, Ziyue and Chen, Chi and Zhu, Yiqi and Luo, Fuwen and Li, Peng and Yan, Ming and Zhang, Ji and Huang, Fei and Sun, Maosong and Liu, Yang},
  journal={arXiv preprint arXiv:2402.12195},
  year={2024}
}
```
