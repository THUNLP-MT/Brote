import sys, os
import datasets
import json
import transformers

from model.instructblip_im import InstructBlipConfig, InstructBlipForConditionalGeneration, InstructBlipProcessor
from PIL import Image
import torch

def init_model(model_scale):
    if model_scale == 'xl': # xl
        print('coming soon')
        return 
    else: # xxl
        # download the model before testing, and place the model files under the following dir
        model_ckpt = './ckpt/Brote-IM-XXL'
        # or download the model from huggingface when using it
        # model_ckpt = 'wangphoebe/Brote-IM-XXL'

        # we use the processor from instructblip
        processor_ckpt = 'Salesforce/instructblip-flan-t5-xxl'
    
    config = InstructBlipConfig.from_pretrained(model_ckpt)
    config.qformer_config.global_calculation = 'add'
    
    print("loading models")
    processor = InstructBlipProcessor.from_pretrained(processor_ckpt)
    model = InstructBlipForConditionalGeneration.from_pretrained(model_ckpt,
            config=config).to('cuda:0',dtype=torch.bfloat16) 
    model.set_mode("implicit")
    
    image_placeholder="图"
    sp = [image_placeholder]+[f"<image{i}>" for i in range(20)]
    sp = sp+processor.tokenizer.additional_special_tokens[len(sp):]
    processor.tokenizer.add_special_tokens({'additional_special_tokens':sp})
    if model.qformer.embeddings.word_embeddings.weight.shape[0] != len(processor.qformer_tokenizer):
        model.qformer.resize_token_embeddings(len(processor.qformer_tokenizer))
    
    global replace_token
    replace_token="".join(32*[image_placeholder])
    return model, processor

def case1():
    image = Image.open("./images/cal_num1.png")
    image1 = Image.open("./images/cal_num2.png")
    image2 = Image.open("./images/cal_num3.png")
    images = [image,image1,image2]
    
    prompt = [f'Use the image 0: <image0>{replace_token},image 1: <image1>{replace_token} and image 2: <image2>{replace_token} as a visual aid to help you calculate the equation accurately. <image 0> is 2+1=3.\n<image 1> is 5+6=11.\n<image 2> is']

    prompt_raw = ['Use the image 0: <image0>{replace_token},image 1: <image1>{replace_token} and image 2: <image2>{replace_token} as a visual aid to help you calculate the equation accurately. <image 0> is 2+1=3.\n<image 1> is 5+6=11.\n<image 2> is']
    return prompt, prompt_raw, images

def case2():
    image = Image.open("./images/chinchilla.png")
    image1 = Image.open("./images/shiba.png")
    image2 = Image.open("./images/flamingo.png")
    images = [image,image1,image2]
    prompt = [f'image 0 is <image0>{replace_token},image 1 is <image1>{replace_token},image 2 is <image2>{replace_token}. Question: <image0> is a chinchilla. They are mainly found in Chile.\n Question: <image1> is a shiba. They are very popular in Japan.\nQuestion: <image2> is']
    prompt_raw = ['image 0 is <image0>{replace_token},image 1 is <image1>{replace_token},image 2 is <image2>{replace_token}. Question: <image0> is a chinchilla. They are mainly found in Chile.\n Question: <image1> is a shiba. They are very popular in Japan.\nQuestion: <image2> is']
    return prompt, prompt_raw, images

def case3():

    image = Image.open("./images/flamingo_photo.png")
    image1 = Image.open("./images/flamingo_cartoon.png")
    image2 = Image.open("./images/flamingo_3d.png")
    images = [image,image1,image2]
    prompt = [f'Use the image 0: <image0>{replace_token}, image 1: <image1>{replace_token} and image 2: <image2>{replace_token} as a visual aids to help you answer the question. Question: Give the reason why image 0, image 1 and image 2 are different? Answer:']
    prompt_raw = ['Use the image 0: <image0>{replace_token}, image 1: <image1>{replace_token} and image 2: <image2>{replace_token} as a visual aids to help you answer the question. Question: Give the reason why image 0, image 1 and image 2 are different? Answer:']
    return prompt, prompt_raw, images

def case4():
    images = [Image.open("./images/left.png"), Image.open("./images/right.png")]
    
    prompt = [f'image 0 is <image0>{replace_token}, image 0 is <image1>{replace_token}. Given these two images, answer the question: Two dog teams in the image 0 and image 1 are each forward-facing, but headed in different directions. True or false?']

    return prompt, images

def case5():
    image = Image.open ("./images/alan.jpg")
    image1 = Image.open ("./images/dog_case.jpg")
    image2 = Image.open ("./images/capybara.jpg")
    images = [image, image1, image2]
    
    prompt = [f'Use the image 0: <image0>{replace_token},image 1: <image1>{replace_token} and image 2: <image2>{replace_token} as a visual aid to help you answer what breed of animal is shown. <image 0> shows a cat sitting on the floor in indoor environment.\n<image 1> shows a dog sitting on a wooden floor.\n<image 2> shows']

    return prompt, images

def predict(images, prompt):
    print("encoding")
    inputs = processor(images=images, text=prompt, return_tensors="pt", padding=True)
    
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
    inputs['img_mask'] = torch.tensor([[1 for i in range(len(images))]])
    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
    
    print("predicting")
    inputs = inputs.to('cuda:0')
    
    outputs = model.generate(
            pixel_values = inputs['pixel_values'],
            input_ids = inputs['input_ids'],
            attention_mask = inputs['attention_mask'],
            img_mask = inputs['img_mask'],
            #output_attentions=True,
            do_sample=False,
            max_length=50,
            min_length=1,
            set_min_padding_size =False,
            #generate_conditions=True,
            #return_logits=True
    )
    
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)
    print(generated_text)


if __name__ == '__main__':
    case = sys.argv[1]
    model_scale = sys.argv[2]
    if not model_scale == 'xxl':
        print('only Brote-IM-XXL model available now, the others will be released soon.')
        exit(1)
    model, processor = init_model(model_scale)

    case = sys.argv[1]
    if case == "1":
        prompt, prompt_raw, images = case1()
        predict(images, prompt)
    elif case == "2":
        prompt, prompt_raw, images = case2()
        predict(images, prompt)
    elif case == "3":
        prompt, prompt_raw, images = case3()
        predict(images, prompt)
    elif case == "4":
        prompt, images = case4()
        predict(images, prompt)
    elif case == "5":
        predict(images, prompt)
        prompt, images = case5()
    elif case == "all":
        print('case 1')
        prompt, _, images = case1()
        predict(images, prompt)

        print('case 2')
        prompt, _, images = case2()
        predict(images, prompt)

        print('case 3')
        prompt, _, images = case3()
        predict(images, prompt)

        print('case 4')
        prompt, images = case4()
        predict(images, prompt)
    
        print('case 5')
        prompt, images = case5()
        predict(images, prompt)


