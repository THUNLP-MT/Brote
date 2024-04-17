import argparse
import json
import base64
import requests
import random
import time
from transformers import BlipProcessor, Blip2ForConditionalGeneration
from PIL import Image
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
import os

from api_key import *
from prompt import *

BASE_IMAGE_PATH = "../images"
BASE_TEXT_PATH = "../text"

MAX_TOKEN = 1000

DESTINATION_TEXT_PATH = "The path to save result"


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def randomly_select_lines(filename, num_lines):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Randomly select 'num_lines' lines
    random.seed(0)
    selected_lines = random.sample(lines, min(num_lines, len(lines)))

    # Convert lines from JSON string to Python dictionary
    selected_json = [json.loads(line) for line in selected_lines]

    return selected_json


# base_image_path: "../images/coco"
# image_path: "./data/coco/train2014/COCO_train2014_000000111636.jpg"
def modify_image_path(image_path, base_image_path):
    # Split the string by '/'
    parts = image_path.split('/')

    # Replace the parts before the third '/'
    if "vcr" in image_path:
        parts[:2] = base_image_path.split('/')
    else:
        parts[:3] = base_image_path.split('/')

    # Join the parts back into a single string
    new_string = '/'.join(parts)

    return new_string


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



def generate_caption(url, username, password, model, input_content):

    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": input_content
            }
        ],
        "max_tokens": MAX_TOKEN,
    }

    data = json.dumps(data)

    response = requests.post(url=url, data=data, auth=HTTPBasicAuth(username=username,password=password))
    return response


def process(text_path, base_image_path, prompt, num_example, use_GPT4V, use_GPT4, use_GPT3, starts_from=0, is_video=False):
    random.seed(0)
    selected_data = randomly_select_lines(text_path, num_example)

    for cnt, json_obj in tqdm(enumerate(selected_data), total=num_example, leave=True):

        if cnt < starts_from:
            continue

        # Print the JSON object
        input_image_path = json_obj['input_image']

        images = []
        if isinstance(input_image_path, list):
            for image_path in input_image_path:
                images.append(modify_image_path(image_path, base_image_path))
        else:
            images.append(modify_image_path(input_image_path, base_image_path))
    
        status = True

        if not is_video:
            if use_GPT4 or use_GPT3:
                captions = []
                status = True
                for image in input_image_path:
                    try:
                        image = modify_image_path(image, base_image_path)
                        images.append(image)
                        caption = image.replace(".png", ".out")
                        caption = caption.replace(".jpg", ".out")
                        caption = caption.replace(".JPEG", ".out")
                        caption = caption.replace("../images", "../captions")
                        with open(caption, "r") as file:
                            content = file.read()
                            captions.append(content)
                    except:
                        status = False
                if not status:
                    continue
        
        else:
            image_path = modify_image_path(input_image_path, base_image_path)
            for j in range(8):
                this_image = image_path.replace(".webm", f"_image{j}.jpg")
                if not os.path.exists(this_image):
                    status = False
                    break
                images.append(this_image)
                try:
                    caption = image.replace(".png", ".out")
                    caption = caption.replace(".jpg", ".out")
                    caption = caption.replace(".JPEG", ".out")
                    caption = caption.replace("../images", "../captions")
                    with open(caption, "r") as file:
                        content =  file.read()
                        captions.append(content)
                    image = Image.open(this_image)
                    images.append(this_image)
                except:
                    status = False
                    break     
            if not status:
                continue

        if use_GPT4V:
            model = "GPT4V"
            input_content = []
            input_text = prompt["GPT4V"] + " " + json_obj['input_text'] + " " + json_obj['output_text']

            input_content.append({"type": "text", "text": input_text})
            for image in images:
                base64_image = encode_image(image)
                input_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

            try:
                response = generate_caption(GPT4V_URL, GPT4V_USERNAME, GPT4V_PASSWORD, GPT4V_MODEL, input_content)
                response = response.json()
            except:
                if type(response) == str:
                    print(response)
                    continue
                elif "Bad Request (400)" in response.text:
                    print('Bad Request (400)')
                    print('****input_text****', input_text)
                    continue
                else:
                    time.sleep(5)
                    response = generate_caption(GPT4V_URL, GPT4V_USERNAME, GPT4V_PASSWORD, GPT4V_MODEL, input_content)
                    response = response.json()
            if "error" in response: 
                if response["error"]["message"] == 'Your input image may contain content that is not allowed by our safety system.':
                    print('not allowed content')
                elif response["error"]["message"].startswith("You uploaded an unsupported image."):
                    print('unsupported image')
                else:
                    print(response["error"]["message"])
                print('****input_text****', input_text)
                continue
            response = response["choices"][0]["message"]["content"]

        if use_GPT4:
            model = "GPT4"
            input_content = prompt["GPT4"]
            
            i = 0
            for caption in captions:
                if i != 0:
                    input_content = input_content + ';\n'
                input_content = input_content + f"<image{i}>: " + caption
                i += 1
            temp_string = json_obj['input_text']
            temp_string = temp_string.replace("\u56fe", "")
            input_content = input_content + '.\n========\n' + "Here are the additional information that you should focus on!\n\n" + temp_string + ' ' + json_obj['output_text']
            try:
                response = generate_caption(GPT4_URL, GPT4_USERNAME, GPT4_PASSWORD, GPT4_MODEL, input_content)
                response = response.json()
            except:
                time.sleep(2)
                response = generate_caption(GPT4_URL, GPT4_USERNAME, GPT4_PASSWORD, GPT4_MODEL, input_content)
                response = response.json()
            response = response["choices"][0]["message"]["content"]

        if use_GPT3:
            model = "GPT3"
            input_content = prompt["GPT3"]

            i = 0
            for caption in captions:
                if i != 0:
                    input_content = input_content + ';\n'
                input_content = input_content + f"<image{i}>: " + caption
                i += 1
            temp_string = json_obj['input_text']
            temp_string = temp_string.replace("\u56fe", "")
            input_content = input_content + '.\n========\n' + "Here are the additional information that you should focus on!\n\n" + temp_string + ' ' + json_obj['output_text']

            response = generate_caption(GPT3_URL, GPT3_USERNAME, GPT3_PASSWORD, GPT3_MODEL, input_content)
            response = json.loads(response.text)["choices"][0]["message"]["content"]

        output_json = {
            "input_text": json_obj['input_text'],
            "output_text": json_obj['output_text'],
            "images": images,
            "input_image": json_obj["input_image"],
            "response": response,
            "model": model,
        }
        with open(DESTINATION_TEXT_PATH, "a") as file:
            json_str = json.dumps(output_json)
            file.write(json_str + '\n')


def process_mic(text_path, base_image_path, prompt, num_example, use_GPT4V, use_GPT4, use_GPT3, starts_from=0):
    random.seed(0)
    with open(text_path, "r") as file:
        data = json.load(file)
    data = data["data"]
    selected_keys = random.sample(list(data.keys()), num_example)
    selected_data = [data[key] for key in selected_keys]

    for cnt, json_obj in tqdm(enumerate(selected_data), total=num_example, leave=True):
        if cnt < starts_from:
            continue

        input_image_path = json_obj['image_ids']
        
        captions = []
        images = []

        status = True
        for image in input_image_path:
            images.append(image)
            try:
                caption_path = f"../captions/CGD/{image}.out"
                with open(caption_path, "r") as file:
                    content = file.read()
                    captions.append(content)
            except:
                status = False
        if not status:
            continue

        if use_GPT4V:
            model = "GPT4V"
            input_content = []
            input_text = prompt["GPT4V"] + " " + json_obj['input_text'] + " " + json_obj['output_text']

            input_content.append({"type": "text", "text": input_text})
            for image in images:
                base64_image = encode_image(image)
                input_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

            try:
                response = generate_caption(GPT4V_URL, GPT4V_USERNAME, GPT4V_PASSWORD, GPT4V_MODEL, input_content)
                response = response.json()
            except:
                if type(response) == str:
                    print(response)
                    continue
                elif "Bad Request (400)" in response.text:
                    print('Bad Request (400)')
                    print('****input_text****', input_text)
                    continue
                else:
                    time.sleep(5)
                    response = generate_caption(GPT4V_URL, GPT4V_USERNAME, GPT4V_PASSWORD, GPT4V_MODEL, input_content)
                    response = response.json()
            if "error" in response: 
                if response["error"]["message"] == 'Your input image may contain content that is not allowed by our safety system.':
                    print('not allowed content')
                elif response["error"]["message"].startswith("You uploaded an unsupported image."):
                    print('unsupported image')
                else:
                    print(response["error"]["message"])
                print('****input_text****', input_text)
                continue
            response = response["choices"][0]["message"]["content"]

        if use_GPT4:
            model = "GPT4"
            input_content = prompt["GPT4"]
            
            i = 0
            for caption in captions:
                if i != 0:
                    input_content = input_content + ';\n'
                input_content = input_content + f"<image{i}>: " + caption
                i += 1
            temp_string = json_obj['input_text']
            temp_string = temp_string.replace("\u56fe", "")
            input_content = input_content + '.\n========\n' + "Here are the additional information that you should focus on!\n\n" + temp_string + ' ' + json_obj['output_text']
            try:
                response = generate_caption(GPT4_URL, GPT4_USERNAME, GPT4_PASSWORD, GPT4_MODEL, input_content)
                response = response.json()
            except:
                time.sleep(2)
                response = generate_caption(GPT4_URL, GPT4_USERNAME, GPT4_PASSWORD, GPT4_MODEL, input_content)
                response = response.json()
            response = response["choices"][0]["message"]["content"]

        if use_GPT3:
            model = "GPT3"
            input_content = prompt["GPT3"]

            i = 0
            for caption in captions:
                if i != 0:
                    input_content = input_content + ';\n'
                input_content = input_content + f"<image{i}>: " + caption
                i += 1
            temp_string = json_obj['input_text']
            temp_string = temp_string.replace("\u56fe", "")
            input_content = input_content + '.\n========\n' + "Here are the additional information that you should focus on!\n\n" + temp_string + ' ' + json_obj['output_text']

            response = generate_caption(GPT3_URL, GPT3_USERNAME, GPT3_PASSWORD, GPT3_MODEL, input_content)
            response = json.loads(response.text)["choices"][0]["message"]["content"]
        
        output_json = {
            "input_text": json_obj['instruction'],
            "output_text": json_obj['answer'],
            "images": images,
            "response": response,
            "model": model,
        }
        with open(DESTINATION_TEXT_PATH, "a") as file:
            json_str = json.dumps(output_json)
            file.write(json_str + '\n')
        


def train(args):

    print("Training started with the following settings:")
    print(f"COCO dataset: {args.coco}")
    print(f"Flickr dataset: {args.flickr}")
    print(f"OK-VQA dataset: {args.okvqa}")
    print(f"VQA dataset: {args.vqa}")
    print(f"nlvr2 dataset: {args.nlvr2}")
    print(f"llava dataset: {args.llava}")
    print(f"stvqa dataset: {args.stvqa}")
    print(f"vcr dataset: {args.vcr}")
    print(f"ivqa dataset: {args.ivqa}")
    print(f"ivqa_frame dataset: {args.ivqa_frame}")
    print(f"vsr dataset: {args.vsr}")
    print(f"iconqa dataset: {args.iconqa}")
    print(f"CGD dataset: {args.CGD}")
    print(f"LA dataset: {args.LA}")
    print(f"use GPT-4V: {args.GPT4V}")
    print(f"use GPT-4: {args.GPT4}")
    print(f"use GPT-3: {args.GPT3}")
    print(f"number of example: {args.num_example}\n")
    print(f"starts from {args.starts_from}")

    if args.coco:
        text_path = BASE_TEXT_PATH + "/coco.jsonl" 
        image_path = BASE_IMAGE_PATH + "/coco"
        process(text_path, image_path, prompt_coco, args.num_example, args.GPT4V, args.GPT4, args.GPT3, starts_from=args.starts_from)
    if args.flickr:
        text_path = BASE_TEXT_PATH + "/flickr.jsonl" 
        image_path = BASE_IMAGE_PATH + "/flickr"
        process(text_path, image_path, prompt_flickr, args.num_example, args.GPT4V, args.GPT4, args.GPT3, starts_from=args.starts_from)
    if args.okvqa:
        text_path = BASE_TEXT_PATH + "/okvqa.jsonl" 
        image_path = BASE_IMAGE_PATH + "/coco"
        process(text_path, image_path, prompt_okvqa, args.num_example, args.GPT4V, args.GPT4, args.GPT3, starts_from=args.starts_from)
    if args.vqa:
        text_path = BASE_TEXT_PATH + "/vqa.jsonl"
        image_path = BASE_IMAGE_PATH + "/coco" 
        process(text_path, image_path, prompt_vqa, args.num_example, args.GPT4V, args.GPT4, args.GPT3, starts_from=args.starts_from)
    if args.nlvr2:
        text_path = BASE_TEXT_PATH + "/nlvr2_one_image.jsonl"
        image_path = BASE_IMAGE_PATH + "/nlvr2" 
        process(text_path, image_path, prompt_nlvr2, args.num_example, args.GPT4V, args.GPT4, args.GPT3, starts_from=args.starts_from)
    if args.llava:
        text_path = BASE_TEXT_PATH + "/llava.jsonl"
        image_path = BASE_IMAGE_PATH + "/coco" 
        process(text_path, image_path, prompt_llava, args.num_example, args.GPT4V, args.GPT4, args.GPT3, starts_from=args.starts_from)
    if args.stvqa:
        text_path = BASE_TEXT_PATH + "/stvqa.jsonl"
        image_path = BASE_IMAGE_PATH + "/stvqa" 
        process(text_path, image_path, prompt_stvqa, args.num_example, args.GPT4V, args.GPT4, args.GPT3, starts_from=args.starts_from)
    if args.vcr:
        text_path = BASE_TEXT_PATH + "/vcr.jsonl"
        image_path = BASE_IMAGE_PATH + "/vcr" 
        process(text_path, image_path, prompt_vcr, args.num_example, args.GPT4V, args.GPT4, args.GPT3, starts_from=args.starts_from)
    if args.ivqa:
        text_path = BASE_TEXT_PATH + "/ivqa_frame.jsonl"
        image_path = BASE_IMAGE_PATH + "/ivqa" 
        process(text_path, image_path, prompt_ivqa, args.num_example, args.GPT4V, args.GPT4, args.GPT3, starts_from=args.starts_from)
    if args.ivqa_frame:
        text_path = BASE_TEXT_PATH + "/ivqa.jsonl"
        image_path = BASE_IMAGE_PATH + "/ivqa" 
        process(text_path, image_path, prompt_ivqa_frame, args.num_example, args.GPT4V, args.GPT4, args.GPT3, starts_from=args.starts_from, is_video=True)
    if args.vsr:
        text_path = BASE_TEXT_PATH + "/vsr.jsonl"
        image_path = BASE_IMAGE_PATH + "/vsr" 
        process(text_path, image_path, prompt_vsr, args.num_example, args.GPT4V, args.GPT4, args.GPT3, starts_from=args.starts_from)
    if args.iconqa:
        text_path = BASE_TEXT_PATH + "/iconqa_modify.jsonl"
        image_path = BASE_IMAGE_PATH + "/iconqa" 
        process(text_path, image_path, prompt_iconqa, args.num_example, args.GPT4V, args.GPT4, args.GPT3, starts_from=args.starts_from)
    if args.CGD:
        text_path = BASE_TEXT_PATH + "/CGD.json"
        image_path = BASE_IMAGE_PATH + "/CGD" 
        process_mic(text_path, image_path, prompt_CGD, args.num_example, args.GPT4V, args.GPT4, args.GPT3, starts_from=args.starts_from)
    if args.LA:
        text_path = BASE_TEXT_PATH + "/LA/LADD_instructions.json"
        image_path = BASE_IMAGE_PATH + "/LA" 
        process_mic(text_path, image_path, prompt_LA, args.num_example, args.GPT4V, args.GPT4, args.GPT3, starts_from=args.starts_from)

        
def main():
    parser = argparse.ArgumentParser(description="Training script for image-text model.")
    parser.add_argument('--coco', type=str2bool, default=False, help='Use COCO dataset')
    parser.add_argument('--flickr', type=str2bool, default=False, help='Use Flickr dataset')
    parser.add_argument('--okvqa', type=str2bool, default=False, help='Use OK-VQA dataset')
    parser.add_argument('--vqa', type=str2bool, default=False, help='Use VQA dataset')
    parser.add_argument('--nlvr2', type=str2bool, default=False, help='Use NLVR2 dataset')
    parser.add_argument('--llava', type=str2bool, default=False, help='Use llava dataset')
    parser.add_argument('--stvqa', type=str2bool, default=False, help='Use stvqa dataset')
    parser.add_argument('--vcr', type=str2bool, default=False, help='Use vcr dataset')
    parser.add_argument('--ivqa', type=str2bool, default=False, help='Use ivqa dataset')
    parser.add_argument('--ivqa_frame', type=str2bool, default=False, help='Use ivqa_frame dataset')
    parser.add_argument('--vsr', type=str2bool, default=False, help='Use vsr dataset')
    parser.add_argument('--iconqa', type=str2bool, default=False, help='Use iconqa dataset')
    parser.add_argument('--CGD', type=str2bool, default=False, help='Use CGD dataset')
    parser.add_argument('--LA', type=str2bool, default=False, help='Use LA dataset')
    parser.add_argument('--GPT4V', type=str2bool, default=False, help='Use GPT-4V')
    parser.add_argument('--GPT4', type=str2bool, default=False, help='Use GPT-4')
    parser.add_argument('--GPT3', type=str2bool, default=False, help='Use GPT-3')
    parser.add_argument('--num_example', type=int, default=5, help='Number of example')
    parser.add_argument('--starts_from', type=int, default=0, help='line number to starts from')

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
        
