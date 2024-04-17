from transformers import BlipProcessor, Blip2ForConditionalGeneration
from PIL import Image
import os, sys
import torch
from tqdm import tqdm 

QUESTION = "Please describe this image in detail."
BATCH_SIZE = 256

MODEL_PATH = "your blip2 model path"
IMAGE_FOLDER = "your image folder path"
DES_FOLDER = "the folder of the caption" # should be "../captions/{SAME SUFFIX AS THE IMAGE FOLDER PATH}"

def init_model():
    processor = BlipProcessor.from_pretrained(MODEL_PATH)
    model = Blip2ForConditionalGeneration.from_pretrained(MODEL_PATH)
    if torch.cuda.is_available():
        device = "cuda"
        model = model.to("cuda")
    else:
        device = "cpu"
    model.eval()
    return model, processor, device

def process_batch(image_paths, model, processor, device):
    images = [Image.open(img_path).convert('RGB') for img_path in image_paths]

    with torch.no_grad():
        inputs = processor(images, [QUESTION]*len(images), return_tensors="pt", padding=True).to(device)
        outputs = model.generate(**inputs)
        outputs_decodes = processor.batch_decode(outputs, skip_special_tokens=True)

    torch.cuda.empty_cache()
    return outputs_decodes


def call_blip(base_image_path, destination_caption_path, model, processor, device):
    
    image_paths = []
    try:
        os.makedirs(destination_caption_path)
    except:
        pass
    files = os.listdir(base_image_path)
    for file in files:
        image_path = os.path.join(base_image_path, file)
        image_paths.append(image_path)
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        batch_output = process_batch(image_paths[i:i+BATCH_SIZE], model, processor, device)
        for output, image_path in zip(batch_output, image_paths[i:i+BATCH_SIZE]):
            parts = image_path.split("/")
            new_end = parts[-1].replace(".jpg", ".out")
            new_end = new_end.replace(".png", ".out")
            new_end = new_end.replace(".JPEG", ".out")
            new_path = os.path.join(destination_caption_path, new_end)
            with open(new_path, 'w') as file:
                file.write(output)

    

if __name__ == '__main__':
    print("**************loading**************")
    model, processor, device = init_model()
    print("**************generating**************")
    call_blip(IMAGE_FOLDER, DES_FOLDER, model, processor, device)