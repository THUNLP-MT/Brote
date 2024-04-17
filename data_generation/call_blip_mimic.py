# for mimic dataset, where image is in bytes in parquet

from transformers import BlipProcessor, Blip2ForConditionalGeneration
from PIL import Image
import os, sys
import torch
from tqdm import tqdm
import pandas as pd
from PIL import Image
import base64
from io import BytesIO

QUESTION = "Please describe this image in detail."
BATCH_SIZE = 256

MODEL_PATH = "your blip2 model path"
IMAGE_FOLDER = "your path for the parquet file of mimic"
DES_FOLDER = "the folder you hope of the caption" # should be "../captions/CGD"

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image

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

def process_batch_parquet(images, model, processor, device):
    with torch.no_grad():
        inputs = processor(images, [QUESTION]*len(images), return_tensors="pt", padding=True).to(device)
        outputs = model.generate(**inputs)
        outputs_decoded = processor.batch_decode(outputs, skip_special_tokens=True)

    torch.cuda.empty_cache()
    return outputs_decoded

                    
def call_blip_from_parquet(parquet_file, destination_caption_path, model, processor, device):
    if not os.path.exists(destination_caption_path):
        os.mkdir(destination_caption_path)

    df = pd.read_parquet(parquet_file)
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch_df = df[i:i+BATCH_SIZE]
        images = [decode_base64_to_image(b64_str) for b64_str in batch_df['base64']]
        batch_output = process_batch_parquet(images, model, processor, device)
        
        for key, caption in zip(batch_df.index, batch_output):
            destination = os.path.join(destination_caption_path, f"{key}.out")
            with open(destination, "w") as file:
                file.write(caption)


if __name__ == '__main__':
    print("**************loading**************")
    model, processor, device = init_model()
    print("**************generating**************")
    call_blip_from_parquet(IMAGE_FOLDER, DES_FOLDER, model, processor, device)
