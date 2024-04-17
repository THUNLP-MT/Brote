#!/bin/bash

python3 train.py \
    --coco False \
    --flickr False \
    --okvqa False \
    --vqa False \
    --nlvr2 False \
    --llava False \
    --stvqa True \
    --vcr False \
    --ivqa False \
    --ivqa_frame False \
    --vsr False \
    --iconqa False \
    --CGD False \
    --LA False \
    --GPT4V False \
    --GPT4 True \
    --GPT3 False \
	--starts_from $1 \
    --num_example 2000
