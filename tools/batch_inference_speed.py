import os
import time
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained("/home/couser/checkpoints/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained("/home/couser/checkpoints/Qwen2.5-VL-7B-Instruct")

single_message = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/data/tangkaihua/datasets/open_scene/depth/frame_00001.png"},
            {"type": "text", "text": "Is there a dog in the image? Answer Yes or No."},
        ],
    }
]

for batch_size in (1, 3, 5, 7):
    inference_time = []
    for i in range(25):
        messages = [single_message for _ in range(batch_size)]
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        start_time = time.time()
        generated_ids = model.generate(**inputs, max_new_tokens=5)
        end_time = time.time()
        
        inference_time.append(end_time - start_time)
    inference_time = sum(inference_time[5:]) / len(inference_time[5:])
    print(f"Inference time for batch size {batch_size} is : {inference_time}")