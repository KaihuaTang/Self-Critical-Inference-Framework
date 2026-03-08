from __future__ import annotations

import os
import sys
import warnings
import math
import logging

import torch
from PIL import Image
from torchvision import transforms

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from ...smp import get_rank_and_world_size, get_gpu_memory, auto_split_flag, listinstr

try:
    from qwen_vl_utils import process_vision_info
except Exception as err:
    logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
    raise err

def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def split_model():
    device_map = {}

    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = total_gpus // world_size
    # + 8 is virtual layers for the memory of visual
    num_layers = 80 + 8
    num_layers_per_gpu = math.ceil(num_layers / num_gpus)
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] -= 6
    num_layers_per_gpu[-1] -= 2
    layer_cnt = 0

    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'model.layers.{layer_cnt}'] = rank + i * world_size
            layer_cnt += 1

    last_gpu = rank + (num_gpus - 1) * world_size
    device_map['visual'] = rank
    device_map['model.embed_tokens'] = rank
    device_map['model.norm'] = last_gpu
    device_map['model.rotary_emb'] = last_gpu
    device_map['lm_head'] = last_gpu
    return device_map

# the following code is copied from Visual Contrastive Decoding
# https://github.com/DAMO-NLP-SG/VCD/blob/master/vcd_utils/vcd_add_noise.py
def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000  # Number of diffusion steps
    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)
    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 
    return image_tensor_cd



def prompt_variation1(texts):
    # new prompt text
    new_texts = []
    for text_item in texts:
        if '请直接回答选项字母。' in text_item:
            new_texts.append(text_item.replace('请直接回答选项字母。', '结合问题与选项仔细观察图像中的信息，请直接回答选项字母。'))
        elif 'Please select the correct answer from the options above.' in text_item:
            new_texts.append(text_item.replace('Please select the correct answer from the options above.', 'Think about the question based on details in the given image. Please select the correct answer from the options above.'))
        elif 'Please answer yes or no.' in text_item:
            new_texts.append(text_item.replace('Please answer yes or no.', 'Think about the question based on details in the given image. Please answer yes or no.'))
        elif 'Please try to answer the question with short words or phrases if possible.' in text_item:
            new_texts.append(text_item.replace('Please try to answer the question with short words or phrases if possible.', 'Think about the question based on details in the given image. Please try to answer the question with short words or phrases if possible.'))
        elif 'Answer the question directly using a single word or phrase.' in text_item:
            new_texts.append(text_item.replace('Answer the question directly using a single word or phrase.', 'Think about the question based on details in the given image. Answer the question directly using a single word or phrase.'))
        else:
            raise ValueError(f"Invalid prompt text: {text_item}")
    return new_texts


def prompt_variation2(texts):
    # new prompt text
    new_texts = []
    for text_item in texts:
        if '请直接回答选项字母。' in text_item:
            new_texts.append(text_item.replace('请直接回答选项字母。', 'Please carefully examine the information in the image, then consider the question and options, and reply directly with the letter corresponding to the correct answer from the options above.'))
        elif 'Please select the correct answer from the options above.' in text_item:
            new_texts.append(text_item.replace('Please select the correct answer from the options above.', '请仔细观察图像中的信息，然后结合问题与选项，从上述所有选项中直接回答正确选项对应的字母。'))
        elif 'Please answer yes or no.' in text_item:
            #new_texts.append(text_item.replace('Please answer yes or no.', '请直接回答yes或no。'))
            new_texts.append(text_item.replace('Please answer yes or no.', '观察给出的图片，请直接回答yes或no。'))
        elif 'Please try to answer the question with short words or phrases if possible.' in text_item:
            new_texts.append(text_item.replace('Please try to answer the question with short words or phrases if possible.', '请仔细观察图像中的细节，然后结合图像上的信息回答问题，请直接用一个简短的英语单词或数字回答。'))
        elif 'Answer the question directly using a single word or phrase.' in text_item:
            new_texts.append(text_item.replace('Answer the question directly using a single word or phrase.', '请仔细观察图像中的细节，然后结合图像上的信息回答问题，请直接用一个简短的英语单词或数字回答。'))
        else:
            raise ValueError(f"Invalid prompt text: {text_item}")
    return new_texts


def prompt_variation3(texts):
    # new prompt text
    new_texts = []
    for text_item in texts:
        if '请直接回答选项字母。' in text_item:
            new_texts.append(text_item.replace('请直接回答选项字母。', '你是一名擅长回答选择题的聪明学生，请直接回答选项字母。'))
        elif 'Please select the correct answer from the options above.' in text_item:
            new_texts.append(text_item.replace('Please select the correct answer from the options above.', 'You are a smart student who is good at answering multiple-choice questions. Please select the correct answer from the options above.'))
        elif 'Please answer yes or no.' in text_item:
            new_texts.append(text_item.replace('Please answer yes or no.', 'You are a smart student who is good at answering yes or no questions. Please answer yes or no.'))
        elif 'Please try to answer the question with short words or phrases if possible.' in text_item:
            new_texts.append(text_item.replace('Please try to answer the question with short words or phrases if possible.', 'You are a smart student who is good at answering questions. Please try to answer the question with short words or phrases if possible.'))
        elif 'Answer the question directly using a single word or phrase.' in text_item:
            new_texts.append(text_item.replace('Answer the question directly using a single word or phrase.', 'You are a smart student who is good at answering questions. Answer the question directly using a single word or phrase.'))
        else:
            raise ValueError(f"Invalid prompt text: {text_item}")
    return new_texts



class Qwen2VLChat(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=2048,
        model_name=None,
        save_logits=False,
        dump_path=None,
        dtype=torch.bfloat16,
        visual_type='default',
        textual_type='default',
        alpha=0.0,
        beta=0.0,
        gamma=0.0,
        theta=0.0,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = False,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        # custom parameters
        self.visual_type = visual_type
        self.textual_type = textual_type
        print(f"==> Using Image type: {visual_type}")
        print(f"==> Using Prompt type: {textual_type}")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        if self.visual_type in ('TIE', 'VCD', 'M3ID', 'SCI3', 'SCI5', 'SCI7', 'Ablation1', 'Ablation2'):
            self.generate_kwargs = dict(
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                return_dict_in_generate=True,   # <‑‑ ask for a dict‑like GenerationOutput
                output_logits=True,
            )
        else:
            self.generate_kwargs = dict(
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2
        rank, world_size = get_rank_and_world_size()
        assert model_path is not None
        self.model_path = model_path
        MODEL_CLS = None

        if listinstr(['2.5', '2_5', 'qwen25'], model_path.lower()):
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            from transformers import Qwen2VLProcessor
            from .modeling_qwen2_vl import Qwen2VLForConditionalGeneration
            MODEL_CLS = Qwen2VLForConditionalGeneration
            self.processor = Qwen2VLProcessor.from_pretrained(model_path)
        
        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        # If only one process and GPU memory is less than 40GB
        if '72b' in self.model_path.lower():
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype=dtype, device_map=split_model(), attn_implementation='flash_attention_2'
            )
            self.model.eval()
        elif auto_split_flag():
            assert world_size == 1, 'Only support world_size == 1 when AUTO_SPLIT is set for non-72B Qwen2-VL'
            # Will Use All GPUs to run one model
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype=dtype, device_map='auto', attn_implementation='flash_attention_2'
            )
        else:
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype=dtype, device_map='cpu', attn_implementation='flash_attention_2'
            )
            self.model.cuda().eval()

        # Kaihua Modified
        if save_logits and (model_name is not None) and (dump_path is not None):
            save_path = os.path.join(dump_path, model_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print(f"Saving logit tensors to {save_path}")
            self.model.dump_path = save_path
        else:
            self.model.dump_path = None


        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
            elif s['type'] == 'video':
                item = {'type': 'video', 'video': ensure_video_url(s['value'])}
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_ids_or_logits(self, messages, visual_type, textual_type, cf_logits=None, cf_params=None, get_logits=False):
        texts = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info([messages])

        # process image
        processed_images = []
        if visual_type in ('default', 'TIE', 'VCD', 'M3ID', 'SCI3', 'SCI5', 'SCI7', 'Ablation1', 'Ablation2'):
            processed_images = images
        elif visual_type == 'vcf_color0':
            for image in images:
                black_image = Image.new("RGB", image.size, (0, 0, 0))
                processed_images.append(black_image)
        elif visual_type == 'vcf_color255':
            for image in images:
                white_image = Image.new("RGB", image.size, (255, 255, 255))
                processed_images.append(white_image)
        elif visual_type == 'vcf_noise400':
            for image in images:
                noise_img = add_diffusion_noise(transforms.ToTensor()(image), noise_step=400)
                processed_images.append(transforms.ToPILImage()(noise_img.cpu()))
        elif visual_type == 'vcf_noise500':
            for image in images:
                noise_img = add_diffusion_noise(transforms.ToTensor()(image), noise_step=500)
                processed_images.append(transforms.ToPILImage()(noise_img.cpu()))
        else:
            raise ValueError("Wrong Image Type")
        
        # process text
        processed_texts = []
        if textual_type in ('default', 'TIE', 'VCD', 'M3ID', 'SCI3', 'SCI5', 'SCI7', 'Ablation1', 'Ablation2'):
            processed_texts = texts
        elif textual_type == 'tcf_v1':
            processed_texts = prompt_variation1(texts)
        elif textual_type == 'tcf_v2':
            processed_texts = prompt_variation2(texts)
        elif textual_type == 'tcf_v3':
            processed_texts = prompt_variation3(texts)
        else:
            raise ValueError("Wrong Text Type")
        
        # inference
        if visual_type in ('TIE', 'VCD', 'M3ID', 'SCI3', 'SCI5', 'SCI7', 'Ablation1', 'Ablation2'):
            assert (cf_logits is not None) and (cf_params is not None)
            inputs = self.processor(text=processed_texts, images=processed_images, videos=videos, padding=True, return_tensors='pt')
            inputs = inputs.to('cuda')
            output_dicts = self.model.generate(
                    cf_logits = cf_logits,
                    cf_params = cf_params,
                    **inputs,
                    **self.generate_kwargs,
                )
            logits = torch.cat(output_dicts.logits, dim=0)
            generated_ids = logits.unsqueeze(0).argmax(dim=-1)
            return generated_ids
        else:
            # Original Code
            inputs = self.processor(text=processed_texts, images=processed_images, videos=videos, padding=True, return_tensors='pt')
            inputs = inputs.to('cuda')
            if get_logits:
                output_dicts = self.model.generate(
                    **inputs,
                    **self.generate_kwargs,
                )
                logits = torch.cat(output_dicts.logits, dim=0)
                return logits.unsqueeze(0).float()
            else:
                generated_ids = self.model.generate(
                    **inputs,
                    **self.generate_kwargs,
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                ]
                return generated_ids

    def generate_inner(self, message, dataset=None):
        

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        if self.visual_type == 'TIE':
            # We inplement TIE based on paper Counterfactual VQA and Unbiased Scene Graph Generation
            tie_logits = self.generate_ids_or_logits(messages, visual_type='vcf_noise500', textual_type='default', get_logits=True)
            cf_logits = {"tie_logits": tie_logits}
            cf_params = {"theta": self.theta}
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'VCD':
            # Visual Contrastive Decoding (VCD)
            # Current BS-Subsets only predict one character or one word, so we don't need to take care of auto-regressive generation, focusing on the first token is enough. 
            # It can be generalized to iterative generation in future work if needed.
            vcd_logits = self.generate_ids_or_logits(messages, visual_type='vcf_noise500', textual_type='default', get_logits=True)
            cf_logits = {"vcd_logits": vcd_logits}
            cf_params = {"alpha": self.alpha, "theta": self.theta}
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'M3ID':
            m3id_logits = self.generate_ids_or_logits(messages, visual_type='vcf_noise500', textual_type='default', get_logits=True)
            cf_logits = {"m3id_logits": m3id_logits}
            cf_params = {"alpha": self.alpha, "theta": self.theta}
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'SCI3':
            vcf1_logits = self.generate_ids_or_logits(messages, visual_type='vcf_color0', textual_type='default', get_logits=True)
            tcf1_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v1', get_logits=True)
            cf_logits = {"vcf1_logits": vcf1_logits, "tcf1_logits": tcf1_logits}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'SCI5':
            # The proposed self-critical inference
            vcf1_logits = self.generate_ids_or_logits(messages, visual_type='vcf_color0', textual_type='default', get_logits=True)
            vcf2_logits = self.generate_ids_or_logits(messages, visual_type='vcf_noise500', textual_type='default', get_logits=True)
            tcf1_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v1', get_logits=True)
            tcf2_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v2', get_logits=True)
            cf_logits = {"vcf1_logits": vcf1_logits, "vcf2_logits": vcf2_logits, "tcf1_logits": tcf1_logits, "tcf2_logits": tcf2_logits}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'SCI7':
            vcf1_logits = self.generate_ids_or_logits(messages, visual_type='vcf_color0', textual_type='default', get_logits=True)
            vcf2_logits = self.generate_ids_or_logits(messages, visual_type='vcf_noise500', textual_type='default', get_logits=True)
            vcf3_logits = self.generate_ids_or_logits(messages, visual_type='vcf_noise400', textual_type='default', get_logits=True)
            tcf1_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v1', get_logits=True)
            tcf2_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v2', get_logits=True)
            tcf3_logits = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v3', get_logits=True)
            cf_logits = {"vcf1_logits": vcf1_logits, "vcf2_logits": vcf2_logits, "vcf3_logits": vcf3_logits, "tcf1_logits": tcf1_logits, "tcf2_logits": tcf2_logits, "tcf3_logits": tcf3_logits}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'Ablation1':
            ablation_vcf1 = self.generate_ids_or_logits(messages, visual_type='vcf_color0', textual_type='default', get_logits=True)
            ablation_vcf2 = self.generate_ids_or_logits(messages, visual_type='vcf_noise500', textual_type='default', get_logits=True)
            ablation_tcf1 = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v1', get_logits=True)
            cf_logits = {"ablation_vcf1": ablation_vcf1, "ablation_vcf2": ablation_vcf2, "ablation_tcf1": ablation_tcf1}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        elif self.visual_type == 'Ablation2':
            ablation_vcf1 = self.generate_ids_or_logits(messages, visual_type='vcf_color0', textual_type='default', get_logits=True)
            ablation_tcf1 = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v1', get_logits=True)
            ablation_tcf2 = self.generate_ids_or_logits(messages, visual_type='default', textual_type='tcf_v2', get_logits=True)
            cf_logits = {"ablation_vcf1": ablation_vcf1, "ablation_tcf1": ablation_tcf1, "ablation_tcf2": ablation_tcf2}
            cf_params = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "theta": self.theta}
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type, cf_logits=cf_logits, cf_params=cf_params)
        else:
            generated_ids = self.generate_ids_or_logits(messages, visual_type=self.visual_type, textual_type=self.textual_type)


        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response
