import os
import numpy as np
import pickle
from PIL import Image
from collections import defaultdict

import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoPipelineForText2Image

# local imports
from construct_prompts import get_prompts_concrete, get_prompts_style, get_prompts_human_related
from controller import VectorStore, register_vector_control

# parsing arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['sd14', 'sd21', 'sd21-turbo', 'sdxl', 'sdxl-turbo'], default="sd14")
parser.add_argument('--mode', type=str, choices=['concrete', 'human-related', 'style'], default="style")
parser.add_argument('--num_denoising_steps', type=int, default=50) # 50 for sd14, sd21, 1 for turbo, 30 for sdxl
parser.add_argument('--concept_pos', type=str, default="anime")
parser.add_argument('--concept_neg', type=str, default=None)
parser.add_argument('--save_dir', type=str, default='steering_vectors') # path to saving steering vectors
args = parser.parse_args()



if args.model == 'sd14':
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
         torch_dtype=torch.float16, 
        cache_dir='./cache'
        )
elif args.model == 'sd21':
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
         torch_dtype=torch.float16, 
        cache_dir='./cache'
        )
elif args.model == 'sd21-turbo':
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo", 
        torch_dtype=torch.float16, 
        variant="fp16",
        cache_dir='./cache'
    )
elif args.model == 'sdxl':
     pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        variant="fp16",
         cache_dir='./cache'
    )
elif args.model == 'sdxl-turbo':
     pipe = AutoPipelineForText2Image.from_pretrained(
         "stabilityai/sdxl-turbo", 
         torch_dtype=torch.float16, 
         variant="fp16",
         cache_dir='./cache'
     ) 
        
        
def run_model(model_type, pipe, prompt, seed, num_denoising_steps):
    if args.model in ['sd14', 'sd21', 'sdxl']:
        image = pipe(prompt=prompt, 
                     num_inference_steps=num_denoising_steps, 
                     generator=torch.Generator(device=device).manual_seed(seed)
                    ).images[0]
      
    elif args.model in ['sd21-turbo', 'sdxl-turbo']:
        image = pipe(prompt=prompt, 
                     num_inference_steps=num_denoising_steps,
                     guidance_scale=0.0,
                     generator=torch.Generator(device=device).manual_seed(seed)
                    ).images[0]
            
    return image
    
    
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipe.to(device)


if args.mode == 'concrete':
    prompts_pos, prompts_neg = get_prompts_concrete(concept_pos=args.concept_pos, 
                                                    concept_neg=args.concept_neg)
elif args.mode == 'human-related':
    prompts_pos, prompts_neg = get_prompts_human_related(concept_pos=args.concept_pos, 
                                                         concept_neg=args.concept_neg)
elif args.mode == 'style':
    prompts_pos, prompts_neg = get_prompts_style(concept_pos=args.concept_pos, 
                                                 concept_neg=args.concept_neg)
    

# Calculating CA outputs for generating steering vectors 
pos_vectors = []
neg_vectors = []
seed=0

for i, (prompt_pos, prompt_neg) in enumerate(zip(prompts_pos, prompts_neg)):
    print('Prompt pair number', i, 'out of', len(prompts_pos))
    print('Positive prompt:', prompt_pos)
    print('Negative prompt', prompt_neg)

    controller = VectorStore()
    controller.steer=False
    register_vector_control(pipe.unet, controller)

    image = run_model(args.model, pipe, prompt_pos, seed, args.num_denoising_steps)

    pos_vectors.append(controller.vector_store)

    controller = VectorStore()
    controller.steer=False
    register_vector_control(pipe.unet, controller)

    image = run_model(args.model, pipe, prompt_neg, seed, args.num_denoising_steps)

    neg_vectors.append(controller.vector_store)


# Calculating steering vectors
steering_vectors = {} 

for denoising_step in range(0, args.num_denoising_steps):
    steering_vectors[denoising_step] = defaultdict(list)
    
    for key in ['up', 'down', 'mid']:
        for layer_num in range(len(pos_vectors[0][denoising_step][key])):
            
            pos_vectors_layer = [pos_vectors[i][denoising_step][key][layer_num] for i in range(len(pos_vectors))]
            pos_vectors_avg = np.mean(pos_vectors_layer, axis=0)
            
            neg_vectors_layer = [neg_vectors[i][denoising_step][key][layer_num] for i in range(len(neg_vectors))]
            neg_vectors_avg = np.mean(neg_vectors_layer, axis=0)
            
            steering_vector = pos_vectors_avg - neg_vectors_avg
            steering_vector = steering_vector / np.linalg.norm(steering_vector)
            
            steering_vectors[denoising_step][key].append(steering_vector)


# Saving steering vectors:
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
with open(os.path.join(args.save_dir, '{}_{}_{}.pickle'.format(args.model, args.concept_pos, args.concept_neg)), 'wb') as handle:
    pickle.dump(steering_vectors, handle)