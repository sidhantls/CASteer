import os
import numpy as np
import pickle
from PIL import Image
from collections import defaultdict
import time

import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoPipelineForText2Image

# local imports
from controller import VectorStore, register_vector_control

# parsing arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['sd14', 'sd21', 'sd21-turbo', 'sdxl', 'sdxl-turbo'], default="sd14")
parser.add_argument('--prompt', type=str, default="a girl with a kitty")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--steering_vectors', type=str) # path to steering vectors file
parser.add_argument('--not_steer', action='store_true')
parser.add_argument('--steer_only_up', action='store_true')
parser.add_argument('--num_denoising_steps', type=int, default=50) # 50 for sd14, sd21, 1 for turbo, 30 for sdxl
parser.add_argument('--steer_back', action='store_true')
parser.add_argument('--alpha', type=int, default=10)
parser.add_argument('--beta', type=int, default=2)
parser.add_argument('--save_dir', type=str, default='images') # path to saving generated images
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
        
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipe.to(device)
        
        
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

print('Generating for prompt:')
print(args.prompt)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if args.not_steer:
    image = run_model(args.model, pipe, args.prompt, args.seed, args.num_denoising_steps)
    
    image.save(os.path.join(args.save_dir, "orig_{}_{}.png".format(args.prompt, args.seed)))
    
    
else:
    with open(args.steering_vectors, 'rb') as handle:
        steering_vectors = pickle.load(handle)
        
    controller = VectorStore(steering_vectors, device=device)
    controller.steer_only_up = True if args.steer_only_up else False
    if args.steer_back:
        controller.steer_back = True
        controller.beta = args.beta
    else:
        controller.steer_back = False
        controller.alpha = args.alpha
    
    register_vector_control(pipe.unet, controller)
    
    image = run_model(args.model, pipe, args.prompt, args.seed, args.num_denoising_steps)
    
    image.save(os.path.join(args.save_dir, "steered_{}_{}.png".format(args.prompt, args.seed)))
    

