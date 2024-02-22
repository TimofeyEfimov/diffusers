from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler
from diffusers import DiffusionPipeline
from PIL import Image
import torch
import os 

# scheduler = DDPMScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
# model = UNet2DModel.from_pretrained("google/ddpm-ema-celebahq-256").to("cuda")


pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

scheduler = DDPMScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler.beta_schedule = "linear"
# pipeline.scheduler = scheduler

print("pipeline is")
#print(pipeline)
# print(pipeline.scheduler, pipeline.scheduler.config)
# print(pipeline.scheduler.config)
# prompt = "Carnegie Mellon University student."

# print("Before generator")
# generator = torch.Generator(device="cuda").manual_seed(8)
# print("After generator")
#image = pipeline(prompt, generator=generator, num_inference_steps=1000).images[0]
# print("After image")

file_name = f"TestImages/SD/DDIM_Test.png"
#image.save(file_name)

prompt = "Banana riding the bicycle"
num_inference_steps = 15
seeds = [0,1,2,3,4,5,6,7,8]  # Example seeds, you can define your own list

for seed in seeds:
    print(f"Before generator for seed {seed}")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    print(f"After generator for seed {seed}")
    
    # Generate the image with the current seed and specified number of inference steps
    image = pipeline(prompt, generator=generator, num_inference_steps=num_inference_steps).images[0]

    print(pipeline.config)
    print(f"After image generation for seed {seed}")

    # Construct the filename to include the sseed and number of steps
    #file_name = f"TestImages/OLD_ODE_new/1order_new_seeds_{seed}_steps_{num_inference_steps}.png"
    file_name = f"TestImages/Banana/1storder_{seed}_steps_{num_inference_steps}.png"

    #file_name = f"TestImages/SD_myDDIM/myDDIM_seeds_{seed}_steps_{num_inference_steps}.png"
    
    # Save the image
    image.save(file_name)
    print(f"Image saved: {file_name}")


