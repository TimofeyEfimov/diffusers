from diffusers import DDPMPipeline
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch 
import torch.nn as nn
import numpy as np
import os 
from accelerate import PartialState
from diffusers import ConsistencyModelPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/ddpm-ema-celebahq-256"



ddpm = DDPMPipeline.from_pretrained(model_id)
print(ddpm.config)

steps = 750
noise_scheduler = DDPMScheduler()
# Load the model and scheduler
ddpm = DDPMPipeline.from_pretrained(model_id)
from diffusers import UNet2DModel


unet = UNet2DModel.from_pretrained(
    model_id
)

seed1, seed2, seed3 = 1,2,3
pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)

print(pipeline.scheduler.config)
distributed_state = PartialState()
pipeline.to(distributed_state.device)

print(pipeline.config)

# Generate images
batch_size = 30

# Save each image in the batch
output_dir = "/home/tefimov/diffusers/src/NewSamplers/CelebaEMA/90samples_NewSDE750/"

os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
with distributed_state.split_between_processes([seed1, seed2, seed3]) as prompt:
    print("prompt is", prompt[0])
    images11 = pipeline(batch_size=batch_size,generator=torch.manual_seed(prompt[0]), num_inference_steps=steps).images
    for i, image in enumerate(images11):
        image_path = os.path.join(output_dir, f"{distributed_state.process_index}_{i}.png")
        image.save(image_path)
        print(f"Saved {image_path}")

