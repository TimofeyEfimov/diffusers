from diffusers import DDPMPipeline
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch 
import torch.nn as nn
import numpy as np
import os 
from accelerate import PartialState

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/ddpm-cifar10-32"

noise_scheduler = DDPMScheduler(num_train_timesteps=1100)
# Load the model and scheduler
ddpm = DDPMPipeline.from_pretrained(model_id)
from diffusers import UNet2DModel


unet = UNet2DModel.from_pretrained(
    model_id
)


pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)

distributed_state = PartialState()
pipeline.to(distributed_state.device)

print(pipeline.config)

# Generate images
batch_size = 2


# Save each image in the batch
output_dir = "/home/tefimov/diffusers/src/NewSamplers/images/"

os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
with distributed_state.split_between_processes([batch_size, batch_size]) as prompt:
    images11 = pipeline(batch_size=batch_size).images
    for i, image in enumerate(images11):
        image_path = os.path.join(output_dir, f"image11_{distributed_state.process_index}_{i}.png")
        image.save(image_path)
        print(f"Saved {image_path}")

# Example for saving images from another pipeline (ddpm in this case)
images10 = ddpm(batch_size=batch_size).images
for i, image in enumerate(images10):
    image_path = os.path.join(output_dir, f"image10_{i}.png")
    image.save(image_path)
    print(f"Saved {image_path}")

