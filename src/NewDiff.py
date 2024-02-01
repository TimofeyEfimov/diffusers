from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
import torch
import os 

scheduler = DDPMScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
model = UNet2DModel.from_pretrained("google/ddpm-ema-celebahq-256").to("cuda")

scheduler.set_timesteps(15)
# Main code to dynamically generate images for different timesteps

for i in range(5, 200, 5):
    scheduler.set_timesteps(i)
    sample_size = model.config.sample_size
    noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")  # Replace "cpu" with "cuda" if using GPU
    input = noise

    for t in scheduler.timesteps:
        with torch.no_grad():
            noisy_residual = model(input, t).sample
            prev_noisy_sample = scheduler.step(model, noisy_residual, t, input).prev_sample
            input = prev_noisy_sample

    image = (input / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = Image.fromarray((image * 255).round().astype("uint8"))

    file_name = f"TestImages/vanillaSDE_{i}.png"
    image.save(file_name)
    print(f"Saved: {file_name}")