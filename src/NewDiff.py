from diffusers import DDPMScheduler, UNet2DModel
from diffusers import DiffusionPipeline
from PIL import Image
import torch
import os 
import torch.distributions as dist
import numpy as np

scheduler = DDPMScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
model = UNet2DModel.from_pretrained("google/ddpm-ema-celebahq-256").to("cuda")


# pipeline = DiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
# )


scheduler.set_timesteps(15)
# Main code to dynamically generate images for different timesteps

for i in range(10, 100, 10):
    scheduler.set_timesteps(i)
    sample_size = model.config.sample_size
    noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")  # Replace "cpu" with "cuda" if using GPU
    input = noise

    p = 0.7  # Probability of success
    n = 1   
    binomial_dist = dist.Binomial(total_count=n, probs=p)

    # Create matrix A with the same shape as y_data using iid binomial distribution
    A = binomial_dist.sample(input.size())

    # Multiply the two tensors element-wise
    #result = sample * A
    previous_output = None
    file_name1 = f"TestImages/VanillaSDE/VanillaSDE_300.png"
    image1 = Image.open(file_name1)

    image_np = np.array(image1)

    # Convert the NumPy array to a PyTorch tensor
    image_tensor = torch.tensor(image_np.astype("float32") / 255)

    
    # Undo the permute operation: from (H, W, C) to (C, H, W)
    image_tensor = image_tensor.permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0)

    # Undo the normalization: from [0, 1] range to the original tensor before saving
    input_tensor4 = (image_tensor - 0.5) * 2


    # for t in scheduler.timesteps:
    #     with torch.no_grad():
        
    #         noisy_residual = model(input, t).sample
            
    #         prev_noisy_sample = scheduler.step(model, noisy_residual, t, input).prev_sample
    #         input = prev_noisy_sample
    #         previous_output = noisy_residual
    
    image = (input_tensor4 / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = Image.fromarray((image * 255).round().astype("uint8"))

    file_name = f"TestImages/VanillaSDE/VanillaSDE_{i}.png"
    image.save(file_name)
    print(f"Saved: {file_name}")