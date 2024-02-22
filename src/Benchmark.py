from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler
from diffusers import DiffusionPipeline
from PIL import Image
import torch
import os 

# scheduler = DDPMScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
# model = UNet2DModel.from_pretrained("google/ddpm-ema-celebahq-256").to("cuda")


scheduler = DDPMScheduler.from_pretrained("google/ddpm-bedroom-256")
#scheduler.beta_schedule = "scaled_linear"
model = UNet2DModel.from_pretrained("google/ddpm-bedroom-256").to("cuda")

model = model.to("cuda")

num_timesteps = 30
scheduler.set_timesteps(num_timesteps)
seeds = [0,1,2,3,4,5,6,7,8]  # Example seeds, you can define your own list

type_model ="NewODE"
type_dataset = "Bedroom"
# Create directory if it doesn't exist
directory = f"NewBenchmark/{type_dataset}/{type_model}/{num_timesteps}_timesteps"
if not os.path.exists(directory):
    os.makedirs(directory)

for seed in seeds:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    sample_size = model.config.sample_size
    noise = torch.randn((1, 3, sample_size, sample_size), generator=generator, device="cuda")  # Replace "cpu" with "cuda" if using GPU
    input = noise
    previous_output = None
    for t in scheduler.timesteps:
        input = torch.tensor(input, requires_grad=True)
        noisy_residual = model(input, t).sample

        prev_noisy_sample = scheduler.step(noisy_residual, t, input, model=model, generator=generator, previous_output=previous_output).prev_sample
        input = prev_noisy_sample
        previous_output = noisy_residual

    image = (input / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).detach().numpy()[0]
    image = Image.fromarray((image * 255).astype("uint8"))

    file_name = f"{directory}/newEq_seed_{seed}.png"
    image.save(file_name)
    print(f"Saved: {file_name}")