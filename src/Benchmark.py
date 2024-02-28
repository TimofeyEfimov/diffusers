from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler, DPMSolverMultistepScheduler, DDPMPipeline
from diffusers import DiffusionPipeline
from PIL import Image
import torch
import os 

# scheduler = DDPMScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
# model = UNet2DModel.from_pretrained("google/ddpm-ema-celebahq-256").to("cuda")


#scheduler = DPMScheduler.from_pretrained("google/ddpm-ema-church-256")
scheduler = DPMSolverMultistepScheduler()
#scheduler.beta_schedule = "scaled_linear"
model_id = "google/ddpm-ema-church-256"
model = UNet2DModel.from_pretrained(model_id).to("cuda")
#scheduler = DDPMScheduler.from_pretrained(model_id)
DPM = DDPMPipeline.from_pretrained(model_id).to("cuda")

model = model.to("cuda")

num_timesteps = 8
scheduler.set_timesteps(num_timesteps)

i = 20  # This would be provided dynamically in the actual use case.
step = 1  # Assuming step is 1 for now, but it can be any positive integer.

# Generate seeds using a list comprehension
seeds = [x for x in range(0, i, step)]  # Example seeds, you can define your own list

type_model ="DPMSolver2"
type_dataset = "ChurchesEMA"
# Create directory if it doesn't exist
directory = f"NewBenchmark/{type_dataset}/{type_model}/{num_timesteps}_timesteps"
if not os.path.exists(directory):
    os.makedirs(directory)

for seed in seeds:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    print(generator)
    sample_size = model.config.sample_size
    noise = torch.randn((1, 3, sample_size, sample_size), generator=generator, device="cuda")  # Replace "cpu" with "cuda" if using GPU
    input = noise
    previous_output = None
    two_previous_output = None 

    # print(scheduler)

    for t in scheduler.timesteps:
        
        #input = torch.tensor(input, requires_grad=True)
        noisy_residual = model(input, t).sample
        
        #prev_noisy_sample = scheduler.step(noisy_residual, t, input, generator=generator, previous_output=previous_output, two_previous_output=two_previous_output).prev_sample
        prev_noisy_sample = scheduler.step(noisy_residual,t, input).prev_sample
        
        two_previous_output = previous_output
        input = prev_noisy_sample
        previous_output = noisy_residual

    image = (input / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).detach().numpy()[0]
    image = Image.fromarray((image * 255).astype("uint8"))

    file_name = f"{directory}/newEq_seed_{seed}.png"
    image.save(file_name)
    print(f"Saved: {file_name}")