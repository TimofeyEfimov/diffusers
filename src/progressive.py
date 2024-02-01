from diffusers import DDPMPipeline
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch 
import torch.nn as nn
import numpy as np
import os 
from accelerate import PartialState
from diffusers import ConsistencyModelPipeline
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/ddpm-ema-bedroom-256"



def create_steps(start, end, step_increment):
    steps = [i for i in range(start, end + 1, step_increment)]
    return steps

# Example usage to create steps from 600 to 1000, in increments of 50
start_step = 800
end_step = 1000
step_increment = 50
steps = create_steps(start_step, end_step, step_increment)
print(steps)


for step in steps:
    current_steps = step
    # noise_scheduler = DDPMScheduler(timestep_spacing='linspace')
    # Load the model and scheduler
    #ddpm = DDPMPipeline.from_pretrained(model_id)
    from diffusers import UNet2DModel


    unet = UNet2DModel.from_pretrained(
        model_id
    )


    seed1, seed2, seed3 = 1,2,3
    # pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
    pipeline = DDPMPipeline.from_pretrained(model_id)


    print(pipeline.scheduler.config)
    distributed_state = PartialState()
    pipeline.to(distributed_state.device)

    # Generate images
    batch_size = 50
    seed1 = 0

    # Save each image in the batch
    output_dir = "/home/tefimov/diffusers/src/NewSamplers/ChurchesEMA/ProgressiveSmallT/VanillaTest/"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists


    num_iterations = 1
    for iteration in range(num_iterations):
        start_time = time.time()
        seed1 = seed1+1
        with distributed_state.split_between_processes([seed1]) as prompt:
            print("prompt is", prompt[0])
            images = pipeline(batch_size=batch_size,generator=torch.manual_seed(prompt[0]), num_inference_steps=current_steps).images
            for i, image in enumerate(images):
                image_number = (iteration+0) * batch_size * 1 + i
                image_path = os.path.join(output_dir, f"image_{image_number}_seed_{current_steps}.png")
                image.save(image_path)
                print(f"Saved {image_path}")
        end_time = time.time()
        iteration_time = end_time - start_time
        print(f"Time for one iteration: {iteration_time} seconds")



device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/ddpm-ema-bedroom-256"



def create_steps(start, end, step_increment):
    steps = [i for i in range(start, end + 1, step_increment)]
    return steps

# Example usage to create steps from 600 to 1000, in increments of 50
start_step = 25
end_step = 200
step_increment = 25
steps = create_steps(start_step, end_step, step_increment)
print(steps)


for step in steps:
    current_steps = step
    noise_scheduler = DDPMScheduler()
    # Load the model and scheduler
    #ddpm = DDPMPipeline.from_pretrained(model_id)
    from diffusers import UNet2DModel


    # unet = UNet2DModel.from_pretrained(
    #     model_id
    # )


    seed1, seed2, seed3 = 1,2,3
    pipeline = DDPMPipeline.from_pretrained(model_id)

    print(pipeline.scheduler.config)
    distributed_state = PartialState()
    pipeline.to(distributed_state.device)

    print(pipeline.config)

    # Generate images
    batch_size = 1
    seed1 = 0

    # Save each image in the batch
    output_dir = "/home/tefimov/diffusers/src/NewSamplers/Bedroom/ProgressiveSmallT/VanillaSDE_50/"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists


    num_iterations = 1
    for iteration in range(num_iterations):
        start_time = time.time()
        seed1 = seed1+1
        with distributed_state.split_between_processes([seed1]) as prompt:
            print("prompt is", prompt[0])
            images = pipeline(batch_size=batch_size,generator=torch.manual_seed(prompt[0]), num_inference_steps=current_steps).images
            for i, image in enumerate(images):
                image_number = (iteration+0) * batch_size * 1 + i
                image_path = os.path.join(output_dir, f"image_{image_number}_seed_{current_steps}.png")
                image.save(image_path)
                print(f"Saved {image_path}")
        end_time = time.time()
        iteration_time = end_time - start_time
        print(f"Time for one iteration: {iteration_time} seconds")

