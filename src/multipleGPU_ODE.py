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



ddpm = DDPMPipeline.from_pretrained(model_id)
print(ddpm.config)

steps = 700
noise_scheduler = DDPMScheduler()
# Load the model and scheduler
ddpm = DDPMPipeline.from_pretrained(model_id)
from diffusers import UNet2DModel


unet = UNet2DModel.from_pretrained(
    model_id
)


seed1, seed2, seed3 = 1,2,3
pipeline = DDPMPipeline.from_pretrained(model_id)

print(pipeline.scheduler.config)
distributed_state = PartialState()
pipeline.to(distributed_state.device)

print(pipeline.config)

# Generate images
batch_size = 50
seed1, seed2, seed3, seed4 = 0,1,2,3

# Save each image in the batch
output_dir = "/home/tefimov/diffusers/src/NewSamplers/Bedroom/SDE/1000_NewSDE_700/"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists


num_iterations = 5
for iteration in range(num_iterations):
    print(output_dir)
    print("It is for 800")
    start_time = time.time()
    seed1, seed2, seed3, seed4 = seed1+4,seed2+4,seed3+4, seed4+4
    with distributed_state.split_between_processes([seed1, seed2, seed3, seed4]) as prompt:
        print("prompt is", prompt[0])
        images = pipeline(batch_size=batch_size,generator=torch.manual_seed(prompt[0]), num_inference_steps=steps).images
        for i, image in enumerate(images):
            image_number = (iteration+0) * batch_size * 5 + i
            image_path = os.path.join(output_dir, f"image_{image_number}_seed_{prompt[0]}.png")
            image.save(image_path)
            print(f"Saved {image_path}")
    end_time = time.time()
    iteration_time = end_time - start_time
    print(f"Time for one iteration: {iteration_time} seconds")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/ddpm-ema-bedroom-256"



ddpm = DDPMPipeline.from_pretrained(model_id)
print(ddpm.config)

steps = 800
noise_scheduler = DDPMScheduler()
# Load the model and scheduler
ddpm = DDPMPipeline.from_pretrained(model_id)
from diffusers import UNet2DModel


unet = UNet2DModel.from_pretrained(
    model_id
)


seed1, seed2, seed3 = 1,2,3
pipeline = DDPMPipeline.from_pretrained(model_id)

print(pipeline.scheduler.config)
distributed_state = PartialState()
pipeline.to(distributed_state.device)

print(pipeline.config)

# Generate images
batch_size = 50
seed1, seed2, seed3, seed4 = 0,1,2,3

# Save each image in the batch
output_dir = "/home/tefimov/diffusers/src/NewSamplers/Bedroom/sDE/1000samples_NewSDE_800/"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists


num_iterations = 5
for iteration in range(num_iterations):
    print("It is for 800")

    start_time = time.time()
    seed1, seed2, seed3, seed4 = seed1+4,seed2+4,seed3+4, seed4+4
    with distributed_state.split_between_processes([seed1, seed2, seed3, seed4]) as prompt:
        print("prompt is", prompt[0])
        images = pipeline(batch_size=batch_size,generator=torch.manual_seed(prompt[0]), num_inference_steps=steps).images
        for i, image in enumerate(images):
            image_number = (iteration+0) * batch_size * 5 + i
            image_path = os.path.join(output_dir, f"image_{image_number}_seed_{prompt[0]}.png")
            image.save(image_path)
            print(f"Saved {image_path}")
    end_time = time.time()
    iteration_time = end_time - start_time
    print(f"Time for one iteration: {iteration_time} seconds")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/ddpm-ema-bedroom-256"



ddpm = DDPMPipeline.from_pretrained(model_id)
print(ddpm.config)

steps = 900
noise_scheduler = DDPMScheduler()
# Load the model and scheduler
ddpm = DDPMPipeline.from_pretrained(model_id)
from diffusers import UNet2DModel


unet = UNet2DModel.from_pretrained(
    model_id
)


seed1, seed2, seed3 = 1,2,3
pipeline = DDPMPipeline.from_pretrained(model_id)

print(pipeline.scheduler.config)
distributed_state = PartialState()
pipeline.to(distributed_state.device)

print(pipeline.config)

# Generate images
batch_size = 50
seed1, seed2, seed3, seed4 = 0,1,2,3

# Save each image in the batch
output_dir = "/home/tefimov/diffusers/src/NewSamplers/Bedroom/SDE/1000samples_NewSDE_900/"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists


num_iterations = 5
for iteration in range(num_iterations):
    print("It is for 800")
    start_time = time.time()
    seed1, seed2, seed3, seed4 = seed1+4,seed2+4,seed3+4, seed4+4
    with distributed_state.split_between_processes([seed1, seed2, seed3, seed4]) as prompt:
        print("prompt is", prompt[0])
        images = pipeline(batch_size=batch_size,generator=torch.manual_seed(prompt[0]), num_inference_steps=steps).images
        for i, image in enumerate(images):
            image_number = (iteration+0) * batch_size * 5 + i
            image_path = os.path.join(output_dir, f"image_{image_number}_seed_{prompt[0]}.png")
            image.save(image_path)
            print(f"Saved {image_path}")
    end_time = time.time()
    iteration_time = end_time - start_time
    print(f"Time for one iteration: {iteration_time} seconds")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/ddpm-ema-bedroom-256"



ddpm = DDPMPipeline.from_pretrained(model_id)
print(ddpm.config)

steps = 1000
noise_scheduler = DDPMScheduler()
# Load the model and scheduler
ddpm = DDPMPipeline.from_pretrained(model_id)
from diffusers import UNet2DModel


unet = UNet2DModel.from_pretrained(
    model_id
)


seed1, seed2, seed3 = 1,2,3
pipeline = DDPMPipeline.from_pretrained(model_id)

print(pipeline.scheduler.config)
distributed_state = PartialState()
pipeline.to(distributed_state.device)

print(pipeline.config)

# Generate images
batch_size = 50
seed1, seed2, seed3, seed4 = 0,1,2,3

# Save each image in the batch
output_dir = "/home/tefimov/diffusers/src/NewSamplers/Bedroom/ODE/1000samples_NewSDE_1000/"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists


num_iterations = 5
for iteration in range(num_iterations):
    print("It is for 800")
    start_time = time.time()
    seed1, seed2, seed3, seed4 = seed1+4,seed2+4,seed3+4, seed4+4
    with distributed_state.split_between_processes([seed1, seed2, seed3, seed4]) as prompt:
        print("prompt is", prompt[0])
        images = pipeline(batch_size=batch_size,generator=torch.manual_seed(prompt[0]), num_inference_steps=steps).images
        for i, image in enumerate(images):
            image_number = (iteration+0) * batch_size * 5 + i
            image_path = os.path.join(output_dir, f"image_{image_number}_seed_{prompt[0]}.png")
            image.save(image_path)
            print(f"Saved {image_path}")
    end_time = time.time()
    iteration_time = end_time - start_time
    print(f"Time for one iteration: {iteration_time} seconds")


# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_id = "google/ddpm-ema-church-256"



# ddpm = DDPMPipeline.from_pretrained(model_id)
# print(ddpm.config)

# steps = 700
# noise_scheduler = DDPMScheduler()
# # Load the model and scheduler
# ddpm = DDPMPipeline.from_pretrained(model_id)
# from diffusers import UNet2DModel


# unet = UNet2DModel.from_pretrained(
#     model_id
# )


# seed1, seed2, seed3 = 1,2,3
# pipeline = DDPMPipeline.from_pretrained(model_id)

# print(pipeline.scheduler.config)
# distributed_state = PartialState()
# pipeline.to(distributed_state.device)

# print(pipeline.config)

# # Generate images
# batch_size = 50
# seed1, seed2, seed3, seed4 = 0,1,2,3

# # Save each image in the batch
# output_dir = "/home/tefimov/diffusers/src/NewSamplers/ChurchesEMA/1000samples_VanillaODE700/"
# os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists


# num_iterations = 5
# for iteration in range(num_iterations):
#     print("It is for 800")
#     start_time = time.time()
#     seed1, seed2, seed3, seed4 = seed1+4,seed2+4,seed3+4, seed4+4
#     with distributed_state.split_between_processes([seed1, seed2, seed3, seed4]) as prompt:
#         print("prompt is", prompt[0])
#         images = pipeline(batch_size=batch_size,generator=torch.manual_seed(prompt[0]), num_inference_steps=steps).images
#         for i, image in enumerate(images):
#             image_number = (iteration+0) * batch_size * 5 + i
#             image_path = os.path.join(output_dir, f"image_{image_number}_seed_{prompt[0]}.png")
#             image.save(image_path)
#             print(f"Saved {image_path}")
#     end_time = time.time()
#     iteration_time = end_time - start_time
#     print(f"Time for one iteration: {iteration_time} seconds")

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_id = "google/ddpm-ema-church-256"



# ddpm = DDPMPipeline.from_pretrained(model_id)
# print(ddpm.config)

# steps = 800
# noise_scheduler = DDPMScheduler()
# # Load the model and scheduler
# ddpm = DDPMPipeline.from_pretrained(model_id)
# from diffusers import UNet2DModel


# unet = UNet2DModel.from_pretrained(
#     model_id
# )


# seed1, seed2, seed3 = 1,2,3
# pipeline = DDPMPipeline.from_pretrained(model_id)

# print(pipeline.scheduler.config)
# distributed_state = PartialState()
# pipeline.to(distributed_state.device)

# print(pipeline.config)

# # Generate images
# batch_size = 50
# seed1, seed2, seed3, seed4 = 0,1,2,3

# # Save each image in the batch
# output_dir = "/home/tefimov/diffusers/src/NewSamplers/ChurchesEMA/1000samples_VanillaODE800/"
# os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists


# num_iterations = 5
# for iteration in range(num_iterations):
#     print(output_dir)
#     print("It is for 800")
#     start_time = time.time()
#     seed1, seed2, seed3, seed4 = seed1+4,seed2+4,seed3+4, seed4+4
#     with distributed_state.split_between_processes([seed1, seed2, seed3, seed4]) as prompt:
#         print("prompt is", prompt[0])
#         images = pipeline(batch_size=batch_size,generator=torch.manual_seed(prompt[0]), num_inference_steps=steps).images
#         for i, image in enumerate(images):
#             image_number = (iteration+0) * batch_size * 5 + i
#             image_path = os.path.join(output_dir, f"image_{image_number}_seed_{prompt[0]}.png")
#             image.save(image_path)
#             print(f"Saved {image_path}")
#     end_time = time.time()
#     iteration_time = end_time - start_time
#     print(f"Time for one iteration: {iteration_time} seconds")

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_id = "google/ddpm-ema-church-256"



# ddpm = DDPMPipeline.from_pretrained(model_id)
# print(ddpm.config)

# steps = 900
# noise_scheduler = DDPMScheduler()
# # Load the model and scheduler
# ddpm = DDPMPipeline.from_pretrained(model_id)
# from diffusers import UNet2DModel


# unet = UNet2DModel.from_pretrained(
#     model_id
# )


# seed1, seed2, seed3 = 1,2,3
# pipeline = DDPMPipeline.from_pretrained(model_id)

# print(pipeline.scheduler.config)
# distributed_state = PartialState()
# pipeline.to(distributed_state.device)

# print(pipeline.config)

# # Generate images
# batch_size = 50
# seed1, seed2, seed3, seed4 = 0,1,2,3

# # Save each image in the batch
# output_dir = "/home/tefimov/diffusers/src/NewSamplers/ChurchesEMA/1000samples_VanillaODE900/"
# os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists


# num_iterations = 5
# for iteration in range(num_iterations):
#     print("It is for 800")
#     start_time = time.time()
#     seed1, seed2, seed3, seed4 = seed1+4,seed2+4,seed3+4, seed4+4
#     with distributed_state.split_between_processes([seed1, seed2, seed3, seed4]) as prompt:
#         print("prompt is", prompt[0])
#         images = pipeline(batch_size=batch_size,generator=torch.manual_seed(prompt[0]), num_inference_steps=steps).images
#         for i, image in enumerate(images):
#             image_number = (iteration+0) * batch_size * 5 + i
#             image_path = os.path.join(output_dir, f"image_{image_number}_seed_{prompt[0]}.png")
#             image.save(image_path)
#             print(f"Saved {image_path}")
#     end_time = time.time()
#     iteration_time = end_time - start_time
#     print(f"Time for one iteration: {iteration_time} seconds")

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_id = "google/ddpm-ema-church-256"



# ddpm = DDPMPipeline.from_pretrained(model_id)
# print(ddpm.config)

# steps = 1000
# noise_scheduler = DDPMScheduler()
# # Load the model and scheduler
# ddpm = DDPMPipeline.from_pretrained(model_id)
# from diffusers import UNet2DModel


# unet = UNet2DModel.from_pretrained(
#     model_id
# )


# seed1, seed2, seed3 = 1,2,3
# pipeline = DDPMPipeline.from_pretrained(model_id)

# print(pipeline.scheduler.config)
# distributed_state = PartialState()
# pipeline.to(distributed_state.device)

# print(pipeline.config)

# # Generate images
# batch_size = 50
# seed1, seed2, seed3, seed4 = 0,1,2,3

# # Save each image in the batch
# output_dir = "/home/tefimov/diffusers/src/NewSamplers/ChurchesEMA/1000samples_VanillaODE1000/"
# os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists


# num_iterations = 5
# for iteration in range(num_iterations):
#     print("It is for 800")
#     start_time = time.time()
#     seed1, seed2, seed3, seed4 = seed1+4,seed2+4,seed3+4, seed4+4
#     with distributed_state.split_between_processes([seed1, seed2, seed3, seed4]) as prompt:
#         print("prompt is", prompt[0])
#         images = pipeline(batch_size=batch_size,generator=torch.manual_seed(prompt[0]), num_inference_steps=steps).images
#         for i, image in enumerate(images):
#             image_number = (iteration+0) * batch_size * 5 + i
#             image_path = os.path.join(output_dir, f"image_{image_number}_seed_{prompt[0]}.png")
#             image.save(image_path)
#             print(f"Saved {image_path}")
#     end_time = time.time()
#     iteration_time = end_time - start_time
#     print(f"Time for one iteration: {iteration_time} seconds")


# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_id = "google/ddpm-cifar10-32"



# ddpm = DDPMPipeline.from_pretrained(model_id)
# print(ddpm.config)

# steps = 1000
# noise_scheduler = DDPMScheduler()
# # Load the model and scheduler
# ddpm = DDPMPipeline.from_pretrained(model_id)
# from diffusers import UNet2DModel


# unet = UNet2DModel.from_pretrained(
#     model_id
# )


# seed1, seed2, seed3 = 1,2,3
# pipeline = DDPMPipeline.from_pretrained(model_id)

# print(pipeline.scheduler.config)
# distributed_state = PartialState()
# pipeline.to(distributed_state.device)

# print(pipeline.config)

# # Generate images
# batch_size = 50
# seed1, seed2, seed3, seed4 = 0,1,2,3

# # Save each image in the batch
# output_dir = "/home/tefimov/diffusers/src/NewSamplers/Cifar/2000samples_VanillaODE1000/"
# os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists


# num_iterations = 10
# for iteration in range(num_iterations):
#     print("It is for 800")
#     start_time = time.time()
#     seed1, seed2, seed, seed4 = seed1+4,seed2+4,seed3+4, seed4+4
#     with distributed_state.split_between_processes([seed1, seed2, seed3, seed4]) as prompt:
#         print("prompt is", prompt[0])
#         images = pipeline(batch_size=batch_size,generator=torch.manual_seed(prompt[0]), num_inference_steps=steps).images
#         for i, image in enumerate(images):
#             image_number = (iteration+0) * batch_size * 5 + i
#             image_path = os.path.join(output_dir, f"image_{image_number}_seed_{prompt[0]}.png")
#             image.save(image_path)
#             print(f"Saved {image_path}")
#     end_time = time.time()
#     iteration_time = end_time - start_time
#     print(f"Time for one iteration: {iteration_time} seconds")