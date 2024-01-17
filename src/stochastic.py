from dataclasses import dataclass
from datasets import load_dataset
import time
from diffusers import UNet2DConditionModel

@dataclass
class TrainingConfig:
    image_size: int = 32 # the generated image resolution
    train_batch_size: int = 256
    eval_batch_size: int = 16  # how many images to sample during evaluation
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    norm_num_groups = 8
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 0
    save_image_epochs: int = 5
    save_model_epochs: int = 5
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str = "mnist-model-cond"  # the model name locally and on the HF Hub
    push_to_hub: bool = False # whether to upload the saved model to the HF Hub
    hub_model_id: str = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    seed: int = 0

# Create an instance of the config
config = TrainingConfig()

print(config)


dataset = load_dataset('mnist', split='train')

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for i, image in enumerate(dataset[:4]["image"]):
    axs[i].imshow(image)
    axs[i].set_axis_off()
fig.show()

from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def transform(examples):
    images = [preprocess(image) for image in examples["image"]]
    labels = examples["label"]
    return {"images": images, "labels": labels}


dataset.set_transform(transform)

import torch

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

# from datasets import load_dataset

# config.dataset_name = "huggan/smithsonian_butterflies_subset"
# dataset = load_dataset(config.dataset_name, split="train")

# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(1, 4, figsize=(16, 4))
# for i, image in enumerate(dataset[:4]["image"]):
#     axs[i].imshow(image)
#     axs[i].set_axis_off()
# fig.show()

# from torchvision import transforms

# preprocess = transforms.Compose(
#     [
#         transforms.Resize((config.image_size, config.image_size)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5]),
#     ]
# )

# def transform(examples):
#     images = [preprocess(image.convert("RGB")) for image in examples["image"]]
#     labels = examples["sim_score"]
#     return {"images": images, "labels": labels}


# dataset.set_transform(transform)

# import torch

# train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)


from diffusers import UNet2DModel

# # Assuming config is defined as in your previous message
# model = UNet2DModel(
#     sample_size=config.image_size,  # the target image resolution
#     in_channels=1,  # the number of input channels, 3 for RGB images
#     out_channels=1,  # the number of output channels
#     layers_per_block=2,  # how many ResNet layers to use per UNet block
#     block_out_channels=(32*4, 64*4, 64*4, 64*4),  # output channels for each UNet block
#     down_block_types=(
#         "DownBlock2D",  # a regular ResNet downsampling block
#         "DownBlock2D",
#         "AttnDownBlock2D",  # a ResNet block with spatial self-attention
#         "DownBlock2D",
#     ),
#     up_block_types=(
#         "UpBlock2D",  # a regular ResNet upsampling block
#         "AttnUpBlock2D",  # a ResNet block with spatial self-attention
#         "UpBlock2D",
#         "UpBlock2D",
#     ),
# )

model = UNet2DConditionModel(
    in_channels = 1,
    out_channels=1,
    layers_per_block=2, 
    block_out_channels=(64, 64, 64, 64),
    cross_attention_dim=1
      # output channels for each UNet block
)

# Sample image from your dataset for testing
# Assuming dataset is defined and has the required structure
sample_image = dataset[0]["images"].unsqueeze(0)
print("Input shape:", sample_image.shape)
# Expected output: Input shape: torch.Size([1, 3, 128, 128


import torch
from PIL import Image
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

# Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])

import torch.nn.functional as F

# noise_pred = model(noisy_image, timesteps).sample
# loss = F.mse_loss(noise_pred, noise)


from diffusers.optimization import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMPipeline
from diffusers.utils import make_image_grid
import os


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
import os


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs")
    )

    # model = torch.nn.DataParallel(model)
    # model.to(device)
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )


    # model = torch.nn.DataParallel(model)
    # model.to(device)

    first_param = next(model.parameters()).device
    if isinstance(model, torch.nn.DataParallel):
        print("Model is on multiple GPUs:", model.device_ids)
    else:
        print("Model is on single device:", next(model.parameters()).device)
    # Assuming 'accelerator' is your Accelerator object
    print("Accel Devices:", accelerator.device_placement)

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        print(epoch)
        start_time = time.time() 
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            
            clean_images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            # print("Labels size is", labels.size())
            # print("Labels are")
            labels = labels.unsqueeze(1)
            labels= labels.repeat(1, 1)
            #labels = labels.expand(-1, -1, 1280)
            labels = labels.unsqueeze(1)
            labels = labels.to(torch.float32)
            labels = labels.to(device)
            #print(labels.size())
            
            
            #print(clean_images.size())
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=device)
            
            
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            #print(noisy_images.size())

            with accelerator.accumulate(model):
                # Predict the noise residual
                # print(noisy_images.size(), timesteps.size(), labels.size())
                #print(model)
                
                noise_pred = model(noisy_images, timesteps, labels, return_dict=False)[0]
                #print(noise_pred.size())
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        progress_bar.update(1)
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
        global_step += 1

        if accelerator.is_main_process:

            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler, cond=labels)
            print("Evaluation labels are labels", labels)
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)
        end_time = time.time() 
        epoch_duration = end_time - start_time  # Calculate the duration of the epoch
        print(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds")

print("Hi i am here")

from accelerate import notebook_launcher


device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.nn.DataParallel(model)
model = model.to(device)
print(device)


train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)


print("Hi i am here")

import glob

sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
Image.open(sample_images[-1])