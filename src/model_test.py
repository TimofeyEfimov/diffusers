import torch
from diffusers import UNet2DModel

# Create an instance of the model architecture
model = UNet2DModel(
    sample_size=32,  # Specify the image size
    in_channels=6,  # Specify the number of input channels
    out_channels=3,  # Specify the number of output channels
    freq_shift=1,
    flip_sin_to_cos=False,
    layers_per_block=2,  # Specify the number of ResNet layers per UNet block
    block_out_channels=(128, 256, 256, 256),  # Specify output channels for each UNet block
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),  # Specify down block types
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),  # Specify up block types
)

# Load the model's state dictionary from the saved file
model.load_state_dict(torch.load("/home/tefimov/diffusers/src/weights/modelV.pth"), strict=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.nn.DataParallel(model)

model = model.to(device)
# Set the model to evaluation mode
model.eval()

timesteps = torch.randint(
                0, 1000, (16,), device="cuda",
                dtype=torch.int64
            )


# Generate a sample 

noise = torch.randn(16, 6, 32, 32)

timesteps = timesteps.to(device)
noise = noise.to(device)

sample = model(noise, timesteps).sample


print(sample.size())
print(model.device)
