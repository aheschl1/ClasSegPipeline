import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
import cv2
from classeg.utils.utils import get_dataloaders_from_fold
import albumentations as A
from classeg.extensions.unstable_diffusion.model.unstable_diffusion import UnstableDiffusion
import json
import matplotlib.pyplot as plt
import tqdm
import os


# Define your model class
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define your model architecture here

    def forward(self, x):
        # Implement the forward pass of your model here
        return x

config = None
results_root= f"/home/andrewheschl/Documents/Dataset/ClassPipeline/results/embed_one_encoder"
with open(f"{results_root}/config.json", "r") as f:
    config = json.load(f)

config['processes'] = 16
# config["batch_size"] //= 8

print(config)
# Define your method to get the model
def get_model():
    model = UnstableDiffusion(
        **config["model_args"]
    )
    weights = torch.load(f"{results_root}/best.pth")['weights']
    model.load_state_dict(weights)
    
    model.encoder_layers = None
    model.middle_layer = None
    model.im_decoder_layers = None
    model.seg_decoder_layers = None

    del model.middle_layer
    del model.encoder_layers
    del model.im_decoder_layers
    del model.seg_decoder_layers

    return model.to("cuda")

resize_image = A.Resize(*config.get("target_size", [512, 512]), interpolation=cv2.INTER_CUBIC)
resize_mask = A.Resize(*config.get("target_size", [512, 512]), interpolation=cv2.INTER_NEAREST)

def my_resize(image=None, mask=None, **kwargs):
    if mask is not None:
        return resize_mask(image=mask)["image"]
    if image is not None:
        if image.max() > 1:
            image -= image.min()
            image /= image.max()
        return resize_image(image=image)["image"]

def norm(image=None, mask=None, **kwargs):
    if mask is not None:
        return mask / 255
    if image is not None:
        return image / image.max()

train_transforms = A.Compose(
    [
        A.RandomCrop(width=512, height=512, p=1),
        A.Lambda(image=my_resize, mask=my_resize, p=1)
    ],
    is_check_shapes=False
)
_, dataloader = get_dataloaders_from_fold("Dataset_large_421", 0, train_transforms, train_transforms, True, config=config)
# Set up the model and SummaryWriter
model = get_model()
name = "recon_random_im2"
writer = SummaryWriter(log_dir=f"./{name}")

# Pass the images through the model and send the embeddings to SummaryWriter
model.eval()
embeddings_total = []
all_images = []
# embeddings = (torch.randn(16, 256, 8, 8, device='cuda')+0)/10
i = 0
og_shape = None
with torch.no_grad():
    loss = 0
    total_samples = 0
    for images, *_ in tqdm.tqdm(dataloader):
        images = images.to("cuda")
        embeddings, recon = model.embbed_bonus(images, recon_im=True, return_projected=False)
        loss += torch.nn.functional.mse_loss(recon, images)*images.shape[0]
        total_samples += images.shape[0]

        recon -= recon.min()
        recon /= recon.max()
        all_images.append(torch.nn.functional.interpolate(recon.cpu(), size=(16, 16), mode="bilinear", align_corners=False))
        og_shape = embeddings.shape[1:]
        # compare images[0] with recon[0] in tensorboard

        writer.add_images("images", images, global_step=i)
        writer.add_images("recon", recon, global_step=i)
        i+=1

        # recon = model.recon_bonus_embed(embeddings)
        # for im in recon.cpu():
        #     writer.add_image("recon", im, global_step=i)
        #     i += 1
        embeddings_total.append(embeddings.flatten(start_dim=1).cpu())
print(loss/total_samples)
torch.save(torch.cat(embeddings_total, dim=0).unflatten(1, og_shape), f"{name}.pt")
writer.add_embedding(torch.cat(embeddings_total, dim=0), global_step=0, label_img=torch.cat(all_images, dim=0))

# Close the SummaryWriter
writer.close()