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
import tqdm 

# Define your model class
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define your model architecture here

    def forward(self, x):
        # Implement the forward pass of your model here
        return x

config = None
results_root= "/work/vision_lab/andrew.heschl/Documents/Dataset/ClassificationPipeline/results/Dataset_large_421/fold_0/embed_one_encoder/"
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
dataloader, _ = get_dataloaders_from_fold("Dataset_large_421", 0, train_transforms, train_transforms, True, config=config)
# Set up the model and SummaryWriter
model = get_model()
name = "embedding_im_tr"
writer = SummaryWriter(log_dir=f"./{name}")

# Pass the images through the model and send the embeddings to SummaryWriter
model.eval()
embeddings_total = []
all_images = []
with torch.no_grad():
    for images, *_ in tqdm.tqdm(dataloader):
        all_images.append(torch.nn.functional.interpolate(images, size=(32, 32), mode="bilinear", align_corners=False))
        images = images.to("cuda")
        embeddings, _ = model.embbed_bonus(images, recon_im=False)
        embeddings_total.append(embeddings.cpu())

writer.add_embedding(torch.cat(embeddings_total, dim=0), global_step=0, label_img=torch.cat(all_images, dim=0))

# Close the SummaryWriter
writer.close()