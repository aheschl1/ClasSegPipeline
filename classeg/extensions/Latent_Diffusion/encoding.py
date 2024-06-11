import PIL.Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.transforms as T
from classeg.extensions.Latent_Diffusion.model.autoencoder.autoencoder import VQModel
import yaml

from omegaconf import OmegaConf

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x


def preprocess(img, target_image_size=1024):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = T.ToTensor()(img)

    print(img.shape)
    if img.shape[0] == 1:
      img = torch.stack([img[0],img[0],img[0]])
    img = torch.unsqueeze(img, 0)
    return img

def reconstruct_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  x = x.to("cuda")
  z, _, [_, _, indices] = model.encode(x)
  print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape}")
  xrec = model.decode(z)
  return xrec

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
  model = VQModel(**config.model.params)
  sd = torch.load(ckpt_path)["state_dict"]
  missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval().cuda()

config_path = "/home/mauricio.murillo/Diffusion_ClasSeg/Autoencoders/vq-f8-n256/config.yaml"
model32x32 = load_vqgan(load_config(config_path), ckpt_path="/home/mauricio.murillo/Diffusion_ClasSeg/Autoencoders/vq-f8-n256/model.ckpt")

def reconstruction_pipeline(image_path, size=1024):
  img = PIL.Image.open(image_path)
  x_vqgan = preprocess(img, target_image_size=size)
  x_vqgan = x_vqgan
  
  print(f"input is of size: {x_vqgan.shape}")
  x0 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model32x32)
  return x0


def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

img_r = PIL.Image.open("/home/mauricio.murillo/SegmentationPipeline/data/oprediction/1/0/Image.jpg")
img_r.save("real_img.jpg")

mas_r = PIL.Image.open("/home/mauricio.murillo/SegmentationPipeline/data/oprediction/1/0/Mask.png")
mas_r.save("real_mas.jpg")

img = reconstruction_pipeline("/home/mauricio.murillo/SegmentationPipeline/data/oprediction/1/0/Image.jpg", size=1024)
img = custom_to_pil(img[0])
img.save("rec_img.jpg")

mas = reconstruction_pipeline("/home/mauricio.murillo/SegmentationPipeline/data/oprediction/1/0/Mask.png", size=1024)
mas = custom_to_pil(mas[0])
mas.save("rec_mas.jpg")



