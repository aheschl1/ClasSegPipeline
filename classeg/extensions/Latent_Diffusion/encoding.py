import PIL.Image
import torch
import numpy as np
import PIL
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch.nn as nn
from classeg.extensions.Latent_Diffusion.model.autoencoder.autoencoder import VQModel
from classeg.extensions.Latent_Diffusion.forward_diffusers.diffusers import LinearDiffuser
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

    if img.shape[0] == 1:
      img = torch.stack([img[0],img[0],img[0]])
    img = torch.unsqueeze(img, 0)
    return img

def encode_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  # x = x.to("cuda")
  z, _, [_, _, indices] = model.encode(x)
  # print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape}")
  return z

def decode_with_vqgan(z, model, ts=None):
  if ts is not None:
    _, _, z, _, _ = diffuser(z, z, t=torch.tensor(ts))
  xrec = model.decode(z)
  return xrec

def load_vqgan(config, ckpt_path=None):
  model = VQModel(**config.model.params)
  sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
  missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def encoding_pipeline(img, size=1024, ts=None):
  x_vqgan = preprocess(img, target_image_size=size)
  x_vqgan = x_vqgan
  # print(f"input is of size: {x_vqgan.shape}")
  z = encode_with_vqgan(preprocess_vqgan(x_vqgan), model32x32)
  return z

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


auto_path = "/home/mauricio.murillo/Diffusion_ClasSeg/Autoencoders/vq-f8-n256"
model32x32 = load_vqgan(load_config(f'{auto_path}/config.yaml'), ckpt_path=f'{auto_path}/model.ckpt')

linear_start: 0.0015
linear_end: 0.0195
diffuser = LinearDiffuser(1000, 0.0001, 0.02)
# diffuser = LinearDiffuser(1000)

generated = PIL.Image.open("/home/mauricio.murillo/Documents/Datasets/DiffusionRoot/DiffusionResults/Dataset_420/fold_0/ayayay/inference/folde_lat/Masks/x0_1.jpg")
gen_t = encoding_pipeline(generated, 256)
print(gen_t.shape)  #Should be 1,4,32,32

print("Cosine Similarity")
cos = nn.CosineSimilarity(dim=1)

import glob

from tqdm import tqdm
def cosine_sim(cmp):
  num = 0
  sim_c = 0
  with torch.no_grad():
    for file in tqdm(sorted(glob.glob("/home/mauricio.murillo/SegmentationPipeline/data/oprediction/1/*/Mask.png"))):
      mas_r = PIL.Image.open(file)
      mas_enc = encoding_pipeline(mas_r, 256)
      sim = cos(cmp, mas_enc) #shape is 1, 32, 32
      sim_c += sim.mean().detach()
      num += 1
      if num >= 100: break
    print(sim_c/num)

cosine_sim(gen_t)


real = PIL.Image.open("/home/mauricio.murillo/SegmentationPipeline/data/oprediction/1/113/Mask.png")
real_t = encoding_pipeline(real, 256)
print(real_t.shape)  #Should be 1,4,32,32
cosine_sim(real_t)
