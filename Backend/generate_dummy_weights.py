import sys
import os
import yaml
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.genconvit_ed import GenConViTED
from model.genconvit_vae import GenConViTVAE

config_path = "model/config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

print("Generating GenConViTED dummy weights...")
ed = GenConViTED(config)
torch.save({'state_dict': ed.state_dict()}, "weight/genconvit_ed_inference.pth")

print("Generating GenConViTVAE dummy weights...")
vae = GenConViTVAE(config)
torch.save({'state_dict': vae.state_dict()}, "weight/genconvit_vae_inference.pth")

print("Dummy weights generated successfully!")
