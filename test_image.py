import torch
import numpy as np
import shutil
from tqdm.autonotebook import tqdm
import os
import cv2
from model import TwinLite as net

from inference_utils import Run_datafusion, Run_depth, Run_rgb

# Determinazione del dispositivo (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
# Caricamento del modello
aviable_modality = ['rgb', 'depth', 'datafusion']
modality = 'datafusion'
if modality not in aviable_modality:
    raise ValueError('Modality not supported')

if modality == 'datafusion':
    model = net.TwinLiteNet_RGBD_Adaptive()
    rgb_folder = 'color'
    depth_folder = 'depth'
elif modality == 'rgb' or modality == 'depth':
    model = net.TwinLiteNet()
    if modality == 'rgb':
        rgb_folder = 'color'
    else:
        depth_folder = 'depth'

model_path = "pretrained//rgbd_adaptive_best.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()



# Elaborazione delle immagini
image_root = "c:\\Users\\Nocerino Antonio\\Desktop\\Dataset_tesi_aligned\\test"
output_root = f"c:\\Users\\Nocerino Antonio\\Desktop\\Dataset_tesi_aligned\\test\\pred_{modality}"
if os.path.exists(output_root):
    shutil.rmtree(output_root)
os.makedirs(output_root)

if modality == 'datafusion':
    rgb_path = os.path.join(image_root, rgb_folder)
    depth_path = os.path.join(image_root, depth_folder)
    rgb_files = os.listdir(rgb_path)
    depth_files = os.listdir(depth_path)
    rgb_files.sort()
    depth_files.sort()

    for i in tqdm(range(len(rgb_files))):
        rgb = cv2.imread(os.path.join(rgb_path, rgb_files[i]))
        depth = cv2.imread(os.path.join(depth_path, depth_files[i]), cv2.IMREAD_UNCHANGED)
        output = Run_datafusion(model, depth, rgb, device)
        cv2.imwrite(os.path.join(output_root, rgb_files[i]), output)
elif modality == 'rgb':
    rgb_path = os.path.join(image_root, rgb_folder)
    rgb_files = os.listdir(rgb_path)
    rgb_files.sort()

    for i in tqdm(range(len(rgb_files))):
        rgb = cv2.imread(os.path.join(rgb_path, rgb_files[i]))
        output = Run_rgb(model, rgb, device)
        cv2.imwrite(os.path.join(output_root, rgb_files[i]), output)
elif modality == 'depth':
    depth_path = os.path.join(image_root, depth_folder)
    depth_files = os.listdir(depth_path)
    depth_files.sort()

    for i in tqdm(range(len(depth_files))):
        depth = cv2.imread(os.path.join(depth_path, depth_files[i]), cv2.IMREAD_UNCHANGED)
        output = Run_depth(model, depth, device)
        cv2.imwrite(os.path.join(output_root, depth_files[i]), output)
print('Done')

