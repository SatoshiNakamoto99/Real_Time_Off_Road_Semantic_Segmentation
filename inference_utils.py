
import cv2
import numpy as np
import torch

def frame16b1c_to_8b3c(frame):

    frame_8bit3channel = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

    frame_8bit3channel[:,:,0] = frame & 0x00FF # first 8 bit
    frame_8bit3channel[:,:,1] = (frame & 0xFF00) >> 8 # next 8 bit
    frame_8bit3channel[:,:,2] = 0 # all 0
    return frame_8bit3channel

def frame8b3c_to_16b1c(frame):
    frame_16bit1channel = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint16)
    #convert i chanbnel to 16 bit
    first_channel = frame[:,:,0].astype(np.uint16)
    second_channel = frame[:,:,1].astype(np.uint16)
    frame_16bit1channel = first_channel + (second_channel << 8)
    


    
    return frame_16bit1channel
def Run_rgb(model, img_rgb, device):
    original_shape = img_rgb.shape
    img = cv2.resize(img_rgb, (640, 360))
    img_rs = img.copy()

    img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # Add a batch dimension
    img = img.to(device).float() / 255.0

    with torch.no_grad():
        img_out = model(img)

    _, da_predict = torch.max(img_out, 1)

    DA = da_predict.byte().cpu().numpy()[0] * 255  # Convert to numpy array
    img_rs[DA > 100] = [0, 255, 0]  # Apply a mask to the original image

    img_rs = cv2.resize(img_rs, (original_shape[1], original_shape[0]))  # Resize back to original shape
    return img_rs

def Run_depth(model, img_depth, device):
    # if img is not 16 bit 1 channel
    if img_depth.dtype != np.uint16:
        img_depth = frame8b3c_to_16b1c(img_depth)
    original_shape = img_depth.shape
    img = cv2.resize(img_depth, (640, 360))
    img_rs = img.copy()
    
    img = np.expand_dims(img, axis=0)#
    img = np.repeat(img, 3, axis=0)
    img = img.astype(np.float32)
    #normalize img
    img = img / 65535.0
    img = np.ascontiguousarray(img)
    # Add a batch dimension
    img = torch.from_numpy(img)
    img = img.to(device)
    img = torch.unsqueeze(img, 0)  # Add a batch dimension

    with torch.no_grad():
        img_out = model(img)

    _, da_predict = torch.max(img_out, 1)
    DA = da_predict.byte().cpu().numpy()[0] * 255  # Convert to numpy array
    img_rs = frame16b1c_to_8b3c(img_rs)
    img_rs[DA > 100] = [0,255,0]  # Apply a mask to the original image

    img_rs = cv2.resize(img_rs, (original_shape[1], original_shape[0]))  # Resize back to original shape
    return img_rs

def Run_datafusion(model, img_depth, img_rgb, device):
    original_shape = img_rgb.shape  # Assuming img_rgb and img_depth have the same shape
    img_rgb = cv2.resize(img_rgb, (640, 360))
    if img_depth.dtype != np.uint16:
        img_depth = frame8b3c_to_16b1c(img_depth)
    img_depth = cv2.resize(img_depth, (640, 360))
    img_rs = img_rgb.copy()

    img_rgb = img_rgb[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB
    img_rgb = np.ascontiguousarray(img_rgb)
    img_rgb = torch.from_numpy(img_rgb)
    img_rgb = torch.unsqueeze(img_rgb, 0)  # Add a batch dimension
    img_rgb = img_rgb.to(device).float() / 255.0

    img_depth = np.expand_dims(img_depth, axis=0)
    img_depth = np.repeat(img_depth, 3, axis=0)
    img_depth = img_depth.astype(np.float32)
    img_depth = img_depth / 65535.0
    img_depth = np.ascontiguousarray(img_depth)
    img_depth = torch.from_numpy(img_depth)
    img_depth = img_depth.to(device)
    img_depth = torch.unsqueeze(img_depth, 0)  # Add a batch dimension

    with torch.no_grad():
        img_out = model(img_rgb, img_depth)

    _, da_predict = torch.max(img_out, 1)

    DA = da_predict.byte().cpu().numpy()[0] * 255  # Convert to numpy array
    img_rs[DA > 100] = [0, 255, 0]  # Apply a mask to the original image

    img_rs = cv2.resize(img_rs, (original_shape[1], original_shape[0]))  # Resize back to original shape
    return img_rs

