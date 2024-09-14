import torch
import cv2
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import os
import random
import math

def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """change color hue, saturation, value"""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
def random_perspective(combination,  degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0), sensor_fusion = False, depth=False):
    """combination of img transform"""
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    #img, gray, line = combination
    if sensor_fusion:
        img, depth, gray = combination
    else:
        img, gray = combination
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            
            
            if sensor_fusion:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
                depth = cv2.warpPerspective(depth, M, dsize=(width, height), borderValue=0)
                
            elif depth:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=0)
                
            else:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
                
            gray = cv2.warpPerspective(gray, M, dsize=(width, height), borderValue=0)
            
        else:  # affine
           
            if sensor_fusion:
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
                depth = cv2.warpAffine(depth, M[:2], dsize=(width, height), borderValue=0)
            elif depth:
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=0)
            else:
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            gray = cv2.warpAffine(gray, M[:2], dsize=(width, height), borderValue=0)



    #combination = (img, gray, line)
    if sensor_fusion:
        combination = (img, depth, gray)
    else:
        combination = (img, gray)
    return combination

class MyDataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self,train_path = None, valid_path = None, test_path= None, rgb_folder_name = "color",\
                label_folder_name = "labels", depth_folder_name = "depth", transform=None,valid=False, test=False,\
                sensor_fusion = False, label = "BDD100K", width=640, height=360, with_agumentation = False, depth = False, deepscene = False):
        """
        Args:
            train_path (str): Path to the training dataset.
            valid_path (str): Path to the validation dataset.
            test_path (str): Path to the test dataset.
            rgb_folder_name (str): Folder name for RGB images.
            label_folder_name (str): Folder name for label images.
            depth_folder_name (str): Folder name for depth images.
            transform (callable, optional): Optional transform to be applied on a sample.
            valid (bool): Flag to indicate validation mode.
            test (bool): Flag to indicate test mode.
            sensor_fusion (bool): Flag to indicate if sensor fusion is used.
            label (str): Type of label processing to use.
        """
        self.width = width
        self.height = height
        self.transform = transform
        self.Tensor = transforms.ToTensor()
        self.valid=valid
        self.test=test
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.rgb_folder_name = rgb_folder_name
        self.depth_folder_name = depth_folder_name
        self.label_folder_name = label_folder_name
        self.sensor_fusion = sensor_fusion
        self.depth = depth
        self.label = label
        self.with_agumentation = with_agumentation
        self.deepscene = deepscene
        if test:
            if self.test_path is None:
                raise ValueError("Test path not provided.")
            self.root=os.path.join(self.test_path, self.rgb_folder_name)
            self.root_depth=os.path.join(self.test_path, self.depth_folder_name)
            self.names=os.listdir(self.root)
        elif valid:
            if self.valid_path is None:
                raise ValueError("Validation path not provided.")
            self.root=os.path.join(self.valid_path, self.rgb_folder_name)
            self.root_depth=os.path.join(self.valid_path, self.depth_folder_name)
            self.names=os.listdir(self.root)
        else:
            if self.train_path is None:
                raise ValueError("Train path not provided.")
            self.root=os.path.join(self.train_path, self.rgb_folder_name)
            self.root_depth=os.path.join(self.train_path, self.depth_folder_name)
            self.names=os.listdir(self.root)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        '''

        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        # '''

        if self.sensor_fusion:
            image_name = os.path.join(self.root, self.names[idx])
            image = cv2.imread(image_name)
            if self.deepscene:
                name_depth = self.names[idx].replace("_Clipped.jpg",".png")
                depth_name=os.path.join(self.root_depth,name_depth)
            else:
                depth_name=os.path.join(self.root_depth,self.names[idx])
            image_depth = cv2.imread(depth_name, cv2.IMREAD_UNCHANGED)
            if self.label == "ORFD":
                raise ValueError("Sensor Fusion with depth not implemented for ORFD")
            # Process label
            label_method_name = f"label_{self.label}"
            if hasattr(self, label_method_name):
                label1 = getattr(self, label_method_name)(image_name, image)
            else:
                raise ValueError(f"Label method {label_method_name} does not exist.")

            # Apply augmentations only to RGB image and label
            if self.with_agumentation and not (self.valid or self.test):
                if random.random() < 0.5:
                    combination = (image,image_depth, label1)
                    (image,image_depth, label1) = random_perspective(
                        combination=combination,
                        degrees=10,
                        translate=0.1,
                        scale=0.25,
                        shear=0.0,
                        sensor_fusion=True
                    )
                
                if random.random() < 0.5:
                    augment_hsv(image)
                
                if random.random() < 0.5:
                    image = np.fliplr(image)
                    label1 = np.fliplr(label1)
                    # Flip depth image horizontally to match RGB
                    image_depth = np.fliplr(image_depth)

            # Resize all images
            label1 = cv2.resize(label1, (self.width, self.height))
            image = cv2.resize(image, (self.width, self.height))
            image_depth = cv2.resize(image_depth, (self.width, self.height))

            # Process label
            _, seg_b1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY_INV)
            _, seg1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY)
            seg1 = self.Tensor(seg1)
            seg_b1 = self.Tensor(seg_b1)
            seg_da = torch.stack((seg_b1[0], seg1[0]), 0)

            # Process RGB image
            image = image[:, :, ::-1].transpose(2, 0, 1)
            # Normalize imageù
            image = image.astype(np.float32)
            image = image / 255.0
            image = np.ascontiguousarray(image)
            if self.deepscene:
                image_depth = image_depth[:, :, ::-1].transpose(2, 0, 1)
                # Normalize image_depthù
                image_depth = image_depth.astype(np.float32)
                image_depth = image_depth / 255.0
                image_depth = np.ascontiguousarray(image_depth)
            else:
                # Process depth image
                image_depth = np.expand_dims(image_depth, axis=0)
                image_depth = np.repeat(image_depth, 3, axis=0)
                image_depth = image_depth.astype(np.float32)
                # Normalize image
                image_depth = image_depth / 65535.0
                image_depth = np.ascontiguousarray(image_depth)

            return image_name, depth_name, torch.from_numpy(image), torch.from_numpy(image_depth), seg_da
        elif self.depth:

            if self.deepscene:
                name_depth = self.names[idx].replace("_Clipped.jpg",".png")
                depth_name=os.path.join(self.root_depth,name_depth)
            else:
                depth_name=os.path.join(self.root_depth,self.names[idx])
            image = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
            if self.label == "ORFD":
                raise ValueError("Depth not implemented for ORFD")
        else:
            image_name=os.path.join(self.root,self.names[idx])
            image = cv2.imread(image_name)
        
        label_method_name = f"label_{self.label}"
        if hasattr(self, label_method_name):
            label1 = getattr(self, label_method_name)(image_name, image)
        else:
            raise ValueError(f"Label method {label_method_name} does not exist.")

        # Apply augmentations only to RGB image and label
        if self.with_agumentation and not (self.valid or self.test):
            
            if random.random() < 0.5:
                combination = (image, label1)
                (image, label1) = random_perspective(
                    combination=combination,
                    degrees=10,
                    translate=0.1,
                    scale=0.25,
                    shear=0.0,
                    sensor_fusion=False,
                    depth=self.depth
                )
            if not self.depth:
                if random.random() < 0.5:
                    augment_hsv(image)
            
            if random.random() < 0.5:
                image = np.fliplr(image)
                label1 = np.fliplr(label1)
        label1 = cv2.resize(label1, (self.width, self.height))
        image = cv2.resize(image, (self.width, self.height))

        _,seg_b1 = cv2.threshold(label1,1,255,cv2.THRESH_BINARY_INV)
        _,seg1 = cv2.threshold(label1,1,255,cv2.THRESH_BINARY)

        seg1 = self.Tensor(seg1)
        seg_b1 = self.Tensor(seg_b1)
        seg_da = torch.stack((seg_b1[0], seg1[0]),0)
        if not self.depth:
            image = image[:, :, ::-1].transpose(2, 0, 1)
            #normalize image
            image = image.astype(np.float32)
            image = image / 255.0
            image = np.ascontiguousarray(image)# 8bit, 0-255
        elif self.deepscene:
            image = image[:, :, ::-1].transpose(2, 0, 1)
            #normalize image
            image = image.astype(np.float32)
            image = image / 255.0
            image = np.ascontiguousarray(image)
        else:
            #convert image to float32
            image = np.expand_dims(image, axis=0)#
            image = np.repeat(image, 3, axis=0)
            image = image.astype(np.float32)
            #normalize image
            image = image / 65535.0
            image = np.ascontiguousarray(image)



       
        return image_name,torch.from_numpy(image),seg_da
    
    

    def label_ORFD(self, image_name, image):
        if self.depth or self.sensor_fusion:
            #Not implemented  for ORFD exception
            raise ValueError("Depth not implemented for ORFD")
        
        else:
            label_path= image_name.replace(self.rgb_folder_name,self.label_folder_name)
            label_path = label_path.split(".")[0]+"_fillcolor.png"
            oriHeight, oriWidth, _ = image.shape
            label_image = cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2RGB)
                
            label1 = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
            label1[label_image[:,:,2] > 200] = 255
            return label1
    
    def label_BDD100K(self, image_name, image):
        if self.depth:
            label_path= image_name.replace(self.depth_folder_name,self.label_folder_name)
        else:
            label_path= image_name.replace(self.rgb_folder_name,self.label_folder_name)
        label1 = cv2.imread(label_path, 0)
        return label1
    
    def label_deepscene(self, image_name, image):
        if self.depth:
            label_path= image_name.replace(self.depth_folder_name,self.label_folder_name)
            #add mask to the label path
            label_path = label_path.split(".")[0]+"_mask.png"
        else:
            label_path= image_name.replace(self.rgb_folder_name,self.label_folder_name)
            label_path = label_path.replace("_Clipped.jpg","_mask.png")
        oriHeight, oriWidth, _ = image.shape
        #verifica se esiiste il file
        if not os.path.exists(label_path):
            label_path = label_path.replace("_mask.png","_Clipped.png")
        # label_image = cv2.imread(label_path)

        # label1 = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
        # label1[label_image[:,:,2] > 200] = 1
        label1 = cv2.imread(label_path, 0)
        return label1


