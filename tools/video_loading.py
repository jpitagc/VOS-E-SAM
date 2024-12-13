
import numpy as np 
import cv2
import os

def load_all_images_davis(loader,video_info):
    name, frames, objects = video_info
    all_images, all_frames = [],[]
    for i in range(0,frames):
        F_last,M_last = loader.load_single_image(name,i)
        all_images.append((np.array(F_last[:,0]).transpose(1, 2, 0)* 255.).astype(np.uint8))
        all_frames.append(np.array(M_last[1:objects+1,0]).astype(np.uint8))
    return all_images,all_frames

def load_images_from_folder(path,image_files):
    images = []
    for file in image_files:
        img = cv2.imread(os.path.join(path,file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return images