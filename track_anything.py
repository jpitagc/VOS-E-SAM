import PIL
from tqdm import tqdm

from tools.interact_tools import SamControler
from tracker.base_tracker import BaseTracker
from inpainter.base_inpainter import BaseInpainter
import numpy as np
import argparse
import torch
from torchvision import transforms
from tracker.util.range_transform import im_normalization


class TrackingAnything():
    def __init__(self, sam_checkpoint, xmem_checkpoint, e2fgvi_checkpoint, args, save_inner_masks_folder):
        self.args = args
        self.sam_checkpoint = sam_checkpoint
        self.xmem_checkpoint = xmem_checkpoint
        self.e2fgvi_checkpoint = e2fgvi_checkpoint
        current_device = 'cuda:0' # Is CPU only available fixed inside each controler 
        self.device = current_device
        self.samcontroler = SamControler(self.sam_checkpoint, 'vit_h', current_device)
        self.xmem = BaseTracker(self.xmem_checkpoint, device=current_device,\
            sam_model=self.samcontroler if self.args['use_refinement'] else None,\
            sam_mode=self.args['refinement_mode'] if self.args['use_refinement'] else None,\
            save_inner_masks_folder = save_inner_masks_folder,\
            points_convertion = self.args['addArgs1'],\
            optimized= self.args['optimized'])
        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])
        #self.baseinpainter = BaseInpainter(self.e2fgvi_checkpoint,current_device) 

    # def inference_step(self, first_flag: bool, interact_flag: bool, image: np.ndarray, 
    #                    same_image_flag: bool, points:np.ndarray, labels: np.ndarray, logits: np.ndarray=None, multimask=True):
    #     if first_flag:
    #         mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
    #         return mask, logit, painted_image
        
    #     if interact_flag:
    #         mask, logit, painted_image = self.samcontroler.interact_loop(image, same_image_flag, points, labels, logits, multimask)
    #         return mask, logit, painted_image
        
    #     mask, logit, painted_image = self.xmem.track(image, logit)
    #     return mask, logit, painted_image
    
    def first_frame_click(self, image: np.ndarray, points:np.ndarray, labels: np.ndarray, multimask=True):
        mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
        return mask, logit, painted_image
    
    # def interact(self, image: np.ndarray, same_image_flag: bool, points:np.ndarray, labels: np.ndarray, logits: np.ndarray=None, multimask=True):
    #     mask, logit, painted_image = self.samcontroler.interact_loop(image, same_image_flag, points, labels, logits, multimask)
    #     return mask, logit, painted_image

    def generator(self, images: list, template_mask:np.ndarray):
        # All loaded images as list of np.arrays. Size of original video 
        # Initial mask as np.array. 0 for background and 1,2,3... for each mask.
        # Example -> Imagenes 300 (540, 960, 3) || (540, 960) [0 1 2 3]
        #print('Imagenes',len(images),images[0].shape,'||',template_mask.shape, np.unique(template_mask))
        masks = []
        logits = []
        painted_images = []
        scores = []
        #tranformed_images = [self.im_transform(image) for image in images]
        #tensor_images = torch.stack(tranformed_images, dim=0).to(self.device)
        for i in tqdm(range(len(images)), desc="Tracking image"): #, disable=True
            if i ==0:           
                mask, logit, painted_image, score = self.xmem.track(images[i], template_mask)
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
                
            else:
                mask, logit, painted_image, score = self.xmem.track(images[i])
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
                scores.append(score)
        #del tensor_images
        return masks, logits, painted_images, scores
    
        
def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    parser.add_argument('--port', type=int, default=6080, help="only useful when running gradio applications")  
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--mask_save', default=False)
    args = parser.parse_args()

    if args.debug:
        print(args)
    return args 


if __name__ == "__main__":
    masks = None
    logits = None
    painted_images = None
    images = []
    image  = np.array(PIL.Image.open('/hhd3/gaoshang/truck.jpg'))
    args = parse_augment()
    # images.append(np.ones((20,20,3)).astype('uint8'))
    # images.append(np.ones((20,20,3)).astype('uint8'))
    images.append(image)
    images.append(image)

    mask = np.zeros_like(image)[:,:,0]
    mask[0,0]= 1
    trackany = TrackingAnything('/ssd1/gaomingqi/checkpoints/sam_vit_h_4b8939.pth','/ssd1/gaomingqi/checkpoints/XMem-s012.pth', args)
    masks, logits ,painted_images= trackany.generator(images, mask)
        
        
    
    
    