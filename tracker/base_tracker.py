# import for debugging
import os
import glob
import numpy as np
from PIL import Image
# import for base_tracker
import torch
import yaml
import torch.nn.functional as F
from tracker.model.network import XMem
from inference.inference_core import InferenceCore
from tracker.util.mask_mapper import MaskMapper
from torchvision import transforms
from tracker.util.range_transform import im_normalization

from tools.painter import mask_painter
from tools.base_segmenter import BaseSegmenter
from torchvision.transforms import Resize
import progressbar
import cv2


class BaseTracker:
    def __init__(self, xmem_checkpoint, device, sam_model=None, sam_mode = None, model_type=None) -> None:
        """
        device: model device
        xmem_checkpoint: checkpoint of XMem model
        """
        # load configurations
        with open("tracker/config/config.yaml", 'r') as stream: 
            config = yaml.safe_load(stream) 
        # initialise XMem
        device = device if torch.cuda.is_available() else torch.device("cpu")
        network = XMem(config, xmem_checkpoint).to(device).eval()
        # initialise IncerenceCore
        self.tracker = InferenceCore(network, config)
        # data transformation
        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])
        self.device = device
        
        # changable properties
        self.mapper = MaskMapper()
        self.initialised = False

        # # SAM-based refinement
        if sam_model:  assert sam_mode in ['point','bbox'], 'Sam Refinment Mode must be point or bbox'
        self.sam_model = sam_model
        self.resizer = Resize([256, 256])
        self.sam_refinement_mode = sam_mode
        self.update_vos_mem = True

        if sam_model: print('Sam Refinement ACTIVATED. Mode: '+ self.sam_refinement_mode)
        else: print('Sam Refinement NOT ACTIVATED')

    @torch.no_grad()
    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(mask, (int(h/min_hw*self.size), int(w/min_hw*self.size)), 
                    mode='nearest')

    @torch.no_grad()
    def track(self, frame, first_frame_annotation=None):
        """
        Input: 
        frames: numpy arrays (H, W, 3)
        logit: numpy array (H, W), logit

        Output:
        mask: numpy arrays (H, W)
        logit: numpy arrays, probability map (H, W)
        painted_image: numpy array (H, W, 3)
        """

        if first_frame_annotation is not None:   # first frame mask
            # initialisation
            mask, labels = self.mapper.convert_mask(first_frame_annotation)
            mask = torch.Tensor(mask).to(self.device)
            self.tracker.set_all_labels(list(self.mapper.remappings.values()))
        else:
            mask = None
            labels = None
        # prepare inputs
        frame_tensor = self.im_transform(frame).to(self.device)
        # track one frame
        probs, logits = self.tracker.step(frame_tensor, mask, labels)   # logits 2 (bg fg) H W
        # # refine
        # Logits is a torch tensor. torch.Size([Num Objects, H, W]
        #if first_frame_annotation is None and self.sam_model:
        #    print('Sam Refineent entry', logits.size())
        #    out_mask = self.custom_sam_refinement(frame, logits, 1)    
        
        # convert to mask
        out_mask = torch.argmax(probs, dim=0)
        out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

        if first_frame_annotation is None and self.sam_model:
            #print('Sam Refinment. Mode: ' + self.sam_refinement_mode)
            out_mask = self.custom_sam_refinement(frame,out_mask)
            if self.update_vos_mem: 
                print('Updating Memory post Refinment')
                self.tracker.step(frame_tensor, torch.Tensor(out_mask).to(self.device), self.tracker.all_labels)

        final_mask = np.zeros_like(out_mask)
        
        # map back
        for k, v in self.mapper.remappings.items():
            final_mask[out_mask == v] = k

        num_objs = final_mask.max()
        painted_image = frame
        for obj in range(1, num_objs+1):
            if np.max(final_mask==obj) == 0:
                continue
            painted_image = mask_painter(painted_image, (final_mask==obj).astype('uint8'), mask_color=obj+1)

        # print(f'max memory allocated: {torch.cuda.max_memory_allocated()/(2**20)} MB')

        return final_mask, logits, painted_image
    
    def compute_bounding_box(self,segmentation_mask):
        # Get the indices where the segmentation mask is non-zero
        nonzero_indices = np.nonzero(segmentation_mask)
        
        # Calculate the bounding box coordinates
        min_row = np.min(nonzero_indices[0])
        max_row = np.max(nonzero_indices[0])
        min_col = np.min(nonzero_indices[1])
        max_col = np.max(nonzero_indices[1])
        
        # Return the bounding box coordinates as a tuple
        bounding_box = [min_col,min_row, max_col, max_row]
        return bounding_box
    
    def get_best_point_of_interest(self,segmentation_mask):
        # Find contours in the segmentation mask
        points = []
        contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Extract pouints from moments
            M = cv2.moments(contour)
            if M["m00"] != 0: points.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])
            else: 
                print('Zero division')
                points.append([int(M["m10"]), int(M["m01"])])

        return np.array(points).astype('int')

    # def get_best_point_of_interest(self,segmentation_mask):
    #     # Find contours in the segmentation mask
    #     contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     # Calculate the area of each contour
    #     contour_areas = [cv2.contourArea(contour) for contour in contours]

    #     # Find the index of the contour with the largest area
    #     max_area_index = np.argmax(contour_areas)

    #     # Extract the bounding box coordinates of the contour
    #     x, y, w, h = cv2.boundingRect(contours[max_area_index])

    #     # Calculate the centroid of the bounding box
    #     centroid_x = x + (w / 2)
    #     centroid_y = y + (h / 2)

    #     return np.array([[centroid_x, centroid_y]]).astype('int')

    @torch.no_grad()
    def custom_sam_refinement(self, frame, out_mask):
        
        all_masks_separated = []
        all_mask_position = []
        for i, v in enumerate(list(np.unique(out_mask))):
            if v == 0: continue
            current_mask = np.zeros_like(out_mask)
            current_mask[out_mask == v] = 1
            all_masks_separated.append(current_mask)
            all_mask_position.append(v)

        self.sam_model.sam_controler.set_image(frame)

        if self.sam_refinement_mode == 'bbox':
            bounding_boxes = [self.compute_bounding_box(mask) for mask in all_masks_separated]
            bounding_boxes = torch.tensor(bounding_boxes, device= self.sam_model.sam_controler.predictor.device)
            transformed_boxes = self.sam_model.sam_controler.predictor.transform.apply_boxes_torch(bounding_boxes, frame.shape[:2])
            mode = 'bounding_boxes'
            prompts = {'bounding_boxes': transformed_boxes}
            masksout, scores, logits = self.sam_model.sam_controler.predict(prompts, mode, multimask=False)
            masksout = masksout.numpy()

        elif self.sam_refinement_mode == 'point':
            points_of_interest = [self.get_best_point_of_interest(mask) for mask in all_masks_separated]
            masksout = []
            for points in points_of_interest:
                mode = 'point'
                prompts = {
                    'point_coords': points,
                    'point_labels': np.ones((points.shape[0])).astype('uint8'), 
                }
                masksout_ind, scores, logits = self.sam_model.sam_controler.predict(prompts, mode, multimask=False)
                masksout.append(masksout_ind)

        final_mask = np.zeros_like(out_mask)
        for v, mask in zip(all_mask_position,masksout):
            final_mask +=  mask.squeeze(0).astype('uint8') * v

        self.sam_model.sam_controler.reset_image()
        return final_mask

    # @torch.no_grad()
    # def custom_sam_refinement(self, frame, logits, ti):
    #     """
    #     refine segmentation results with mask prompt
    #     """
    #     for i in range(0,logits.size(dim=0)):
    #         # convert to 1, 256, 256
    #         self.sam_model.sam_controler.set_image(frame)
    #         mode = 'mask'
    #         ind_logits = logits[i].unsqueeze(0)
    #         ind_logits = self.resizer(ind_logits).cpu().numpy()
    #         prompts = {'mask_input': ind_logits}    # 1 256 256
    #         masks, scores, logits_out = self.sam_model.sam_controler.predict(prompts, mode, multimask=True)  # masks (n, h, w), scores (n,), logits (n, 256, 256)
    #         painted_image = mask_painter(frame, masks[np.argmax(scores)].astype('uint8'), mask_alpha=0.8)
    #         painted_image = Image.fromarray(painted_image)
    #         painted_image.save(f'./result/refinement/obj{i}-{ti:05d}.png')
    #         self.sam_model.sam_controler.reset_image()

    @torch.no_grad()
    def sam_refinement(self, frame, logits, ti):
        """
        refine segmentation results with mask prompt
        """
        # convert to 1, 256, 256
        self.sam_model.sam_controler.set_image(frame)
        mode = 'mask'
        logits = logits.unsqueeze(0)
        logits = self.resizer(logits).cpu().numpy()
        prompts = {'mask_input': logits}    # 1 256 256
        masks, scores, logits = self.sam_model.sam_controler.predict(prompts, mode, multimask=True)  # masks (n, h, w), scores (n,), logits (n, 256, 256)
        painted_image = mask_painter(frame, masks[np.argmax(scores)].astype('uint8'), mask_alpha=0.8)
        painted_image = Image.fromarray(painted_image)
        painted_image.save(f'./result/refinement/{ti:05d}.png')
        self.sam_model.sam_controler.reset_image()

    @torch.no_grad()
    def clear_memory(self):
        self.tracker.clear_memory()
        self.mapper.clear_labels()
        torch.cuda.empty_cache()


##  how to use:
##  1/3) prepare device and xmem_checkpoint
#   device = 'cuda:2'
#   XMEM_checkpoint = '/ssd1/gaomingqi/checkpoints/XMem-s012.pth'
##  2/3) initialise Base Tracker
#   tracker = BaseTracker(XMEM_checkpoint, device, None, device)    # leave an interface for sam model (currently set None)
##  3/3) 


if __name__ == '__main__':
    # video frames (take videos from DAVIS-2017 as examples)
    video_path_list = glob.glob(os.path.join('/ssd1/gaomingqi/datasets/davis/JPEGImages/480p/horsejump-high', '*.jpg'))
    video_path_list.sort()
    # load frames
    frames = []
    for video_path in video_path_list:
        frames.append(np.array(Image.open(video_path).convert('RGB')))
    frames = np.stack(frames, 0)    # T, H, W, C
    # load first frame annotation
    first_frame_path = '/ssd1/gaomingqi/datasets/davis/Annotations/480p/horsejump-high/00000.png'
    first_frame_annotation = np.array(Image.open(first_frame_path).convert('P'))    # H, W, C

    # ------------------------------------------------------------------------------------
    # how to use
    # ------------------------------------------------------------------------------------
    # 1/4: set checkpoint and device
    device = 'cuda:2'
    XMEM_checkpoint = '/ssd1/gaomingqi/checkpoints/XMem-s012.pth'
    # SAM_checkpoint= '/ssd1/gaomingqi/checkpoints/sam_vit_h_4b8939.pth'
    # model_type = 'vit_h'
    # ------------------------------------------------------------------------------------
    # 2/4: initialise inpainter
    tracker = BaseTracker(XMEM_checkpoint, device, None, device)
    # ------------------------------------------------------------------------------------
    # 3/4: for each frame, get tracking results by tracker.track(frame, first_frame_annotation)
    # frame: numpy array (H, W, C), first_frame_annotation: numpy array (H, W), leave it blank when tracking begins
    painted_frames = []
    for ti, frame in enumerate(frames):
        if ti == 0:
            mask, prob, painted_frame = tracker.track(frame, first_frame_annotation)
            # mask: 
        else:
            mask, prob, painted_frame = tracker.track(frame)
        painted_frames.append(painted_frame)
    # ----------------------------------------------
    # 3/4: clear memory in XMEM for the next video
    tracker.clear_memory()
    # ----------------------------------------------
    # end
    # ----------------------------------------------
    print(f'max memory allocated: {torch.cuda.max_memory_allocated()/(2**20)} MB')
    # set saving path
    save_path = '/ssd1/gaomingqi/results/TAM/blackswan'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # save
    for painted_frame in progressbar.progressbar(painted_frames):
        painted_frame = Image.fromarray(painted_frame)
        painted_frame.save(f'{save_path}/{ti:05d}.png')

    # tracker.clear_memory()
    # for ti, frame in enumerate(frames):
    #     print(ti)
    #     # if ti > 200:
    #     #     break
    #     if ti == 0:
    #         mask, prob, painted_image = tracker.track(frame, first_frame_annotation)
    #     else:
    #         mask, prob, painted_image = tracker.track(frame)
    #     # save
    #     painted_image = Image.fromarray(painted_image)
    #     painted_image.save(f'/ssd1/gaomingqi/results/TrackA/gsw/{ti:05d}.png')

    # # track anything given in the first frame annotation
    # for ti, frame in enumerate(frames):
    #     if ti == 0:
    #         mask, prob, painted_image = tracker.track(frame, first_frame_annotation)
    #     else:
    #         mask, prob, painted_image = tracker.track(frame)
    #     # save
    #     painted_image = Image.fromarray(painted_image)
    #     painted_image.save(f'/ssd1/gaomingqi/results/TrackA/horsejump-high/{ti:05d}.png')

    # # ----------------------------------------------------------
    # # another video
    # # ----------------------------------------------------------
    # # video frames
    # video_path_list = glob.glob(os.path.join('/ssd1/gaomingqi/datasets/davis/JPEGImages/480p/camel', '*.jpg'))
    # video_path_list.sort()
    # # first frame
    # first_frame_path = '/ssd1/gaomingqi/datasets/davis/Annotations/480p/camel/00000.png'
    # # load frames
    # frames = []
    # for video_path in video_path_list:
    #     frames.append(np.array(Image.open(video_path).convert('RGB')))
    # frames = np.stack(frames, 0)    # N, H, W, C
    # # load first frame annotation
    # first_frame_annotation = np.array(Image.open(first_frame_path).convert('P'))    # H, W, C

    # print('first video done. clear.')

    # tracker.clear_memory()
    # # track anything given in the first frame annotation
    # for ti, frame in enumerate(frames):
    #     if ti == 0:
    #         mask, prob, painted_image = tracker.track(frame, first_frame_annotation)
    #     else:
    #         mask, prob, painted_image = tracker.track(frame)
    #     # save
    #     painted_image = Image.fromarray(painted_image)
    #     painted_image.save(f'/ssd1/gaomingqi/results/TrackA/camel/{ti:05d}.png')

    # # failure case test
    # failure_path = '/ssd1/gaomingqi/failure'
    # frames = np.load(os.path.join(failure_path, 'video_frames.npy'))
    # # first_frame = np.array(Image.open(os.path.join(failure_path, 'template_frame.png')).convert('RGB'))
    # first_mask = np.array(Image.open(os.path.join(failure_path, 'template_mask.png')).convert('P'))
    # first_mask = np.clip(first_mask, 0, 1)

    # for ti, frame in enumerate(frames):
    #     if ti == 0:
    #         mask, probs, painted_image = tracker.track(frame, first_mask)
    #     else:
    #         mask, probs, painted_image = tracker.track(frame)
    #     # save
    #     painted_image = Image.fromarray(painted_image)
    #     painted_image.save(f'/ssd1/gaomingqi/failure/LJ/{ti:05d}.png')
    #     prob = Image.fromarray((probs[1].cpu().numpy()*255).astype('uint8'))

    #     # prob.save(f'/ssd1/gaomingqi/failure/probs/{ti:05d}.png')
