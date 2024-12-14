

import os
import numpy as np 
import pandas as pd
from .video_loading import load_all_images_davis, load_images_from_folder
from .masks_handler import unifyMasks, pad_to_divisible_by_two, generate_video_from_frames
from .metrics import compute_all_video_metrics, calculate_video_AP
from PIL import Image

all_tests_csv = './result/all_tests.csv'

def run_model_on_davis_set(name, model,videoLoader, compute_metrics = False,save_masks = False, compute_video = False, verbose = True):
    df_whole_metrics,df_per_frame_metrics,df_score = None,None,None


    if compute_metrics: 
        g_measures_by_video = ['Video','J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay','AP','AP@.5','AP@.75','AP_s','AP_m','AP_l','J-Statiscts-Object','F-Statiscts-Object']
        df_whole_metrics = pd.DataFrame(columns=g_measures_by_video)
        df_score = pd.DataFrame(columns=['Video','Scores'])
        df_per_frame_metrics_col = ['Video','Frame','J-Mean','F-Mean','J-Mean-Object','F-Mean-Object','J-Object','F-Object']
        df_per_frame_metrics = pd.DataFrame(columns=df_per_frame_metrics_col)
    
    folder_path = f'./result/{name}'
    if compute_metrics or compute_video or save_masks: 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if compute_video: 
            path_to_videos = folder_path + '/videos'
            if not os.path.exists(path_to_videos): os.makedirs(path_to_videos)

    for video in list(videoLoader):

        num_objects, info = video
        video_name = info['name'] 
        num_frames = info['num_frames']
        if videoLoader.resolution == '480p':
            height = info['size_480p'][1]
            width = info['size_480p'][0]
        else: 
            if videoLoader.year == '2017':
                height = 3840
                width = 2026
            else:
                height = 1920
                width = 1080        
        if verbose: print(f'Tracking Video {video_name} with dimensions {width}x{height}')
        if verbose: print('Loading dataset images and masks')
        images,ground_truth_masks = load_all_images_davis(videoLoader,(video_name,num_frames,num_objects.item()))
        
        if verbose: print('Creating first annotated mask for VOS model')

        model.xmem.current_video = video_name

        combined_masks = [[mask * (i+1) for i, mask in enumerate(frameMask)] for frameMask in ground_truth_masks]
        ground_truth_masks = [unifyMasks(mask, height, width) for mask in combined_masks]
        initial_mask = ground_truth_masks[0]
        #Compute masks for all images
        
        if verbose:print('Computing all masks')
        model.xmem.clear_memory()
        masks, logits, painted_images, scores = model.generator(images=images, template_mask=initial_mask)
        model.xmem.clear_memory() 
        
        df_score.loc[len(df_score)] = [video_name, [item[0] for item in scores]]
        #df_score.append({'Video': video_name, 'Scores': [item[0] for item in scores]}, ignore_index=True)
        
        if compute_metrics:
            if verbose: print('Computing Metrics')
            (f_mean, f_recall, f_decay),(j_mean, j_recall, j_decay),\
            (f_mean_obj, f_recall_obj, f_decay_obj),(j_mean_obj, j_recall_obj, j_decay_obj),\
            f_frame,j_frame = compute_all_video_metrics(video_name,masks[1:],ground_truth_masks[1:],df_per_frame_metrics)

            AP, AP_objectSize = calculate_video_AP(ground_truth_masks[1:],masks[1:], scores)

            AP_n = sum(AP.values())/len(AP)
            AP_5 = AP[0.5]
            AP_75 = AP[0.75]
            AP_s = sum(AP_objectSize[0].values())/len(AP_objectSize[0]) if AP_objectSize[0] is not None else np.nan
            AP_m = sum(AP_objectSize[1].values())/len(AP_objectSize[1]) if AP_objectSize[1] is not None else np.nan
            AP_l = sum(AP_objectSize[2].values())/len(AP_objectSize[2]) if AP_objectSize[2] is not None else np.nan

            df_whole_metrics.loc[len(df_whole_metrics)] = np.array([video_name,(f_mean+j_mean)/2,j_mean,j_recall,j_decay,f_mean,f_recall,f_decay,AP_n,AP_5,AP_75,AP_s,AP_m,AP_l,j_frame,f_frame])
            df_whole_metrics.loc[len(df_whole_metrics)] = np.array([video_name + '_object',(f_mean_obj+j_mean_obj)/2,j_mean_obj,j_recall_obj,j_decay_obj,f_mean_obj,f_recall_obj,f_decay_obj,AP_n,AP_5,AP_75,AP_s,AP_m,AP_l,j_frame,f_frame])

         
        if compute_video: 
            if verbose: print('Generating video')
            if width % 2 != 0 or height % 2 != 0: 
                painted_images = pad_to_divisible_by_two(painted_images)
            generate_video_from_frames(painted_images, output_path= path_to_videos + f"/{video_name}.mp4", fps = 10) 

        if save_masks:
            if verbose: print('Saving masks') 
            path_to_masks = folder_path + '/masks/' + video_name
            if not os.path.exists(path_to_masks): os.makedirs(path_to_masks)
            #for i,mask in enumerate(painted_images): 
            #    image = Image.fromarray(mask)
            #    image.save(os.path.join(path_to_masks, '{:05d}.png'.format(i)))
            from davisImpaiting.davisBaseImpainter import save_mask
            for i,mask in enumerate(masks): 
                save_mask(mask,os.path.join(path_to_masks, '{:05d}.png'.format(i)))
    
    if compute_metrics:
        df_per_frame_metrics.to_csv(folder_path + '/per_object_metrics.csv',index=False)
        df_whole_metrics.to_csv(folder_path + '/whole_metrics.csv',index=False)

        all_test_metrics = None
        if os.path.exists(all_tests_csv):
            all_test_metrics = pd.read_csv(all_tests_csv,index_col = None)
        else:
            all_test_col = ['Test','J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay',\
                                    'AP-Mean','AP@.5-Mean','AP@.75-Mean','AP_s-Mean','AP_m-Mean','AP_l-Mean',\
                                    'J&F-Mean-Obj', 'J-Mean-Obj', 'J-Recall-Obj', 'J-Decay-Obj', 'F-Mean-Obj', 'F-Recall-Obj','F-Decay-Obj']
            all_test_metrics = pd.DataFrame(columns=all_test_col)
        normal_mean = df_whole_metrics[~df_whole_metrics['Video'].str.contains('_object')].iloc[:, 1:-8].mean().tolist()
        ap_mean = df_whole_metrics[df_whole_metrics['Video'].str.contains('_object')].iloc[:, 8:14].mean().tolist()
        per_object_mean = df_whole_metrics[df_whole_metrics['Video'].str.contains('_object')].iloc[:, 1:-8].mean().tolist()
        all_test_metrics.loc[len(all_test_metrics)] = np.array([name] + normal_mean + ap_mean + per_object_mean)
        all_test_metrics.to_csv(all_tests_csv, index = False)
    
    df_score.to_csv(folder_path + '/scores.csv',index=False)

    return masks, logits, painted_images


def run_model_on_longdata_set(name, model,videoLoader, compute_metrics = False,save_masks = False, compute_video = False, verbose = True):
    df_whole_metrics,df_per_frame_metrics = None,None
    if compute_metrics: 
        g_measures_by_video = ['Video','J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay','AP','AP@.5','AP@.75','AP_s','AP_m','AP_l','J-Statiscts-Object','F-Statiscts-Object']
        df_whole_metrics = pd.DataFrame(columns=g_measures_by_video)
        df_per_frame_metrics_col = ['Video','Frame','J-Mean','F-Mean','J-Mean-Object','F-Mean-Object','J-Object','F-Object']
        df_per_frame_metrics = pd.DataFrame(columns=df_per_frame_metrics_col)
    
    folder_path = f'./result/{name}'
    if compute_metrics or compute_video or save_masks: 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if compute_video: 
            path_to_videos = folder_path + '/videos'
            if not os.path.exists(path_to_videos): os.makedirs(path_to_videos)
    
    for seq in list(videoLoader.get_sequences()):
        
        model.xmem.current_video = seq
        if verbose: print(f'Video: {seq}')

        all_gt_masks, _, all_masks_id = videoLoader.get_all_masks(seq, True)
        if os.name == 'nt': all_masks_id = [int(folder.split('\\')[-1]) for folder in all_masks_id]
        images_root = os.path.join(videoLoader.root_folder ,'JPEGImages',seq)
        file_names = sorted(os.listdir(images_root))
        file_ids = [int(name.split('.')[0]) for name in file_names]
        test_ids = [file_ids.index(int(mask_id)) for mask_id in all_masks_id[1:]]
        all_frames = load_images_from_folder(images_root,file_names)
        initial_mask = all_gt_masks[0,0,:,:]
        #test_ids = [int((int(all_masks_id[i]) - int(all_masks_id[0]))/3) for i in range(1,len(all_masks_id))]
        width,height,_= all_frames[0].shape
        
        #Compute masks for all images
        if verbose:print('Computing all masks')
        model.xmem.clear_memory()
        masks, logits, painted_images, scores = model.generator(images=all_frames, template_mask=initial_mask)
        model.xmem.clear_memory()  
        
        if compute_metrics:
            if verbose: print('Computing Metrics')
            masks_compute =  [masks[i] for i in test_ids]
            (f_mean, f_recall, f_decay),(j_mean, j_recall, j_decay),\
            (f_mean_obj, f_recall_obj, f_decay_obj),(j_mean_obj, j_recall_obj, j_decay_obj),\
            f_frame,j_frame = compute_all_video_metrics(seq,masks_compute,all_gt_masks[0,1:,:,:],df_per_frame_metrics)
            
            AP, AP_objectSize = calculate_video_AP(all_gt_masks[0,1:,:,:],masks_compute, scores)

            AP_n = sum(AP.values())/len(AP)
            AP_5 = AP[0.5]
            AP_75 = AP[0.75]
            AP_s = sum(AP_objectSize[0].values())/len(AP_objectSize[0]) if AP_objectSize[0] is not None else np.nan
            AP_m = sum(AP_objectSize[1].values())/len(AP_objectSize[1]) if AP_objectSize[1] is not None else np.nan
            AP_l = sum(AP_objectSize[2].values())/len(AP_objectSize[2]) if AP_objectSize[2] is not None else np.nan

            df_whole_metrics.loc[len(df_whole_metrics)] = np.array([seq,(f_mean+j_mean)/2,j_mean,j_recall,j_decay,f_mean,f_recall,f_decay,AP_n,AP_5,AP_75,AP_s,AP_m,AP_l,j_frame,f_frame])
            df_whole_metrics.loc[len(df_whole_metrics)] = np.array([seq + '_object',(f_mean_obj+j_mean_obj)/2,j_mean_obj,j_recall_obj,j_decay_obj,f_mean_obj,f_recall_obj,f_decay_obj,AP_n,AP_5,AP_75,AP_s,AP_m,AP_l,j_frame,f_frame])

                
        if compute_video: 
            if verbose: print('Generating video')
            if width % 2 != 0 or height % 2 != 0: 
                painted_images = pad_to_divisible_by_two(painted_images)
            generate_video_from_frames(painted_images, output_path= path_to_videos + f"/{seq}.mp4", fps = 10) 

        if save_masks:
            if verbose: print('Saving masks') 
            path_to_masks = folder_path + '/masks/' + seq
            if not os.path.exists(path_to_masks): os.makedirs(path_to_masks)
            #for i,mask in enumerate(painted_images): 
            #    image = Image.fromarray(mask)
            #    image.save(os.path.join(path_to_masks, '{:05d}.png'.format(i)))
            path_to_masks = path_to_masks + '/testedmasks'
            if not os.path.exists(path_to_masks): os.makedirs(path_to_masks)
            for i,j in zip(test_ids,all_masks_id[1:]): 
                image = Image.fromarray(painted_images[i])
                image.save(os.path.join(path_to_masks, '{:05d}.png'.format(j)))
            
                
    if compute_metrics:
        df_per_frame_metrics.to_csv(folder_path + '/per_object_metrics.csv',index=False)
        df_whole_metrics.to_csv(folder_path + '/whole_metrics.csv',index=False)

        all_test_metrics = None
        if os.path.exists(all_tests_csv):
            all_test_metrics = pd.read_csv(all_tests_csv,index_col = None)
        else:
            all_test_col = ['Test','J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay',\
                                    'AP-Mean','AP@.5-Mean','AP@.75-Mean','AP_s-Mean','AP_m-Mean','AP_l-Mean',\
                                    'J&F-Mean-Obj', 'J-Mean-Obj', 'J-Recall-Obj', 'J-Decay-Obj', 'F-Mean-Obj', 'F-Recall-Obj','F-Decay-Obj']
            all_test_metrics = pd.DataFrame(columns=all_test_col)
        normal_mean = df_whole_metrics[~df_whole_metrics['Video'].str.contains('_object')].iloc[:, 1:-8].mean().tolist()
        ap_mean = df_whole_metrics[df_whole_metrics['Video'].str.contains('_object')].iloc[:, 8:14].mean().tolist()
        per_object_mean = df_whole_metrics[df_whole_metrics['Video'].str.contains('_object')].iloc[:, 1:-8].mean().tolist()
        all_test_metrics.loc[len(all_test_metrics)] = np.array([name] + normal_mean + ap_mean + per_object_mean)
        all_test_metrics.to_csv(all_tests_csv, index = False)
                
    return masks, logits, painted_images


def run_model_on_longVOS_set(name, model,videoLoader, compute_metrics = False,save_masks = False, compute_video = False, verbose = True):
    
    folder_path = f'./resultLongVOS/{name}'
    if compute_metrics or compute_video or save_masks: 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if compute_video: 
            path_to_videos = folder_path + '/videos'
            if not os.path.exists(path_to_videos): os.makedirs(path_to_videos)
    
    for seq in list(videoLoader.get_sequences()):
        
        model.xmem.current_video = seq
        if verbose: print(f'Video: {seq}')

        all_gt_masks, _, all_masks_id = videoLoader.get_all_masks(seq, True) 
        if os.name == 'nt': all_masks_id = [int(folder.split('\\')[-1]) for folder in all_masks_id]
        images_root = os.path.join(videoLoader.root_folder ,'JPEGImages',seq)
        file_names = sorted(os.listdir(images_root))
        file_ids = [int(name.split('.')[0]) for name in file_names]
        test_ids = [file_ids.index(int(mask_id)) for mask_id in all_masks_id[1:]]
        all_frames = load_images_from_folder(images_root,file_names)
        initial_mask = all_gt_masks[0,0,:,:]
        #test_ids = [int((int(all_masks_id[i]) - int(all_masks_id[0]))/3) for i in range(1,len(all_masks_id))]
        width,height,_= all_frames[0].shape
        
        #Compute masks for all images
        if verbose:print('Computing all masks')
        model.xmem.clear_memory()
        masks, logits, painted_images, scores = model.generator(images=all_frames, template_mask=initial_mask)
        model.xmem.clear_memory()  
        
        if compute_video: 
            if verbose: print('Generating video')
            if width % 2 != 0 or height % 2 != 0: 
                painted_images = pad_to_divisible_by_two(painted_images)
            generate_video_from_frames(painted_images, output_path= path_to_videos + f"/{seq}.mp4", fps = 10) 

        if save_masks:
            if verbose: print('Saving masks') 
            path_to_masks = folder_path + '/masks/' + seq
            if not os.path.exists(path_to_masks): os.makedirs(path_to_masks)
            for i,mask in enumerate(painted_images): 
                image = Image.fromarray(mask)
                image.save(os.path.join(path_to_masks, '{:05d}.png'.format(i)))
            #path_to_masks = path_to_masks
            #if not os.path.exists(path_to_masks): os.makedirs(path_to_masks)
            #for i,mask in enumerate(masks): 
            #    print('Saving Masks 2')
            #    image = Image.fromarray(mask)
            #    image.save(os.path.join(path_to_masks, '{:05d}_2.png'.format(i)))

            #from davisImpaiting.davisBaseImpainter import save_mask
            #for i,mask in enumerate(masks): 
            #    save_mask(mask,os.path.join(path_to_masks, '{:05d}.png'.format(i)))
            
            

    return masks, logits, painted_images