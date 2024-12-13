
import numpy as np
from dataset.errorfunctions import db_eval_boundary,db_eval_iou
import warnings
import pandas as pd
import cv2

def calculate_real_iou(mask1, mask2):
    # Ensure both masks have the same shape
    assert mask1.shape == mask2.shape, "Mask shapes must be the same."

    # Calculate intersection and union for each label
    #labels = np.unique(np.concatenate((mask1, mask2)))[1:]
    labels =np.unique(mask2)
    
    iou_per_label = {}
    iou_values = []

    for label in labels:
        mask1_label = mask1 == label
        mask2_label = mask2 == label
        iou = db_eval_iou(mask1_label,mask2_label)
        iou_per_label[label] = iou
        iou_values.append(iou)

    # Calculate IoU as total of the mask 
    #iou = np.sum(intersection) / np.sum(union)
    iou = np.nanmean(iou_values)
    
    # Calculate IoU as mean of objects
    iou_mean_object = sum(iou_per_label.values()) / len(iou_per_label)

    return iou,iou_mean_object, iou_per_label

def compute_f_score(true_positives, false_positives, false_negatives):
    divider = (true_positives + false_positives)
    precision = (true_positives / divider) if divider != 0 else 0

    divider = (true_positives + false_negatives)
    recall = (true_positives / divider) if divider != 0 else 0

    divider = (precision + recall)
    f_measure = (2 * (precision * recall) / divider) if divider != 0 else 0
    return f_measure

def compute_f_measure(mask1, mask2):
    # Ensure both masks have the same shape
    assert mask1.shape == mask2.shape, "Mask shapes must be the same."

    # Calculate F-measure for each label
    #labels = np.unique(np.concatenate((mask1, mask2)))[1:]
    labels =np.unique(mask2)
    f_measure_per_label = {}
    add_true_positives = 0
    add_false_positives = 0
    add_false_negatives = 0

    for label in labels:
        mask1_label = mask1 == label
        mask2_label = mask2 == label

        true_positives = np.logical_and(mask1_label, mask2_label).sum()
        false_positives = np.logical_and(mask1_label, np.logical_not(mask2_label)).sum()
        false_negatives = np.logical_and(np.logical_not(mask1_label), mask2_label).sum()

        add_true_positives += true_positives
        add_false_positives += false_positives
        add_false_negatives += false_negatives

        f_measure = compute_f_score(true_positives, false_positives, false_negatives)
        f_measure_per_label[label] = f_measure

    # Calculate F Measure as total of the mask 
    overall_f_measure = compute_f_score(add_true_positives, add_false_positives, add_false_negatives)

    # Calculate IoU as mean of objects
    f_mean_object = sum(f_measure_per_label.values()) / len(f_measure_per_label)

    return overall_f_measure,f_mean_object,f_measure_per_label


def calculate_iou(mask1, mask2):
    # Ensure both masks have the same shape
    assert mask1.shape == mask2.shape, "Mask shapes must be the same."

    # Calculate intersection and union for each label
    labels = np.unique(np.concatenate((mask1, mask2)))[1:]
    #labels =np.intersect1d(np.unique(mask1), np.unique(mask2))[1:]

    labels =np.unique(mask2)[1:]
    intersection = np.zeros_like(mask1, dtype=np.float32)
    union = np.zeros_like(mask1, dtype=np.float32)
    iou_per_label = {}

    for label in labels:
        mask1_label = mask1 == label
        mask2_label = mask2 == label
        c_intersection = np.logical_and(mask1_label, mask2_label)
        c_union = np.logical_or(mask1_label, mask2_label)
        intersection += c_intersection
        union += c_union
        iou_per_label[label] = np.sum(c_intersection) / np.sum(c_union)
    # Calculate IoU as total of the mask 
 
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else np.nan
    #aaa iou = np.sum(intersection) / np.sum(union) 
    
    
    # Calculate IoU as mean of objects
    if len(iou_per_label) > 0: 
        iou_mean_object = sum(iou_per_label.values()) / len(iou_per_label)
    else: iou_mean_object  = 0

    #print('J_Measure->OverAll, Object Mean, Per label',iou,iou_mean_object, iou_per_label)
    return iou,iou_mean_object, iou_per_label


def compute_real_f_measure(mask1, mask2): #(mask_infered,mask_gt)
    # Ensure both masks have the same shape
    assert mask1.shape == mask2.shape, "Mask shapes must be the same."

    # Calculate F-measure for each label
    labels = np.unique(np.concatenate((mask1, mask2)))[1:]
    labels =np.unique(mask2)[1:]
    f_measure_per_label = {}
    f_measures = []
    all_precision = []
    all_recall = []

    for label in labels:
        mask1_label = mask1 == label
        mask2_label = mask2 == label

        f_measure,precision,recall = db_eval_boundary(mask1_label, mask2_label)
        all_precision.append(precision)
        all_recall.append(recall)
        f_measure_per_label[label] = f_measure
        f_measures.append(f_measure)

    # Calculate F Measure as total of the mask 

    overall_f_measure = np.nanmean(f_measures) if len(f_measures) != 0 else np.nan  
    #aaa overall_f_measure = np.nanmean(f_measures)    

   
    # Calculate IoU as mean of objects
    f_mean_object = sum(f_measure_per_label.values()) / len(f_measure_per_label) if len(f_measure_per_label) != 0 else 0.0
    #aaaf_mean_object = sum(f_measure_per_label.values()) / len(f_measure_per_label)

    return overall_f_measure,f_mean_object,f_measure_per_label

def split_dict_list_to_lists(dict_list):

    key_lists = {}
    for dictionary in dict_list:
        for key, value in dictionary.items():
            if key in key_lists:
                key_lists[key].append(value)
            else:
                key_lists[key] = [value]
    result = [values_list for _, values_list in key_lists.items()]
    keys_in_order = list(key_lists.keys())
    return result, keys_in_order

def add_dict(list_of_dicts):
    mean_dict = {}
    key_counts = {}

    for d in list_of_dicts:
        for key, value in d.items():
            mean_dict[key] = mean_dict.get(key, 0) + value
            key_counts[key] = key_counts.get(key, 0) + 1

    for key in mean_dict:
        mean_dict[key] /= key_counts[key]
    return mean_dict

def db_statistics(per_frame_values):
    """ Compute mean,recall and decay from per-frame evaluation.
    Arguments:
        per_frame_values (ndarray): per-frame evaluation

    Returns:
        M,O,D (float,float,float):
            return evaluation statistics: mean,recall,decay.
    """

    # strip off nan values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M = np.nanmean(per_frame_values)
        O = np.nanmean(per_frame_values > 0.5)

    N_bins = 4
    ids = np.round(np.linspace(1, len(per_frame_values), N_bins + 1) + 1e-10) - 1
    ids = ids.astype(np.uint8)

    D_bins = [per_frame_values[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])

    return M, O, D
def compute_statistics_per_label(args): 
    f_per_objectframe_list,list_of_keys = args
    metrics_dict = {}
    for metrics,key in zip (f_per_objectframe_list,list_of_keys):
        metrics_dict[f'{key}'] = db_statistics(np.array(metrics))
    return metrics_dict

def compute_all_video_metrics(name,masks,ground_truth_masks,df_per_frame_metrics):
    f_measure_lst,f_measure_object_lst, f_measure_per_label_lst, iou_lst, iou_object_lst, iou_per_label_lst  =  [], [], [], [], [], []
    for i,(mask_infered, mask_gt) in enumerate(zip(masks,ground_truth_masks)):
            #print(f'Frame {i+1}: Real Values {len(np.unique(mask_gt)) - 1}, Values Infered {len(np.unique(mask_infered)) - 1}  ')
            f_measure,f_measure_object, f_measure_per_label = compute_real_f_measure(mask_infered,mask_gt)
            #f_real_measure,f_real_measure_object, f_real_measure_per_label = compute_real_f_measure(mask_infered, mask_gt)
            #print(f'F measure: {f_measure}, F measure per object {f_measure_object}, F measure {f_measure_per_label}')
            #print(f'REAL: F measure: {f_real_measure}, F measure per object {f_real_measure_object}, F measure {f_real_measure_per_label}')
            iou,iou_object, iou_per_label = calculate_iou(mask_infered,mask_gt)
            #iou_real,iou_object_real, iou_per_label_real = calculate_real_iou(mask_infered,mask_gt)
            #print(f'Iou measure: {iou}, Iou measure per object {iou_object}, Iou measure {iou_per_label}')
            #print(f'REAL Iou measure: {iou_real}, Iou measure per object {iou_object_real}, Iou measure {iou_per_label_real}')
            df_per_frame_metrics.loc[len(df_per_frame_metrics)] = np.array([name,i + 1,f_measure,iou,f_measure_object,iou_object,f_measure_per_label,iou_per_label])
            #print(f'Mask {i + 1}: f_mesure {f_measure}, per label {f_measure_per_label}, iou {iou}, per label {iou_per_label}')

            f_measure_lst.append(f_measure)
            f_measure_object_lst.append(f_measure_object)
            f_measure_per_label_lst.append(f_measure_per_label)
            
            iou_lst.append(iou)
            iou_object_lst.append(iou_object)
            iou_per_label_lst.append(iou_per_label)
    
    f_statistics = db_statistics(np.array(f_measure_lst))
    j_statistics = db_statistics(np.array(iou_lst))

    f_statistics_object = db_statistics(np.array(f_measure_object_lst))
    j_statistics_object = db_statistics(np.array(iou_object_lst))

    f_statistics_per_label = compute_statistics_per_label(split_dict_list_to_lists(f_measure_per_label_lst))
    j_statistics_per_label = compute_statistics_per_label(split_dict_list_to_lists(iou_per_label_lst))
    return f_statistics,j_statistics,f_statistics_object,j_statistics_object, f_statistics_per_label, j_statistics_per_label


def compute_iou(mask1, mask2):
    """Compute Intersection over Union (IoU) for two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union

def compute_auc_interpolated(precisions, recalls):
    sorted_indices = np.argsort(recalls)
    sorted_recalls = np.array(recalls)[sorted_indices]
    sorted_precisions = np.array(precisions)[sorted_indices]
    
    interpolated_precisions = np.maximum.accumulate(sorted_precisions[::-1])[::-1]
    area = np.trapz(interpolated_precisions, sorted_recalls)
    
    return area



def true_positives(gt_masks, pred_masks, scores):

    iou_thresholds = [round(v,2) for v in np.arange(0.5, 1.0, 0.05)]
    columns = ['Frame','Object','Object_Type','Confidences'] + iou_thresholds 
    df_tp = pd.DataFrame(columns=columns)
    n_objects,n_objects_s,n_objects_m,n_objects_l = 0,0,0,0

    for i,(gt_mask, pred_mask) in enumerate(zip(gt_masks, pred_masks)):

        labels_gt = np.unique(gt_mask)
        labels_gt = np.delete(labels_gt,np.where(labels_gt == 0))
        n_objects += len(labels_gt)
        labels_dt = np.unique(pred_mask)
        labels_dt = np.delete(labels_dt,np.where(labels_dt == 0))

        iou_thresholds = np.arange(0.5, 1.0, 0.05)

        '''
        if len(labels_gt) != len(labels_dt): 
            print(f'Frame {i}, GT {len(labels_gt)}, DT{len(labels_dt)}')
            print('Ground_truth')
            plt.imshow(gt_mask)
            plt.show()
            print('Prediction')
            plt.imshow(pred_mask)
            plt.show()
        '''
        #print(f'Labels GT {labels_gt}')
        #print(f'Labels DT {labels_dt}')

        #print(f'Scores {scores[i]}')

        for j,label in enumerate(labels_gt):
            gt_mask_label = gt_mask == label
            object_type = None
            area = cv2.countNonZero(gt_mask_label*1)
            #print(f'Frame {i}, Object {j}')
            if area <= 32*32: 
                object_type = 0
                n_objects_s += 1
            elif area <= 96*96: 
                object_type = 1
                n_objects_m += 1
            else: 
                object_type = 2
                n_objects_l += 1

            if label in labels_dt:
                pred_mask_label = pred_mask == label

                iou = compute_iou(gt_mask_label, pred_mask_label)
                tp_values = [1 if iou >= iou_threshold else 0 for iou_threshold in iou_thresholds]
                df_tp.loc[len(df_tp)] = np.array([i + 1,label,object_type,scores[i][np.where(labels_dt == label)[0].item()]] + tp_values )

    return df_tp,[n_objects,n_objects_s,n_objects_m,n_objects_l]


def compute_AP_for_df(df,n):
    df = df.sort_values(by='Confidences', ascending=False)
    AP = {}
    for column in df.columns[4:].tolist():
        pos = df.columns.get_loc(column)
        df.insert(pos + 1,f'{column}_fp',1 - df[column])
        df.insert(pos + 2,f'{column}_accTp',df[column].cumsum())
        df.insert(pos + 3,f'{column}_accFp',df[f'{column}_fp'].cumsum())
        df.insert(pos + 4,f'{column}_Precision',df[f'{column}_accTp'] / (df[f'{column}_accTp']+df[f'{column}_accFp']))
        df.insert(pos + 4,f'{column}_Recall',df[f'{column}_accTp'] / n)
        AP[column] = compute_auc_interpolated(df[f'{column}_Precision'].values,df[f'{column}_Recall'].values)
        #print_curve(df[f'{column}_Precision'].values,df[f'{column}_Recall'].values,f'Precision Recall Curve_{column}')
        #print(AP[column])
        df.drop(columns = [f'{column}_fp',f'{column}_accTp',f'{column}_accFp',f'{column}_Precision',f'{column}_Recall'])
    return AP


def calculate_video_AP(gt_masks, pred_masks, scores):
    df,object_counts = true_positives(gt_masks, pred_masks,scores)
    AP = compute_AP_for_df(df,object_counts[0])
    AP_size = []
    for object_type in range(0,3):
        if df['Object_Type'].isin([object_type]).any(): AP_size.append(compute_AP_for_df(df[df['Object_Type'] == object_type],[object_counts[object_type+1]]))
        else: AP_size.append(None)
    return AP, AP_size
