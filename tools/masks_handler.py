import numpy as np
import torch 
import os 
import torchvision

def unifyMasks(masks, width, height):
    # Crear una matriz vacía para la máscara combinada
    unified = np.zeros((height, width), dtype=np.uint8)

    # Combinar las máscaras en la matriz vacía
    for mask in masks:
        unified += mask  # Sumar la máscara a la máscara combinada

    
    return unified


def pad_to_divisible_by_two(frames):
    max_height = max(frame.shape[0] for frame in frames)
    max_width = max(frame.shape[1] for frame in frames)
    new_height = max_height + 1 if max_height % 2 != 0 else max_height
    new_width = max_width + 1 if max_width % 2 != 0 else max_width

    padded_frames = []
    for frame in frames:
        height_pad = new_height - frame.shape[0]
        width_pad = new_width - frame.shape[1]
        padded_frame = np.pad(frame, ((0, height_pad), (0, width_pad), (0, 0)), mode='constant')
        padded_frames.append(padded_frame)

    return padded_frames


def generate_video_from_frames(frames, output_path, fps=30):
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path