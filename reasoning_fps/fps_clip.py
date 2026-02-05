import json
import os
import cv2
import numpy as np
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import re

import traceback

def uniform_indices(start, end, num=3):
                   
    idx = np.linspace(start, end, num=num, dtype=float)
    idx = np.round(idx).astype(int) 
    idx = np.clip(idx, start, end)
    uniq = np.unique(idx)
    if len(uniq) < num:
        extra = np.setdiff1d(idx, uniq, assume_unique=False)
        idx = np.concatenate([uniq, extra])[:num]
    else:
        idx = uniq[:num]
    return idx

def extract_frames_fps(video_path, output_dir="", fps=1):
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video:", video_path)
        return
    frame_list = []
    if fps > 0:

        video_fps = cap.get(cv2.CAP_PROP_FPS)  
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(video_fps // fps) if fps < video_fps else 1
        
        num = int(np.ceil(total_frames/frame_interval))
        frame_count = 0
        saved_count = 0
        target_index = np.linspace(0, total_frames-1, num=num, dtype=int)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count in target_index:
                saved_count += 1
                frame_filename = os.path.join(output_dir, f"fps_frame_{frame_count:04d}.jpg")
                frame_list.append(frame_filename)
                cv2.imwrite(frame_filename, frame)

            frame_count += 1

        cap.release()
        print(f"{output_dir}: total frames({frame_count}), saves frames({saved_count})")
    
    elif fps == -1:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        #---- 2/5 of the frames are fixed in number for the later part.
        start_frame_idx = int(total_frames * (3/5))  # The first 3/5 is completed, and the last 2/5 begins (total frame count × 0.6)
        end_frame_idx = total_frames - 1  
        frame_count = 3
        idxs = uniform_indices(start_frame_idx, end_frame_idx, frame_count)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            
            if not ok:
                continue
            frame_filename = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            frame_list.append(frame_filename)
            cv2.imwrite(frame_filename, frame)
            print(f"Saved: {frame_filename}")

        cap.release()
    
    return frame_list


def extra_last_frame(video_path, output_dir=""):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video:", video_path)
        return
    frame_list = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, last_frame = cap.read()
    if ret:
        last_frame_path = os.path.join(output_dir, "frame_1.jpg")
        frame_list.append(last_frame_path)
        cv2.imwrite(last_frame_path, last_frame)
        print(f"save last frame: {last_frame_path}")
    
    return frame_list

if __name__ == "__main__":
    
    
    data_json = " " #video result json
    root_folder = " " #path to save frame 
    
    os.makedirs(root_folder, exist_ok=True)
    data = json.load(open(data_json, "r"))
    for idx, data_dict in enumerate(data):
        out_path = os.path.join(root_folder, data_dict["task_id"])
        os.makedirs(out_path, exist_ok=True)

        if type(data_dict["extra_frame"]) is int and "video_path" in data_dict.keys():##############
            video = data_dict["video_path"]
            frames_list = extract_frames_fps(video, output_dir=out_path, fps=data_dict["extra_frame"])
            data_dict['frame_path'] = frames_list
        elif data_dict["extra_frame"] in ["maze", "sym", "gt"]:
            video = data_dict["video_path"]
            frames_list = extra_last_frame(video, output_dir=out_path)
            data_dict['frame_path'] = frames_list

with open(data_json, "w", encoding="utf-8") as f:
    json.dump(data, f,  indent=4, ensure_ascii=False)

