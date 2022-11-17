# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 17:43:48 2022

@author: Yuuki Misaki
"""

import os
import h5py
import numpy as np
from matplotlib import pyplot as plt


def upsampling(shot_level, segment):
    total_frame = segment[-1][1]
    frame_level = np.zeros(total_frame)
    for i, seg in enumerate(segment):
        shot_score = shot_level[i]
        start = seg[0]
        end = seg[1]
        
        frame_level[start:end+1] = shot_score
    return frame_level
    

def plot(pred_score, gt_score, segment=None, summary=None, title=None):
    
    plt.figure(figsize=(120,60))
    plt.title(title)
    plt.plot(pred_score, c='r', label='predict', linewidth=5)
    plt.plot(gt_score, c='blue', label='GT', linewidth=5)
    
    if segment is not None:
        for seg in segment:
            start = seg[0]
            plt.plot((start,start), (0,1), '--', c='black')
    if summary is not None:
        for frame_index, is_summary in enumerate(summary):
            if is_summary:
                frame_score = pred_score[frame_index]
                plt.bar(frame_index, frame_score, width=1, color='springgreen')
    
    
    plt.show()
    
def greedy(segment_score, segment, total_length):
    n_segment = len(segment_score)
    nframe_table = {}
    for i, seg in enumerate(segment):
        start_frame = seg[0]
        end_frame = seg[1]
        seg_frames = end_frame - start_frame + 1
        
        nframe_table.update({i:seg_frames})
    sort = np.argsort(segment_score)
    
    total_frame = 0
    selected_shot = []
    for i in range(n_segment-1,0,-1):
        shot = sort[i]
        nframe_shot = nframe_table[shot]
        total_frame += nframe_shot
        
        if total_frame<total_length:
            selected_shot.append(shot)
        else:
            continue
        
    selected_shot = sorted(selected_shot)
    
    return selected_shot
    

if __name__ == '__main__':
    score_dir = '../save_score'
    dataset_path = '../datasets/cosum_goo3DRes_shot_center.h5'
    
    dataset = h5py.File(dataset_path, 'r')
    video_list = list(dataset.keys())
    
    for video_name in video_list:
        video_name = 'base3'
        pred_path = os.path.join(score_dir,f'{video_name}.npy')
        pred_score = np.load(pred_path, allow_pickle=True)
        
        video = dataset[video_name]
        summaries = video['summaries'][...]
        gt_score_frame = np.average(summaries,0)
        segment = video['segmentation'][...]
        total_frame = video['number_of_frames'][...]
        summary_length = int(0.15*total_frame)
        
        pred_score_frame = upsampling(pred_score, segment)
        
        selected_shot = greedy(pred_score, segment, summary_length)
        shot_summary = np.zeros(pred_score.shape[0])
        for shot_index in selected_shot:
            shot_summary[shot_index] = 1
        frame_summary = upsampling(shot_summary, segment)
    
        
        plot(pred_score_frame, gt_score_frame, segment, frame_summary, title=video_name)
        break
        
        
        
        
        
        
        
    

