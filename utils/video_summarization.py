import torch
import numpy as np
from scipy.stats import kendalltau as kendall
from scipy.stats import spearmanr as spearman
from sklearn.metrics import average_precision_score as mAP


from utils.knapsack import solve_knapsack

def preprocess(data, label, segmentation=None, summaries=None, key=None, user_score=None):
  data = torch.FloatTensor(data[0].numpy()), torch.FloatTensor(data[1].numpy())
  segmentation = segmentation.squeeze(0).numpy()
  summaries = summaries.squeeze(0).numpy()
  key = key[0] if key is not None else None
  label = torch.FloatTensor(label.numpy())
  label = apply_normalization(label)
  if user_score is not None:
      user_score = user_score.squeeze(0)
      user_score = user_score.detach().cpu().numpy()
  return data, label, segmentation, summaries, key, user_score

def aug_preprocess(data, label):
    data = torch.FloatTensor(data[0].numpy()), torch.FloatTensor(data[1].numpy())
    label = torch.FloatTensor(label.numpy())
    label = apply_normalization(label)
    return data, label



def get_segment_features(features, segmentation):
  upsampled_features = upsample_sequence(features)
  upsampled_features = torch.stack(upsampled_features)
  segment_features = []
  for i,segment in enumerate(segmentation):
      segment_features[i]=upsampled_features[segment[0]:segment[1] + 1]
  segment_features = torch.stack(segment_features)
  return segment_features


def get_segment_labels(labels, segmentation):
    upsampled_labels = upsample_sequence(labels)
    segment_labels = []
    for segment in segmentation:
        segment_labels.append(np.mean(upsampled_labels[segment[0]:segment[1] + 1], 0))
    segment_labels = torch.FloatTensor(segment_labels)
    return segment_labels


def upsample_sequence(sequence, rate = 16):
    upsampled_sequence = []
    for element in sequence:
        upsampled_sequence += [element] * rate
    return upsampled_sequence


def get_frame_scores(scores, segmentation):
    frame_scores = []
    
    for score, segment in zip(scores, segmentation):
        frame_scores += [score] * (segment[1] - segment[0] + 1)
    
    return frame_scores


def apply_normalization(sequence):
    if sequence.max() == 0:
        return sequence
    else:
        normalized_sequence = (sequence - sequence.min()) / sequence.max()
        return normalized_sequence


def generate_summary(frame_scores, segmentation, mode, proportion = 0.15):
    # score for each segment
    segment_scores = []
    for segment in segmentation:
        starting_index = segment[0]
        stopping_index = segment[1]
        segment_scores.append(np.mean(frame_scores[starting_index:stopping_index]))

    # number of frames for each segment
    segment_frames = [segment[1] - segment[0] + 1 for segment in segmentation]

    video_length = len(frame_scores)
    # print(video_length)
    summary_length = int(video_length * proportion)
    # print(segment_scores,segment_frames,summary_length)
    if mode == 'knapsack':
        selected_segment_indexes = solve_knapsack(segment_scores, [segment_frames], [summary_length])
    elif mode == 'greedy':
        selected_segment_indexes = greedy(segment_scores, segmentation, summary_length)
            
    summary = []
    for index in range(len(segment_frames)):
        if index in selected_segment_indexes:
            summary += [1] * segment_frames[index]
        else:
            summary += [0] * segment_frames[index]
    return summary

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

def generate_summary_shot(segment_score, segmentation, mode, proportion=0.15):
    segment_frames = [segment[1] - segment[0] + 1 for segment in segmentation]
    video_length = segmentation[-1][1]
    summary_length = int(video_length * proportion)
    
    if mode == 'knapsack':
        selected_segment_indexes = solve_knapsack(segment_score, [segment_frames], [summary_length])
    elif mode == "greedy":
        selected_segment_indexes = greedy(segment_score, segmentation, summary_length)
    else:
        raise AttributeError(f"Unrecognitize key shot selection argument:{mode}")
    
    summary = []
    for index in range(len(segment_frames)):
        if index in selected_segment_indexes:
            summary += [1] * segment_frames[index]
        else:
            summary += [0] * segment_frames[index]
    return summary

def evaluate_summary(summary, ground_truth_summaries, metric = 'average'):
    precisions = []
    recalls = []
    f1_scores = []
    # print([each for each in ground_truth_summaries[0]])
    # print(summary)
    if len(ground_truth_summaries.shape) == 1:
        ground_truth_summaries = np.reshape(ground_truth_summaries, (1, ground_truth_summaries.shape[0]))
    if len(summary) != len(ground_truth_summaries[0]):
      summary = summary[:-1]
    for ground_truth_summary in ground_truth_summaries:
        summary_length = np.sum(summary)
        ground_truth_summary_length = np.sum(ground_truth_summary)
        overlapping_length = np.sum(summary * ground_truth_summary)
        # print(summary_length,ground_truth_summary_length,overlapping_length)
        precision, recall, f1_score = get_performance(overlapping_length, summary_length, ground_truth_summary_length)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    if metric == 'average':
        return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)
    else:
        max_f1_score_index = np.argmax(f1_scores)
        max_precision = precisions[max_f1_score_index]
        max_recall = recalls[max_f1_score_index]
        max_f1_score = f1_scores[max_f1_score_index]
        return max_precision, max_recall, max_f1_score
    
def get_propotion(summaries):
    total_frame = summaries.shape[1]
    rate = 0
    for i in range(summaries.shape[0]):
        rate += np.sum(summaries[i])/total_frame
    rate /= summaries.shape[0]
    return rate

def get_performance(true_positive, predicted_positive, positive):
    precision = true_positive / predicted_positive
    recall = true_positive / positive
    f1_score = 0 if precision == 0 and recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score


def read_tvsum_score(path):
    
    with open(path, 'r') as f:
        lines = f.readlines()
    score_dict = {}
    for line in lines:
        video_name, _, score = line.split('\t')
        score = score.replace('\n','')
        score = [int(s) for s in score.split(',')]
        
        if video_name not in list(score_dict.keys()):
            score_dict.update({video_name:[]})
        score_dict[video_name].append(np.array(score))
    
    return score_dict

def read_index_table(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    table = {}
    for line in lines:
        index, name = line.split('\t')
        name = name.replace('\n','')
        
        table.update({name: f'video_{index}'})
    return table

def read_name_table(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    table = {}
    for line in lines:
        index, name = line.split('\t')
        name = name.replace('\n', '')
        table.update({f'video_{index}': name})
    
    return table

def create_shot_summary(frame_summary, segment):
    shot_summary = np.zeros(segment.shape[0])
    for i, seg in enumerate(segment):
        start = seg[0]
        if frame_summary[start] == 1:
            shot_summary[i] = 1
        else:
            pass
    return shot_summary
            

def mean_average_precision(pred_score, frame_summaries, segment, top=None):
    n_segment = pred_score.shape[0]
    # 正規化
    pred_score -= pred_score.min()
    pred_score /= pred_score.max()
    
    # 排序
    sorted_index = np.argsort(pred_score)
    
    # 選top個
    if top is not None:
        try:
            top_index = [sorted_index[i] for i in range(n_segment-1, n_segment-top-1, -1)]
        except:
            print('top value excess number of shot')
            top_index = sorted_index
    else:
        top_index = sorted_index
    
    # 算mAP
    map_score = 0
    top_score = pred_score[top_index]
    n_user = frame_summaries.shape[0]
    for user_summary in frame_summaries:
        shot_summary = create_shot_summary(user_summary, segment)
        top_summary = shot_summary[top_index]
        score = mAP(top_summary, top_score)
        score = 0 if np.isnan(score) else score
        map_score += score
    map_score /= n_user
    return map_score

def kendall_spearman(pred_shot, user_scores, segment):
    n_frame = segment[-1][1]
    pred_frame = np.zeros(n_frame)
    
    for i, seg in enumerate(segment):
        pred_score = pred_shot[i]
        start_frame = seg[0]
        end_frame = seg[1]
        
        pred_frame[start_frame: end_frame] = pred_score
    
    average_kendall = 0
    average_spearman = 0 
    n_user = user_scores.shape[0]
    for user_score in user_scores:
        user_score = user_score[:n_frame]
        kendall_coeff, _ = kendall(pred_frame, user_score)
        spearman_coeff, _ = spearman(pred_frame, user_score)
        
        average_kendall += kendall_coeff
        average_spearman += spearman_coeff
    average_kendall /= n_user
    average_spearman /= n_user
    return average_kendall, average_spearman
    
    
        

if __name__ == '__main__':
    pass
