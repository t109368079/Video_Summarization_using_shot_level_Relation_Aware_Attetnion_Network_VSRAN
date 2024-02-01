from torch.utils.data import Dataset

class VideoSummarizationDataset(Dataset):

    def __init__(self, static_dataset, motion_dataset,mode):
        self.static_dataset = static_dataset
        self.motion_dataset = motion_dataset
        self.keys = list(self.static_dataset.keys())
        self.mode = mode
        return


    # def __getitem__(self, index):
    #     key = self.keys[index]
    #     static_video = self.static_dataset[key]
    #     motion_video = self.motion_dataset[key]
    #     data = static_video['features'].value, motion_video['features'].value
    #     label = static_video['annotations_mean'].value
    #     segmentation = static_video['segmentation'].value
    #     summaries = static_video['summaries'].value
    #     return data, label, segmentation, summaries, key
    def __getitem__(self, index):
        key = self.keys[index]
        static_video = self.static_dataset[key]
        motion_video = self.motion_dataset[key]
        indexes=[]
        if self.mode == 'fusion':
          data = static_video['features'][:], motion_video['features'][:]
          indexes = motion_video['segment_feature_index'][:]
        if self.mode == 'outlier':
          data = static_video['selected_clip_mean_features'][:], motion_video['selected_clip_mean_features'][:]
        if self.mode == 'origin':
          data = static_video['clip_mean_features'][:], motion_video['clip_mean_features'][:]
        if self.mode == 'outlier_fusion':
          data= static_video['selected_clip_nonmean_features'][:],motion_video['selected_clip_nonmean_features'][:]
          indexes = static_video['selected_segment_feature_index'][:],motion_video['selected_segment_feature_index'][:]
        if self.mode == 'split_fusion':
          data= static_video['kmeans_selected_clip_nonmean_features'][:],motion_video['kmeans_selected_clip_nonmean_features'][:],static_video['kmeans_clip_outlier_nonmean_features'][:],motion_video['kmeans_clip_outlier_nonmean_features'][:]
          indexes = static_video['kmeans_selected_segment_feature_index'][:],motion_video['kmeans_selected_segment_feature_index'][:],static_video['kmeans_clip_outlier_feature_index'][:],motion_video['kmeans_clip_outlier_feature_index'][:]
        label = static_video['segment_annotation'][:]
        segmentation = static_video['segmentation'][:]
        summaries = static_video['summaries'][:]
        return data, label, segmentation, summaries, key, indexes

    def __len__(self):
        return len(self.keys)


if __name__ == '__main__':
    pass
