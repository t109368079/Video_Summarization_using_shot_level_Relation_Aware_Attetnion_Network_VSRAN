import pandas as pd

class Metric():

    def __init__(self):
        self.table = pd.DataFrame()
        self.validtable = pd.DataFrame()
        return


    def update(self, epoch, key, value):
        self.table.loc[epoch, key] = value
        return

    def update_valid(self, epoch, key, value):
        self.validtable.loc[epoch, key] = value
        return

    # find the mean value for each epoch and get the max one
    def get_max_epoch_mean(self):
        return self.table.mean(axis = 1).max()
    
    def get_max_epoch_std(self):
        max_epoch = self.table.mean(axis=1).argmax()
        return self.table.std(axis=1)[max_epoch]
    
    # 找最小，保留給MSE用
    def get_min_epoch_mean(self):
        return self.table.mean(axis=1).min()


    # find the mean value for each video and get the max one
    def get_max_video_mean(self):
        return self.table.mean(axis = 0).max()


    # find the mean value for each epoch and get the specific one
    def get_epoch_mean(self, epoch):
        return self.table.mean(axis = 1)[epoch]
    
    def get_epoch_std(self, epoch):
        return self.table.std(axis = 1)[epoch]

    def get_valid_epoch_mean(self, epoch):
        return self.validtable.mean(axis = 1)[epoch]

    def get_epoch_means(self):
        return self.table.mean(axis = 1)

    def get_valid_epoch_means(self):
        return self.validtable.mean(axis = 1)

    # find the mean value for each video and get the specific one
    def get_video_mean(self, video):
        return self.table.mean(axis = 0)[video]


    def get_video_means(self):
        return self.table.mean(axis = 0)


    def get_epoch(self, epoch):
        return self.table.iloc[epoch]


    def get_status(self):
        raise NotImplementedError


if __name__ == '__main__':
    pass
