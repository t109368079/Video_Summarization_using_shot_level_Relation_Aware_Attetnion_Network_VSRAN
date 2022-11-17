from utils.metric import Metric

class F1ScoreMetric(Metric):

    def __init__(self):
        super().__init__()
        self.previous_max_epoch_mean = None
        self.previous_max_epoch_std = None
        return


    def is_max_epoch_mean_updated(self):
        max_epoch_mean = self.get_max_epoch_mean()
        if self.previous_max_epoch_mean is None or max_epoch_mean > self.previous_max_epoch_mean:
            self.previous_max_epoch_std = self.get_max_epoch_std()
            self.previous_max_epoch_mean = max_epoch_mean
            return True
        else:
            return False


    def get_current_status(self):
        is_max_epoch_mean_updated = self.is_max_epoch_mean_updated()
        return self.get_epoch_mean(self.table.last_valid_index()), self.previous_max_epoch_mean, self.get_epoch_std(self.table.last_valid_index()) , is_max_epoch_mean_updated


if __name__ == '__main__':
    pass
