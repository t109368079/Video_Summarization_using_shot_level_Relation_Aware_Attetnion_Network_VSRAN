from utils.metric import Metric

class LossMetric(Metric):

    def __init__(self):
        super().__init__()
        self.previous_min_epoch_mean = None
        return

    def is_min_epoch_mean_update(self):
        if self.previous_min_epoch_mean is None:
            self.previous_min_epoch_mean = self.get_epoch_mean(self.table.last_valid_index())
        
        if self.get_epoch_mean(self.table.last_valid_index()) < self.previous_min_epoch_mean:
            self.previous_min_epoch_mean = self.get_epoch_mean(self.table.last_valid_index())
            return True
        else:
            return False
        

    def get_current_status(self):
        is_min_epoch_mean_update = self.is_min_epoch_mean_update()
        return self.get_epoch_mean(self.table.last_valid_index()), is_min_epoch_mean_update

    def get_current_validate_status(self):
        
        return self.get_valid_epoch_mean(self.validtable.last_valid_index())

if __name__ == '__main__':
    pass
