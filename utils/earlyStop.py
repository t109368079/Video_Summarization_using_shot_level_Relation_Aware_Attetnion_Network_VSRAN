
class EarlyStopping:
    """
    When calling early stop.
    Check if current metric below best loss at lease delta level
    Return True when metric stop decreasing
    """

    def __init__(self, delta, patient):
        self.delta = delta
        self.patient = patient
        self.bsf_metric = None
        self.counter = 0

    def __call__(self, metric):
        if self.bsf_metric is None:
            self.bsf_metric = metric
        if self.bsf_metric - metric >=  self.delta*self.bsf_metric:
            self.bsf_metric = metric
            self.counter = 0
        else:
            if self.counter > self.patient:
                return True
            else:
                self.counter += 1
        return False

