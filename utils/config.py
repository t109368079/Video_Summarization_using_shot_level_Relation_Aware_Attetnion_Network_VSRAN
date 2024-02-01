import yaml

class ConfigParser(object):

    def __init__(self, config_path):
        self.config = self._initialize_config(config_path)
        return


    def _initialize_config(self, config_path):
        config_file = open(config_path, 'r', encoding = 'utf-8')
        config = yaml.load(config_file, Loader = yaml.FullLoader)
        return config


    def __getattr__(self, attribute):
        if attribute in self.config:
            return self.config[attribute]
        else:
            return None


if __name__ == '__main__':
    pass
