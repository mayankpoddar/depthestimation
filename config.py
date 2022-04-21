import yaml

class Config(object):
    def __init__(self):
        with open("config.yaml", "r") as f:
            self.__data = yaml.safe_load(f)
    def __call__(self):
        return self.__data
