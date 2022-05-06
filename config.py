import yaml
import sys

class Config(object):
    def __init__(self):
        configFileName = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
        with open(configFileName, "r") as f:
            self.__data = yaml.safe_load(f)
    def __call__(self):
        return self.__data

if __name__ == "__main__":
    c = Config()
    print(c())