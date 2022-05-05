import argparse

from Trainer import Trainer

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Configuration File Path')
    parser.add_argument(
        "-c", "--conf", action="store", dest="conf_file",
        help="Path to config file"
    )
	args = parser.parse_args()
	conf_path =  args.conf_file
	if conf_path is None:
		conf_path = 'configs/config.yaml'
	Trainer(conf_path).train()
