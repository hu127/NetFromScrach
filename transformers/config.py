import yaml
from pathlib import Path

class Configs:
    def __init__(self, config_path):
        with open(config_path, 'r') as stream:
            self.configs = yaml.safe_load(stream)

    def get_config(self, key):
        return self.configs[key]

    def get_all_configs(self):
        return self.configs
    
def get_latest_weight(configs):
    model_folder = f"{configs['data_source']}_{configs['model_folder']}"
    model_filename = configs['model_basename']
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return weights_files[-1]

def get_weight_file_path(configs, epoch):
    model_folder = f"{configs['data_source']}_{configs['model_folder']}"
    model_filename = configs['model_basename']
    return Path(model_folder) / f"{model_filename}_{epoch}.pt"


# config_path = 'configs/config.yaml'
# configs = Configs(config_path)
# print(configs.get_all_configs())
# print(configs.get_config('lang_src'))