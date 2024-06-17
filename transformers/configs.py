import yaml

class Configs:
    def __init__(self, config_path):
        with open(config_path, 'r') as stream:
            self.configs = yaml.safe_load(stream)

    def get_config(self, key):
        return self.configs[key]

    def get_all_configs(self):
        return self.configs

config_path = 'configs/config.yaml'
configs = Configs(config_path)
print(configs.get_all_configs())
print(configs.get_config('lang_src'))