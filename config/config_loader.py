import yaml

# -------------------
# YAML を読み込む関数
# -------------------
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    import os
    #yamlファイルのパス
    config_path = os.path.join(os.path.dirname(__file__), "debug", "train_param.yaml")
    #yamlファイルの読み込み
    cfg = load_config(config_path)
    #読み込んだ内容の確認
    for param1 in cfg:
        for param2 in cfg[param1]:
            cfg_value = cfg[param1][param2]
            print(f"{param1} - {param2}: {cfg_value}")