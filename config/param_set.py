import os
import yaml
import itertools

# --- FlowStyleList を定義し、SafeDumper にフロースタイルの表現器を登録 ---
class FlowStyleList(list):
    """このクラスのインスタンスは YAML 出力時に [a, b, c] の形式（flow style）になる"""
    pass

def flow_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.SafeDumper.add_representer(FlowStyleList, flow_list_representer)


# --- ユーティリティ：各値を必ずリストに正規化（単体なら [単体] にする） ---
def normalize_param_dict(d):
    norm = {}
    for k, v in d.items():
        if not isinstance(v, list):
            norm[k] = [v]
        else:
            # v がリスト（中身がさらにリスト＝ネストリストの場合、ネストを維持する）
            norm[k] = v
    return norm



# パラメータyaml作成
def create_param_yaml(paramdate, model, peft, training, dataset, flow_keys=("target_modules",)):
    
    # 保存ディレクトリ作成
    param_dir = os.path.join(os.path.dirname(__file__), paramdate)
    os.makedirs(param_dir, exist_ok=True)

    # 正規化
    model_n = normalize_param_dict(model)
    peft_n = normalize_param_dict(peft)
    training_n = normalize_param_dict(training)
    dataset_n = normalize_param_dict(dataset)

    # 組み合わせ（デカルト積）を作る
    def build_combinations(norm_dict):
        keys = list(norm_dict.keys())
        vals = [norm_dict[k] for k in keys]
        combos = list(itertools.product(*vals))
        return keys, combos

    model_keys, model_combos = build_combinations(model_n)
    peft_keys, peft_combos = build_combinations(peft_n)
    training_keys, training_combos = build_combinations(training_n)
    dataset_keys, dataset_combos = build_combinations(dataset_n)

    set_index = 1
    for m in model_combos:
        for p in peft_combos:
            for t in training_combos:
                for d in dataset_combos:
                    # 各セクションのキーと組み合わせから辞書を作成
                    params = {
                        "model": dict(zip(model_keys, m)),
                        "peft": dict(zip(peft_keys, p)),
                        "training": dict(zip(training_keys, t)),
                        "dataset": dict(zip(dataset_keys, d)),
                    }

                    # flow_keys に含まれるキー（例：target_modules）が peft/model/training/dataset にあれば
                    # その値が list の場合 FlowStyleList でラップする
                    for section in ("model", "peft", "training", "dataset"):
                        for k in list(params[section].keys()):
                            if k in flow_keys and isinstance(params[section][k], list):
                                # すでにネスト（[[...]]）になっている場合は内側を取ることがあるため、
                                # もし要素が1つでその要素が list なら内側を使う
                                v = params[section][k]
                                if len(v) == 1 and isinstance(v[0], list):
                                    v = v[0]
                                # FlowStyleList にラップ
                                params[section][k] = FlowStyleList(v)

                    # ファイル名を param_set_{index}.yaml にして保存
                    param_path = os.path.join(param_dir, f"param_set_{set_index}.yaml")
                    with open(param_path, "w", encoding="utf-8") as f:
                        yaml.dump(params, f, allow_unicode=True, sort_keys=False, Dumper=yaml.SafeDumper)
                    print(f"✅ YAML created: {param_path}")
                    set_index += 1

if __name__ == "__main__":
    #import datetime
    #paramdate = datetime.datetime.now().strftime("%Y%m%d")
    from anyParameters import model, peft, training, dataset
    paramdate = "debug" # デバッグ用に固定
    print(f"Creating parameter YAML files in directory: config/{paramdate}")
    create_param_yaml(paramdate, model, peft, training, dataset)