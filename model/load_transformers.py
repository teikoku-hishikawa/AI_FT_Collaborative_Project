import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# PEFT のバージョンにより関数名が異なる場合があるので安全にインポート
try:
    from peft import prepare_model_for_kbit_training
except Exception:
    prepare_model_for_kbit_training = None  # なければ None にしてフォールバックする

# -------------------
# モデルとトークナイザーの読み込み
# -------------------
def load_model_and_tokenizer(cfg: dict):
    model_name = cfg["model"]["name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT系モデルなどではpadding tokenが未定義の場合があるので定義
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # QLoRA の場合は 4bit でロードする必要があるので強制
    load_in_4bit = bool(cfg["model"].get("load_in_4bit", False))
    if cfg["peft"]["training_mode"].lower() == "qlora":
        load_in_4bit = True

    # QLoRA or FullFineTuning を切り替え
    if cfg["peft"]["training_mode"].lower() in ["qlora", "lora"]:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, cfg["model"]["dtype"]),
            load_in_4bit=load_in_4bit,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:  # FullFineTuning
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, cfg["model"]["dtype"]),
            device_map="auto",
        )

    return model, tokenizer

# -------------------
# 学習モードに応じたモデル準備（Full / LoRA / QLoRA 切替）
# -------------------
def setup_peft(model, cfg: dict):
    
    # 学習方針を確認（存在しない場合は qlora を使う）
    mode = cfg["peft"].get("training_mode", "qlora")

    # lora,qloraの場合modelを変更
    if mode.lower() in ["lora", "qlora"]:
        peft_cfg = LoraConfig(
            r=cfg["peft"]["r"],
            lora_alpha=cfg["peft"]["lora_alpha"],
            lora_dropout=cfg["peft"]["lora_dropout"],
            target_modules=cfg["peft"]["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)

    return model

