#configファイルのパラメータ（手動）
model = {
    "name":["Qwen/Qwen2.5-7B-Instruct"], # モデル名("Qwen/Qwen2.5-14B-Instruct", "cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese", "openai/gpt-oss-20b")
    "max_seq_length":[2048], # 最大シーケンス長
    "dtype":["float16"], # データ型（例：float16, float32, bfloat16）
    "load_in_4bit":[True] # 4bit量子化で読み込むか (True or False)
}

peft = {
    "training_mode":["qlora"], # fine-tuningのモード（"full" or "lora" or "qlora"）
    "r":[8], # LoRAのランク
    "lora_alpha":[16], # LoRAのスケーリングファクター
    "lora_dropout":[0], # LoRAのドロップアウト率 
    "target_modules":[
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    ] # LoRAを適用するモジュール
}

training = {
    "seed":[42], # 乱数シード
    "output_dir":["./model"], # モデルの保存先
    "num_train_epochs":[1], # エポック数
    "per_device_train_batch_size":[2], # trainバッチサイズ
    "per_device_eval_batch_size":[2], # evalバッチサイズ
    "learning_rate":[2e-4], # 学習率
    "logging_steps":[50], # ロギングのステップ数
    "eval_strategy":["steps"], # 評価の頻度 ("no", "steps", "epoch")
    "fp16":[True], # 16ビット浮動小数点精度で学習するか
    "push_to_hub":[False], # Hugging Face Hubにモデルをプッシュするか
    "eval_steps":[200],                     # 評価のステップ間隔
    "save_steps":[200],                     # 保存のステップ間隔
    "save_total_limit":[3],                 # 保存するモデルの最大数（古いものから削除）
    "load_best_model_at_end":[True],        # ✅ トレーニング完了時にベストモデルを読み込む
    "metric_for_best_model":["eval_loss"],  # ✅ 評価指標を指定（例: "accuracy", "eval_loss" など）
    "greater_is_better":[False],            # ✅ 指標が小さいほど良い場合（例: loss）
}

dataset = {
    "seed":[42], # 乱数シード
    "train_ratio":[0.8], # 訓練データの割合（例：0.8は80%を訓練、20%を検証に使用）
    "SFT_train_size":[100], # SFT用訓練データサイズ
    "SFT_valid_size":[20], # SFT用検証データサイズ
}