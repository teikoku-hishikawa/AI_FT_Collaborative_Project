import os
import wandb

#公開モジュールのインストール
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

#自作モジュールのインストール
from config.config_loader import load_config
from config.param_set import create_param_yaml
from dataset.dataset_maker import DatasetMaker
from dataset.load_dataset import load_and_tokenize_data, load_and_tokenize_SFT_data
from model.load_transformers import load_model_and_tokenizer, setup_peft
from utils.seed_fixed import set_seed

#configファイルのパラメータを読み込み
from anyParameters import model, peft, training, dataset

# Fine-tuning実行
class Finetuning:
    def __init__(self, paramdate, header_row=1):
        self.paramdate = paramdate
        self.header_row = header_row
        self.config_folder = os.path.join(os.path.dirname(__file__), "config", paramdate)
        self.dataset_folder = os.path.join(os.path.dirname(__file__), "dataset", paramdate)

    def main(self):
        #yamlファイル作成
        create_param_yaml(paramdate, model, peft, training, dataset)
        #yamlファイル参照
        config_files = os.listdir(self.config_folder)
        config_files = [i for i in config_files if i.endswith(".yaml")]

        for idx, config_file in enumerate(config_files, 1):
            #yamlファイルの読み込み
            config_path = os.path.join(self.config_folder, config_file)
            cfg = load_config(config_path)
            print(f"config_path: {config_path}")

            #データセット作成(既存のデータセットがあればそのまま使用)
            DatasetMakerSet = DatasetMaker(makedate=self.paramdate, header_row=self.header_row, train_ratio=cfg["dataset"]["train_ratio"], seed=cfg["dataset"]["seed"])
            DatasetMakerSet.main()
            dataset_path = DatasetMakerSet.dataset_dir

            #Fine-tuning実行(動作確認用にコメントアウト)
            self.training(cfg, dataset_path, idx)

    def training(self, cfg, dataset_path, train_num):
        # シード値の設定
        set_seed(cfg["training"]["seed"])

        # # モデル・Tokenizer
        # model, tokenizer = load_model_and_tokenizer(cfg=cfg)
        # model = setup_peft(model, cfg)

        # # データセット（例：独自データを準備して置き換え）
        # dataset = load_and_tokenize_data(cfg, tokenizer, dataset_path)

        # # データの出力先を確認
        # output_dir = os.path.join(cfg["training"]["output_dir"], f"{self.paramdate}/train{train_num}")
        # os.makedirs(output_dir, exist_ok=True)

        # # Wandbの記録名
        # run_name = self.run_name_set(cfg, train_num)
        # print(f"=== Wandb run name: {run_name} ===")

        # # TrainingArguments に YAML の値を渡す
        # training_args = TrainingArguments(
        #     output_dir=output_dir,
        #     per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        #     per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        #     num_train_epochs=cfg["training"]["num_train_epochs"],
        #     learning_rate=cfg["training"]["learning_rate"],
        #     logging_steps=cfg["training"]["logging_steps"],
        #     fp16=cfg["training"]["fp16"],
        #     eval_strategy=cfg["training"]["eval_strategy"],
        #     push_to_hub=cfg["training"]["push_to_hub"],
        #     eval_steps=cfg["training"]["eval_steps"],
        #     save_steps=cfg["training"]["save_steps"],
        #     save_total_limit=cfg["training"]["save_total_limit"],
        #     load_best_model_at_end=cfg["training"]["load_best_model_at_end"],
        #     metric_for_best_model=cfg["training"]["metric_for_best_model"],
        #     greater_is_better=cfg["training"]["greater_is_better"],
        #     report_to=["wandb"],  # ✅ Wandbに記録
        #     run_name=run_name,  # ✅ 実験名を設定
        # )

        # trainer = Trainer(
        #     model=model,
        #     args=training_args,
        #     train_dataset=dataset["train"],
        #     eval_dataset=dataset.get("validation", None),
        #     tokenizer=tokenizer,
        # )

        # trainer.train()
        # trainer.save_model(output_dir)

        # # wandbの終了
        # wandb.finish()

        # ===教師あり学習(指示チューニング)===
        # モデル・Tokenizer
        # SFTmodel_name =  trainer.state.best_model_checkpoint
        SFTmodel_name =  "./model/20251126/train1/checkpoint-5678"
        model, tokenizer = load_model_and_tokenizer(cfg, SFTmodel_name)
        model = setup_peft(model, cfg)

        # データセット（SFT用）
        dataset = load_and_tokenize_SFT_data(cfg, tokenizer)

        # データの出力先を確認
        # output_dir = os.path.join(output_dir, f"SFT")
        output_dir = os.path.join("./model/20251126/train1", f"SFT")
        os.makedirs(output_dir, exist_ok=True)

        # TrainingArguments に YAML の値を渡す
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
            per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
            num_train_epochs=cfg["training"]["num_train_epochs"],
            learning_rate=cfg["training"]["learning_rate"],
            logging_steps=cfg["training"]["logging_steps"],
            fp16=cfg["training"]["fp16"],
            eval_strategy=cfg["training"]["eval_strategy"],
            push_to_hub=cfg["training"]["push_to_hub"],
            eval_steps=cfg["training"]["eval_steps"],
            save_steps=cfg["training"]["save_steps"],
            save_total_limit=cfg["training"]["save_total_limit"],
            load_best_model_at_end=cfg["training"]["load_best_model_at_end"],
            metric_for_best_model=cfg["training"]["metric_for_best_model"],
            greater_is_better=cfg["training"]["greater_is_better"],
            report_to=["wandb"],  # ✅ Wandbに記録
            run_name=f"{run_name}_SFT",  # ✅ 実験名を設定
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation", None),
            tokenizer=tokenizer,
        )

        trainer.train()
        trainer.save_model(output_dir)

        # wandbの終了
        wandb.finish()


    def run_name_set(self, cfg, train_num):
        model_name = cfg["model"]["name"]
        print(f"モデル名: {model_name}")

        if "DeepSeek" in model_name:
            if "7B" in model_name:
                run_name = f"DeepSeek-R1-7B-Finetune-{self.paramdate}-train{train_num}"
            elif "14B" in model_name:
                run_name = f"DeepSeek-R1-14B-Finetune-{self.paramdate}-train{train_num}"
            elif "32B" in model_name:
                run_name = f"DeepSeek-R1-32B-Finetune-{self.paramdate}-train{train_num}"
        elif "Qwen2.5" in model_name:
            if "7B" in model_name:
                run_name = f"Qwen2.5-7B-Finetune-{self.paramdate}-train{train_num}"
            elif "14B" in model_name:
                run_name = f"Qwen2.5-14B-Finetune-{self.paramdate}-train{train_num}"
            elif "32B" in model_name:
                run_name = f"Qwen2.5-32B-Finetune-{self.paramdate}-train{train_num}"
        elif "gpt-oss" in model_name:
            if "20B" in model_name:
                run_name = f"gpt-oss-20B-Finetune-{self.paramdate}-train{train_num}"
            elif "120B" in model_name:
                run_name = f"gpt-oss-120B-Finetune-{self.paramdate}-train{train_num}"
        else:
            run_name = f"othermodel-Finetune-{self.paramdate}-train{train_num}"

        return run_name



if __name__ == "__main__":
    import datetime
    import transformers

    print(transformers.__file__)  # 出力結果：/home/teikoku/.conda/envs/.venv/lib/python3.11/site-packages/transformers/__init__.py
    print(transformers.__version__) # 出力結果：4.55.3 ← ver.4.5以上なら"evaluation_strategy"はあるらしい？

    #configファイルのパラメータを読み込み
    from anyParameters import model, peft, training
    #yamlファイルのパス(自動入力)
    # paramdate = datetime.datetime.now().strftime("%Y%m%d")
    paramdate = "20251126"
    #paramdate = "debug"
    Finetuning(paramdate=paramdate, header_row=5).main()
