import os
import torch
import json
import re
import csv
import sys
import argparse

from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config.config_loader import load_config

# 任意モデルの動作テスト
class ModelTest:
    def __init__(self):
        # コマンドライン引数の解析
        parser = argparse.ArgumentParser(description="LLM動作テスト")
        parser.add_argument("--input_mode", choices=["interactive", "batch"],
                            default="interactive",
                            help="interactive: 任意入力 / batch: JSONL既定入力")
        parser.add_argument("--model_mode", choices=["default", "trained"],
                            default="default",
                            help="default: 既定モデル / trained: Fine-tuning済みモデル")
        parser.add_argument("--paramdate", type=str,
                            default=None,
                            help="動作テストパラメータ日付(YYYYMMDD形式（未指定時は最新日付）)")
        parser.add_argument("--config_num", type=int,
                            default=None,
                            help="動作テストパラメータ番号（複数設定時）")
        parser.add_argument("--model_name", type=str,
                            default=None,
                            help="動作テストモデル名（未指定時は最新モデル）")
        parser.add_argument("--jsonl_path", type=str,
                            default=os.path.join(os.path.dirname(__file__), "test", "prompt_set.jsonl"),
                            help="既定入力用JSONLファイル（batchモード時必須）")
        args = parser.parse_args()
        
        # 引数の保存
        self.input_mode = args.input_mode
        self.model_mode = args.model_mode
        self.paramdate = args.paramdate
        self.config_num = args.config_num
        self.model_name = args.model_name
        self.jsonl_path = args.jsonl_path

        # 引数の妥当性確認
        print("\n=== 引数の妥当性確認 ===")
        # paramdateの形式チェック
        if self.paramdate is not None:
            if not re.fullmatch(r"\d{8}", self.paramdate):
                print("paramdateはYYYYMMDD形式で指定してください。")
                sys.exit(1)
        else:
            print("paramdateが指定されていません。最新のパラメータ日付を自動選択します。")

        # config_numの形式チェック
        if self.config_num is not None:
            if not isinstance(self.config_num, int) or self.config_num < 1:
                print("config_numは1以上の整数で指定してください。")
                sys.exit(1)
        else:
            self.config_num = 1
            print("config_numが指定されていません。1を自動設定します。")

        # JSONLファイルの存在確認（batchモード時）
        if self.input_mode == "batch" and not os.path.isfile(self.jsonl_path):
            print(f"指定されたJSONLファイルが存在しません: {self.jsonl_path}")
            sys.exit(1)

        print("引数の妥当性確認が完了しました。\n")

    def main(self):
        # パラメータファイルとモデルパスの取得
        print("=== パラメータファイルとモデルパスの取得 ===")
        config_path, model_path, model_name = self.load_pathset()
        cfg = load_config(config_path)

        if self.model_mode == "default":
            model_path = cfg["model"]["name"]
            model_name = cfg["model"]["name"]
            if "/" in model_name:
                model_name = model_name.split("/")[-1]
        
        print(f"使用モデルパス/名前: {model_name}\n")
        print("パラメータファイルとモデルパスの取得が完了しました。\n")

        # QLoRA の場合は 4bit でロードする必要があるので強制
        load_in_4bit = bool(cfg["model"].get("load_in_4bit", False))
        if cfg["peft"]["training_mode"].lower() == "qlora":
            load_in_4bit = True

        # モデル・トークナイザーの読み込み
        print("=== モデル・トークナイザーの読み込み ===")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer = self.load_model(model_path, device=device, load_in_4bit=load_in_4bit)

        # === CSV出力先設定 ===
        print("=== モデルテスト開始 ===")
        csv_path = self.get_unique_csv_path(model_name)
        print(f"\n出力結果は CSV にも保存されます → {csv_path}\n")
        records = []

        # 動作テストモードの選択
        # interactiveモード(任意入力)
        if self.input_mode == "interactive":
            while True:
                user_input = input("質問を入力してください（終了するには 'exit' と入力）： ")
                if user_input.lower() == "exit":
                    break
                # モデルに入力して応答を取得
                result = self.generate_text(
                    model, 
                    tokenizer, 
                    user_input, 
                    max_length=cfg["model"].get("max_seq_length", 1024),
                    device=device
                )
                
                # 結果の表示
                print("\n--- 出力結果 ---")
                print(result)
                print("----------------\n")

                # CSV保存
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                records.append([timestamp, user_input, result])
                self.save_to_csv(csv_path, [[timestamp, user_input, result]])

        # batchモード(JSONL既定入力)
        elif self.input_mode == "batch":
            inputs_list = load_jsonl(self.jsonl_path)
            print(f"=== バッチモード（{len(inputs_list)}件） ===")
            for idx, user_input in enumerate(inputs_list, 1):
                print(f"[{idx}] User > {user_input}")
                # モデルに入力して応答を取得
                result = self.generate_text(
                    model, 
                    tokenizer,  
                    user_input, 
                    max_length=cfg["model"].get("max_seq_length", 1024),
                    device=device
                )
                # 結果の後処理（入力文を削除）
                if result.startswith(user_input):
                    result = result[len(user_input):].strip()
                
                # 結果の表示
                print("Model >", result)
                print("-"*40)

                # CSV保存
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                records.append([timestamp, user_input, result])
                self.save_to_csv(csv_path, [[timestamp, user_input, result]])
    
    def generate_text(self, model, tokenizer, prompt, max_length=1024, device="cuda"):
        # 入力をトークナイズ
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # モデルで生成
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )

        # トークンを文字列に変換
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text

    #日付フォルダかどうかチェック(自動選択用)
    def is_valid_date_folder(self, name: str) -> bool:
        """フォルダ名がYYYYMMDD形式かをチェック"""
        return bool(re.fullmatch(r"\d{8}", name))
    
    # パラメータとモデルパスの取得
    def load_pathset(self):
        # configフォルダ内のパラメータ日付を取得する
        config_path = os.path.join(os.path.dirname(__file__), "config")
        
        # 動作実行日を確認
        if self.paramdate is not None:
            paramdate = self.paramdate
            print(f"指定されたパラメータ日付: {paramdate}")
        else:
            paramdates = [d for d in os.listdir(config_path) if os.path.isdir(os.path.join(config_path, d))]
            paramdates = sorted([d for d in paramdates if self.is_valid_date_folder(d)])
            if not paramdates:
                print("configフォルダ内にパラメータ日付のフォルダが存在しません。")
                sys.exit(1)
            paramdate = paramdates[-1]
            print(f"最新のパラメータ日付を自動選択: {paramdate}")

        # configファイルの読み込み確認
        config_path = os.path.join(config_path, paramdate, f"param_set_{self.config_num}.yaml")
        if not os.path.isfile(config_path):
            print(f"指定されたパラメータファイルが存在しません: {config_path}")
            sys.exit(1)

        # モデルパスの設定
        model_path = os.path.join(os.path.dirname(__file__), "model", paramdate, f"train{self.config_num}")
        if self.model_name is not None:
            model_name = self.model_name
            print(f"指定されたモデル名: {model_name}")
            model_path = os.path.join(model_path, model_name)
            if not os.path.isdir(model_path):
                print(f"指定されたモデルフォルダが存在しません: {model_path}")
                sys.exit(1)
        else:
            model_names = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
            model_names = sorted([d for d in model_names])
            for model_dir in model_names:
                if model_dir == "SFT":
                    model_name = model_dir
            if model_name == "":
                model_name = model_names[-1]
            print(f"最新のモデル名を自動選択: {model_name}")

        return config_path, model_path, f"{paramdate}_train{self.config_num}_{model_name}"

    # トークナイザーとモデルの読み込み
    def load_model(self, model_path_or_name, device="cpu", load_in_4bit=False):
        """モデルをロードする"""
        print(f"\nモデルをロード中: {model_path_or_name}")
        print(f"使用デバイス: {device}\n")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path_or_name,
            torch_dtype="auto",
            device_map="auto"
            )
        print("トークナイザーをロードしました。")

        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_name,
            torch_dtype="auto",
            device_map="auto",
            load_in_4bit=load_in_4bit
        )
        print("モデルをロードしました。")
        
        # tokenizer.to(device)
        model.to(device)
        model.eval()

        return model, tokenizer

    # csvファイルのユニークパス取得
    def get_unique_csv_path(self, model_name: str) -> str:
        # 保存ディレクトリ
        output_dir = os.path.join(os.path.dirname(__file__), "test")
        os.makedirs(output_dir, exist_ok=True)

        # ベースファイル名
        base_filename = f"LLMtest_{model_name}_results"
        ext = ".csv"
        csv_path = os.path.join(output_dir, base_filename + ext)

        # 同名ファイルが存在する場合、連番を付ける
        counter = 1
        while os.path.exists(csv_path):
            csv_path = os.path.join(output_dir, f"{base_filename}_{counter}{ext}")
            counter += 1

        return csv_path

    # 結果をCSVに保存
    def save_to_csv(self, csv_path, records):
        """結果をCSVに保存"""
        header = ["timestamp", "input_text", "generated_text"]
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerows(records)

if __name__ == "__main__":
    ModelTest().main()