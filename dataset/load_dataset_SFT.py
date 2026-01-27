import os
import pandas as pd

from datasets import Dataset, load_dataset, concatenate_datasets   

class SFTDataset:
    def __init__(self, cfg, tokenizer, custom_path):
        self.max_len = cfg["model"]["max_seq_length"]
        self.seed = cfg["dataset"]["seed"]
        self.JAQuAD_num = cfg["dataset"]["SFT_publish_size"]
        self.custom_num = cfg["dataset"]["SFT_custom_size"]
        self.train_ratio = cfg["dataset"]["train_ratio"]
        self.tokenizer = tokenizer
        self.custom_path = custom_path

    def load_data(self):
        # JAQuAD
        jaquad = load_dataset("softjapan/jaquad-sft")
        jaquad_all = concatenate_datasets([jaquad["train"], jaquad["validation"]])
        jaquad_all = jaquad_all.shuffle(seed=self.seed)
        jaquad_all = jaquad_all.select(range(min(self.JAQuAD_num, len(jaquad_all))))

        # 自作データ
        df = pd.read_excel(self.custom_path)
        df = df.fillna("")
        if "No" in df.columns:
            df = df.drop(columns=["No"])
        custom_ds = Dataset.from_pandas(df)
        custom_ds = custom_ds.shuffle(seed=self.seed)
        custom_ds = custom_ds.select(range(min(self.custom_num, len(custom_ds))))

        # トークナイズ
        jaquad_tok = jaquad_all.map(
            lambda x: self.tokenize_for_sft(x, self.build_prompt_JAQuAD),
            remove_columns=jaquad_all.column_names
        )
        custom_tok  = custom_ds.map(
            lambda x: self.tokenize_for_sft(x, self.build_prompt_custom),
            remove_columns=custom_ds.column_names
        )

        # JAQuAD と自作データを結合してシャッフル、train/val 分割
        full_dataset = concatenate_datasets([jaquad_tok, custom_tok])
        full_dataset = full_dataset.shuffle(seed=self.seed)
        split = full_dataset.train_test_split(test_size= 1 -self.train_ratio, seed=self.seed)
        
        return {"train": split["train"], "validation": split["test"]}

    # プロンプトとラベルを作成
    def build_prompt_JAQuAD(self, example):
        prompt = (
            "以下の参考情報から、質問に答えてください。\n\n"
            f"質問：{example['instruction']}\n"
            f"参考情報：{example['input']}\n\n"
            "回答："
        )
        answer = example["output"]
        return prompt, answer

    def build_prompt_custom(self, example):
        def safe(v):
            return "" if v is None else str(v)
        
        prompt = (
            "以下の参考情報から、質問に答えてください。\n\n"
            f"質問：{safe(example['Input'])}\n\n"
            "参考情報：\n"
        )

        for i in range(1, 6):
            src = safe(example.get(f"Source_{i}"))
            ctx = safe(example.get(f"Context_{i}"))
            if src or ctx:
                prompt += f"（参考{i}）{src}\n{ctx}\n\n"

        prompt += (
            f"回答の方針：{safe(example.get('Context_know-how'))}\n\n"
            "回答："
        )
        
        answer = example["Output"]
        return prompt, answer

    def tokenize_for_sft(self, example, build_fn):
        prompt, answer = build_fn(example)

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False
        )["input_ids"]

        answer_ids = self.tokenizer(
            answer + self.tokenizer.eos_token,
            add_special_tokens=False
        )["input_ids"]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids

        return {
            "input_ids": input_ids[:self.max_len],
            "labels": labels[:self.max_len]
        }

if __name__ == "__main__":
    from transformers import AutoTokenizer

    # 動作確認用
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    cfg = {
        "model": {"max_seq_length": 1024},
        "dataset": {
            "seed": 42,
            "SFT_publish_size": 100,
            "SFT_custom_size": 50,
            "train_ratio": 0.8
        }
    }
    custom_path = os.path.join(os.path.dirname(__file__), "ORG", "custom_data.xlsx")

    dataset_maker = SFTDataset(cfg, tokenizer, custom_path)
    dataset = dataset_maker.load_data()

    print(dataset["train"][0])
    print(dataset["validation"][0])