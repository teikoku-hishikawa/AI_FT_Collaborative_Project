from datasets import load_dataset

# -----------------------------
# データセットの読み込み
# -----------------------------
def load_and_tokenize_data(cfg, tokenizer, dataset_path, SFTbool=False):
    
    dataset_param = {
        "name": "json",
        "data_files":{
            "train": f"{dataset_path}/train.jsonl",
            "validation": f"{dataset_path}/valid.jsonl",
        },
        "columns":{
            "input": "input",
            "output": "output",
        },
    }
    
    ds = load_dataset(
        dataset_param["name"],
        data_files=dataset_param["data_files"],
        split=None
    )

    def tokenize_fn(example):
        if "input" in example and "output" in example:
            # Alpaca形式
            prompt = example["input"]
            output = example["output"]
            if prompt is None: prompt = ""
            if output is None: output = ""
            full_text = prompt + output

            tokenized = tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=cfg["model"]["max_seq_length"],
            )

            # output 部分のみ学習するようにラベル設定(教師あり学習)
            if SFTbool:
                prompt_len = len(tokenizer(prompt).input_ids)
                labels = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]
                labels = labels[: cfg["model"]["max_seq_length"]]
                tokenized["labels"] = labels

        elif "text" in example:
            # text-only形式
            text = example["text"]
            if text is None: text = ""
            tokenized = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=cfg["model"]["max_seq_length"],
            )
            
            # 全部学習対象(教師あり学習)
            if SFTbool:
                tokenized["labels"] = tokenized["input_ids"]  # 全部学習対象

        else:
            # input/output/text がない場合はスキップ
            return None

        return tokenized

    tokenized_ds = ds.map(tokenize_fn, batched=False)
    return tokenized_ds