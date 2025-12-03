import random
from datasets import load_dataset
from transformers import AutoTokenizer

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
            # context = ""
            # full_text =  f"Context:\n{context}\nQuestion:\n{prompt}\nAnswer:\n{output}"


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
            else:
                tokenized["labels"] = tokenized["input_ids"][:]  # 全部学習対象

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
            tokenized["labels"] = tokenized["input_ids"]  # 全部学習対象

        else:
            # input/output/text がない場合はスキップ
            return None

        return tokenized

    tokenized_ds = ds.map(tokenize_fn, batched=False)
    return tokenized_ds


#支持チューニング用データセット読み込み関数
def load_and_tokenize_SFT_data(cfg, tokenizer):
    # 乱数シード設定
    random.seed(cfg["dataset"]["seed"])
    print(f"=== Loading and tokenizing SFT dataset {cfg['dataset']['SFT_train_size'] + cfg['dataset']['SFT_valid_size']}個 ===")
    
    # データセット(Natural Questions)読み込み
    dataset = load_dataset(
        "natural_questions", 
        "default", 
        split=f"train[:{cfg['dataset']['SFT_train_size'] + cfg['dataset']['SFT_valid_size']}]"
    )

    # 一度だけシャッフルしてから train/val を分割
    shuffled = dataset.shuffle(seed=cfg["dataset"]["seed"])
    train_ds_all = shuffled.select(range(cfg["dataset"]["SFT_train_size"]))
    valid_ds_all = shuffled.select(range(cfg["dataset"]["SFT_train_size"],
                                         cfg["dataset"]["SFT_train_size"] + cfg["dataset"]["SFT_valid_size"]))

    max_len = cfg["model"]["max_seq_length"]

    def tokenize_nq(example):
        question = example["question_text"]
        context = example["document_text"]

        # answer抽出
        annotations = example.get("annotations", [])
        if annotations and annotations[0].get("short_answers"):
            answer = annotations[0]["short_answers"][0]["text"][0]
        else:
            answer = ""

        full_prompt = f"Context:\n{context}\nQuestion:\n{question}\nAnswer:\n{answer}"

        # 一度のトークナイズで済ませる
        tokenized_prompt = tokenizer(
            full_prompt,
            truncation=True,
            max_length=max_len,
            padding="max_length"
        )

        # 回答部分のみ別途トークナイズ
        answer_ids = tokenizer(
            answer,
            truncation=True,
            max_length=max_len,
            add_special_tokens=False
        ).input_ids

        # labels を input_ids と同じ長さで作る
        labels = [-100] * max_len

        # prompt のトークン数
        prompt_len = sum(1 for id in tokenized_prompt["input_ids"] if id != tokenizer.pad_token_id)

        # answer を prompt の末尾に貼る（はみ出したら切る）
        for i, tid in enumerate(answer_ids):
            pos = prompt_len + i
            if pos < max_len:
                labels[pos] = tid
            else:
                break

        tokenized_prompt["labels"] = labels

        return tokenized_prompt

    # map
    tokenized_train = train_ds_all.map(tokenize_nq, batched=False)
    tokenized_valid = valid_ds_all.map(tokenize_nq, batched=False)
    tokenized_ds = {
        "train": tokenized_train,
        "validation": tokenized_valid
    }
    return tokenized_ds