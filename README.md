2025年10月28日更新
# 1.ファイル構成
```
AI_FT_Collaborative Project/
├── config/ ：各種パラメーターをまとめたコンフィグファイル(yaml形式)を保存します。
│   ├── ["%Y%m%d"]/ ：param_set.pyで作成されるコンフィグファイルを保存します。フォルダ名は任意の作成年月日（実行日）を示します。
│   ├── config_loader.py：コンフィグファイルを参照する各種関数が格納されます。
│   └── param_set.py    ：コンフィグファイルを作成する各種関数が格納されます。
│
├── dataset/：データセットを保存します。
│   ├── ["%Y%m%d"]/ dataset_maker.pyで作成されるデータセットを保存します。フォルダ名は任意の作成年月日（実行日）を示します。
│   ├── ORG/：Excel（csv含む）形式のデータセットを保存します。
│   │   ├── QA/     ：QA形式で作成したデータセットを保存します。
│   │   └── Text/   ：原文形式で作成したデータセットを保存時ます。
│   ├── ["%Y%m%d"]/ ：dataset_maker.pyで作成されるデータセットを保存します。フォルダ名は任意の作成年月日を示します。
│   ├── dataset_maker.py：[ORG]からjsonl形式でデータセットを作成する各種関数が格納されます。
│   └── load_dataset.py ：データセットを参照する各種関数が格納されます。
│   
├── model/  ：学習後のＬＬＭモデルを保存します。
│   ├── ["%Y%m%d"]/ ：Finetuning.pyで作成されるモデルを保存します。フォルダ名は任意の作成年月日（実行日）を示します。
│   ├── load_transformers.py：モデルとトークナイザーを読み込む各種関数が格納されます。[transformers]モジュールから作成されます（推奨）。
│   └── load_unsloth.py     ：モデルとトークナイザーを読み込む各種関数が格納されます。[unsloth]モジュールから作成されます（現在調整中）。
│
├── test/   ：モデルの動作テスト結果を保存します。
│
├── utils/  ：その他関数を格納したpythonファイルを保存します。
│   └── seed_fixed.py   ：各種モジュール(random、numpy、torch)のシード値を任意値に固定します。
│
├── wandb/  ：外部サイトwandbへのデータを保存します。学習結果を表示します。
│
├── anyParameters.py：各種パラメーターを手動で指定します。
├── Finetuning.py   ：ファインチューニングを実施します。
├── ModelTest.py    ：モデルの動作テストを実施します。
├── README.md       ：このプログラムの使い方を簡潔にまとめたマークダウン形式のファイル。
└── requirements.txt：このプログラムを動作するのに必要な各種モジュールをまとめたテキスト形式のファイル。
```

# 2.プログラムの使い方
1. 必要な各種モジュールのインストール（初回のみ）<br>
    　requirements.txt内に必要なモジュールを整理しています。ターミナルでAI_FT_Collaborative Project(本プログラム)内にアクセスし以下のコマンドを実行して、必要なモジュールをインストールします。<br>
    　なお、２回目以降は実行不要です。
    ```　shell
    # 仮想環境(.venv)をアクティブ化
    pip install -r requirements.txt
    ```
    　なお、仮想環境内で実行することを推奨します。
    ```　shell
    # python3.11環境で仮想環境(.venv)を作成
    conda create -n .venv python=3.11
    
    # 仮想環境(.venv)をアクティブ化
    conda activate .venv
    ```
    　tmux
    ```　shell
    # 新規セッション開始
    tmux

    # 名前をつけて新規セッション開始
    tmux new -s <セッション名>
    
    # tmuxをアクティブ化
    tmux a

    # tmuxを閉じる
    ctrl + b →　d
    ```
    <br>
2. 各種パラメーターの設定<br>
    　anyParameters.py内で各種パラメーターを設定することができます。特定のパラメーターで複数の条件がある場合、[]内に複数設定することで、それぞれのコンフィグファイルを作成することができます。
    ```　python
    # 各種パラメーターの設定例
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
        "lora_dropout":[0.05], # LoRAのドロップアウト率 
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
        "train_ratio":[0.8] # 訓練データの割合（例：0.8は80%を訓練、20%を検証に使用）
    }
    ```
    <br>
3. モデルのトレーニング開始<br>
    　Finetuning.pyを実行することで、実行日の年月日でコンフィグファイルとデータセットを自動生成し、それらを参照したファインチューニングを開始します。
    　学習モデルは./model内に保存されます。この時、実行日と参照コンフィグファイル番号で整理されます（例：実行日＝2025/10/28、参照コンフィグファイル＝param_set_1.yamlの場合、./model/20251028/train1内に作成されます）。
    　trainingモデルでは、「metric_for_best_model」を採用しております。これにより、ベストな学習パラメータを随時保存しています。
    　ターミナルで以下のコマンドを実行します。
    ```　shell
    # Finetuning.pyを実行
    python Finetuning.py
    ```
4. モデルの動作テスト<br>
    　ModelTest.pyを実行することで、モデルの動作テストを実施することができます。
    　なお、コマンドラインに引数を設定する必要があります。引数を設定しない場合、最後に実行されたtrainingのparam_set_1.yamlにて設定されたデフォルトモデル（学習前モデル）を実施します。コマンドラインの引数の内訳は以下の通りです。
    ```　shell
    # 実行コマンド例
    python ModelTest.py --input_mode interactive --model_mode trained --paramdate --config_num --model_name --jsonl_path
    '''
    input_mode = interactive　→　インプットを任意で設定する
    model_mode = trained　→　trainingモデルで動作テストする
    paramdate = [未設定]　→　デフォルトを参照する
    config_num = [未設定]　→　デフォルトを参照する
    model_name = [未設定]　→　デフォルトを参照する
    jsonl_path = [未設定]　→　デフォルトを参照する
    '''
    # コマンドラインの引数内訳
    ・input_mode；
        ＜説明＞　LLMのインプット形式を指定します。入力形式は[任意入力]か[既定入力]のいずれかを設定できます。なお、[既定入力]の場合は[jsonl_path]を指定する必要があります。
        ＜入力形式＞　　interactive（任意入力）　／　batch（既定入力）
        ＜デフォルト＞　interactive

    ・model_mode；
        ＜説明＞　動作テストを実施するLLMモデル形式を指定します。LLMモデル形式は[学習前モデル]と[学習後モデル]のいずれかを設定できます。なお、[学習前モデル]の場合は参照するコンフィグファイルの[model][name]によって決定します（[model_name]は指定する必要がありません）。
        ＜入力形式＞　　default（学習前モデル）　／　trained（学習後モデル）
        ＜デフォルト＞　default

    ・paramdate；
        ＜説明＞　参照したい学習の実施年月日を指定します。実施年月日はコンフィグファイルやデータセット、モデルファイルのフォルダ名に使用されます。
        ＜入力形式＞　8桁の整数（YYYYMMDD形式）　→　例：2025/10/28の場合、「20251028」と入力する　
        ＜デフォルト＞　直近の学習実行日時を自動設定（configフォルダ参照）

    ・config_num；
        ＜説明＞　参照したいコンフィグファイルの番号を指定します。
        ＜入力形式＞　　整数（param_set_*.py の * を設定）　→　例：param_set_3.yamlの場合、「3」と入力する
        ＜デフォルト＞　「1」を自動設定

    ・model_name；
        ＜説明＞　使用したいモデル名を指定します。なお、モデルの作成日と参照コンフィグファイル番号は、それぞれ[paramdate]と[config_num]を参照します。存在しないモデル名を入力した場合、動作を中断します。
        ＜入力形式＞　学習モデルの保存フォルダ名（例：checkpoint-3680　←　./model/[paramdate]/train{[config_num]}/内に存在するフォルダ名である必要があります。）
        ＜デフォルト＞　一番最後に保存された学習モデルフォルダを自動設定

    ・jsonl_path；
        ＜説明＞　input_modeがbatch（既定入力）のときに参照するインプット一覧（jsonlファイル）の保存先を指定します。インプット一覧の保存先は./test内を推奨します。
        ＜入力形式＞　int形式
        ＜デフォルト＞　./test内の任意のjsonlファイル（存在しない場合、Noneになります）。

    ```