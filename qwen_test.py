import requests
import re
import csv
import os
import json
from datetime import datetime

from RAG.RAGinput import RAG

class CoverAgent:

    def __init__(self, model="qwen3.5:27b", max_contexts=5, score_threshold=0.7, stream=False):
        self.model = model
        self.max_contexts = max_contexts
        self.score_threshold = score_threshold
        self.stream = stream

    # ----------------------------
    # System Prompt
    # ----------------------------
    def system_prompt(self):

        return """
あなたは厳密なRAGベースQAシステムです。

以下に複数の「参照情報」が与えられます。
各コンテキストは「ソース名」と「本文」で構成されています。

あなたのタスクは以下の通りです。

【絶対ルール】
- 参照情報に含まれる情報のみを使用すること
- あなたの知識や推測を使用してはいけない
- 情報が不足している場合は「情報が不足しています」と出力する
- コンテキスト同士を混同しないこと
- 存在しない情報を補完しないこと

---

【処理手順】

### Step1: コンテキストごとの要約
各コンテキストについて以下を行う：

- ソース名をそのまま出力
- 本文の内容を簡潔に要約（2〜4文）
- 必ず本文に書かれている内容のみ使用

---

### Step2: 総括
Step1の結果のみを使用して、全体の要点をまとめる：

- 共通点・重要ポイントを整理
- 矛盾があれば明示
- 推測は禁止

---

### Step3: 自己検証（内部チェック）
以下を確認する：

- 参照情報外の知識を使っていないか
- 各要約が対応するコンテキストに基づいているか

違反があれば、出力全体を「情報が不足しています」に置き換える

---

【出力フォーマット】

■コンテキスト要約
1.
ソース名: <source_1>
要約: <context_1>

2.
ソース名: <source_2>
要約: <context_2>

（必要数繰り返し）

---

■総括
<overall_summary>

---
"""

    # ----------------------------
    # Ollama呼び出し
    # ----------------------------
    def call_ollama(self, prompt):
        # Ollamaで生成
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "system": self.system_prompt(),
                "stream": self.stream,
                "options": {
                    "num_predict": 512,
                    "temperature": 0.2
                }
            },
            stream=self.stream
        )

        # エラーが出たら、エラーメッセージを返す
        if response.status_code != 200:
            raise RuntimeError(response.text)

        # 出力結果をJSON化
        if self.stream:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    data = line.decode("utf-8")
                    try:
                        json_data = json.loads(data)
                        chunk = json_data.get("response", "")
                        print(chunk, end="", flush=True)
                        full_response += chunk
                    except:
                        pass
            print()  # 改行
        else:
            data = response.json()
            full_response = data["response"]

        return full_response

    # ----------------------------
    # メイン処理
    # ----------------------------
    def run(self, csv_path):

        print("===RAGデータベースを確認中===")
        rag = RAG( 
            os.path.join(os.path.dirname(__file__), "RAG", "data", "chunks_embedded.jsonl"),
            max_contexts=self.max_contexts, 
            score_threshold=self.score_threshold 
            ) 

        max_seq_length = 4096

        while True:
                
            user_input = input("質問を入力してください（終了するには 'exit' と入力）： ")
            if user_input.lower() == "exit":
                break

            RAG_results = rag.search(user_input) 
            prompt = rag.generate_prompt(user_input, RAG_results, max_seq_length) 

            print("\n----------------")
            print(">User")
            print(prompt)

            # モデルに入力して応答を取得
            result = self.call_ollama(prompt)

            # 結果の表示
            print("\n>Model")
            print(result)
            print("----------------\n")

            # CSV保存
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.save_to_csv(csv_path, [[timestamp, self.model, user_input, RAG_results, result]])

    # 結果をCSVに保存
    def save_to_csv(self, csv_path, records):
        """結果をCSVに保存"""
        header = ["timestamp", "model_name", "input_text", "RAG_results", "generated_text"]
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerows(records)


# ----------------------------
# 実行
# ----------------------------
if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "test", "Qwen35_test.csv")
    
    agent = CoverAgent(stream=True)
    agent.run(csv_path)