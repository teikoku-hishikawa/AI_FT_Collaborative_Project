import requests
import json
import re
import csv
import os
from tqdm import tqdm


class CoverAgent:

    def __init__(self, model="qwen3:8b"):
        self.model = model

    # ----------------------------
    # CSV → JSON
    # ----------------------------
    def csv_to_json(self, csv_path):

        rows = []

        with open(csv_path, mode="r", encoding="utf-8") as f:

            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:
                rows.append(row)

        return rows


    # ----------------------------
    # JSON → LLM入力テキスト
    # ----------------------------
    def build_prompt(self, row):

        texts = []

        for key, value in row.items():

            if value:
                texts.append(f"{key}:{value}")

        return "\n".join(texts)


    # ----------------------------
    # System Prompt
    # ----------------------------
    def system_prompt(self):

        return """
あなたは行政文書・業務仕様書からLLM学習用のデータセットを作成するAIです。
収集情報を元にOutputを改良したNewOutputを生成してください。
以下は、収集情報とNewOutputの作成例です。
作成例と重要事項の内容を考慮して、作成してください。

【収集情報例】
Input：視覚障害者誘導用ブロックの点状突起を配列するブロックの最小の大きさは、目地込みで何mm四方以上とされていますか？
Output：視覚障害者誘導用ブロックの点状突起を配列するブロックの最小の大きさは、目地込みで**300mm四方以上**とされています。
Source_1：06_交通安全（愛知_設計手引き）.pdf
Context_1：点状ブロックの形状・寸法および配列\n・点状突起を配列するブロック等の大きさは**300mm (目地込み) 四方以上**とする。
Source_2：福島県＿道路設計マニュアル506580.pdf
Context_2：最小可能施工厚さは仕上がり厚で、骨材最大粒径が**20mmの場合5cm**、骨材最大粒径が**13mmの場合4cm**を標準とする。
Source_3：福島県＿道路設計マニュアル506580.pdf
Context_3：最小可能施工厚さは仕上がり厚で、骨材最大粒径が**20mmの場合5cm**、骨材最大粒径が**13mmの場合4cm**を標準とする[18]。
Source_4：道路安全施設 機能別分類説明.pdf
Context_4：視覚障害者は、... 弱視者は、視覚障害者誘導用ブロックの**色のコント ラスト**により認識している場合もある。 [18]
Source_5：道路構造令の解説と運用（令和3年）_R3年3月-13.pdf
Context_5：設ri-速1¢ (屯位1時問につきキロ メートル) 視距(単位メー トル)\n120 **210**\n100 **160**\n80 **110**\n60 **75**\n50 **55**\n40 **40**\n30 **30**\n20 **20**"

【NewOutputの作成例】

質問に関連する資料では、以下のように記載されています。

・06_交通安全（愛知_設計手引き）.pdf
状突起を配列するブロック等の大きさは300mm四方以上とされています。

・福島県＿道路設計マニュアル506580.pdf
最小可能施工厚さは仕上がり厚で、骨材最大粒径が20mmの場合5cm、骨材最大粒径が3mmの場合4cmを標準とする。

・福島県＿道路設計マニュアル506580.pdf
最小可能施工厚さは仕上がり厚で、骨材最大粒径が20mmの場合5cm、骨材最大粒径が3mmの場合4cmを標準とする。

・道路安全施設 機能別分類説明.pdf
視覚障害者は、視覚障害者誘導用ブロックの色のコントラストにより認識している

総じて、覚障害者誘導用ブロックの点状突起を配列するブロックの最小の大きさは、目地込みで300mm四方以上とされています。


【NewOutputの生成ルール】
複数のソースがある場合、「質問に関連する資料では、以下のように記載されています。」といった書き始めとし、
その次に、各ソース名とコンテキストの内容の要約を整理してください。
最後に、「総じて」から始まり、Outputの内容を整理してください。

【重要事項】
・必ず以下のJSON形式で出力してください。

{
 "NewOutput": "生成した文章"
}

・Contextに存在しない内容は追加しないこと。
・Contextの内容が不明な場合、無視してください。
・Contextの内容がInputと関係なさそうでも、検索でヒットした文章なので、箇条書するようにしてください。
ただし、「総じて」以降の文章には反映させないものとしてください。
"""


    # ----------------------------
    # Ollama呼び出し
    # ----------------------------
    def call_ollama(self, prompt):

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "system": self.system_prompt(),
                "stream": False
            }
        )

        if response.status_code != 200:
            raise RuntimeError(response.text)

        data = response.json()

        return data["response"]


    # ----------------------------
    # JSON安全パース
    # ----------------------------
    def safe_json_parse(self, text):

        match = re.search(r"\{.*\}", text, re.S)

        if not match:
            return {"NewOutput": "ERROR: JSON parse failed"}

        try:
            return json.loads(match.group())
        except:
            return {"NewOutput": "ERROR: JSON decode failed"}


    # ----------------------------
    # メイン処理
    # ----------------------------
    def run(self, input_csv_path, output_csv_path):

        data = self.csv_to_json(input_csv_path)

        results = []

        for i, row in tqdm(enumerate(data)):

            prompt = self.build_prompt(row)

            try:

                raw = self.call_ollama(prompt)

                parsed = self.safe_json_parse(raw)

                row["AI_outputs"] = parsed.get("NewOutput", "")

            except Exception as e:

                row["AI_outputs"] = f"ERROR: {str(e)}"

            results.append(row)

            if i == 10:
                break

        # CSV出力

        fieldnames = list(results[0].keys())

        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:

            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")

            writer.writeheader()

            writer.writerows(results)

        print("Completed")


# ----------------------------
# 実行
# ----------------------------
if __name__ == "__main__":

    input_csv_path = os.path.join(os.path.dirname(__file__), "sample.csv")

    output_csv_path = os.path.join(os.path.dirname(__file__), "AIsample.csv")

    agent = CoverAgent()

    agent.run(input_csv_path, output_csv_path)