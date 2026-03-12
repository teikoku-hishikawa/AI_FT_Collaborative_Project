import requests
import json
import re
from docx import Document

class CoverAgent:

    def __init__(self, model="qwen3:8b"):
        self.model = model

    # word冒頭の数段落（max_paragraphs）を抜粋
    def extract_first_pages(self, doc, max_paragraphs=30):
        texts = []
        for i, p in enumerate(doc.paragraphs):
            if i >= max_paragraphs:
                break
            texts.append(p.text)
        return "\n".join(texts)
    
    # システムプロンプト案
    def system_prompt(self):
        return f"""
                あなたは行政文書・業務仕様書を解析するAIエージェントです。
                以下の文書が「仕様書の表紙（タイトルページ）」かどうかを判定してください。

                【判定基準】
                - 表紙とは、仕様書名・作成年月・発行主体等が記載され、
                本文（目次・章・条文）が始まる前のページを指します。
                - 表紙であると明確に判断できない場合は false としてください。
                - 推測や補完は行わないでください。

                結果は必ずJSON形式で出力してください。以下は、出力形式です。
                {{
                    "is_title_page": true/false,
                    "document_title": "",
                    "publisher": "",
                    "created_date": "",
                    "version":""
                }}
                【内訳】
                - is_title_page : 仕様書の表紙ならtrue、違うならfalse
                - document_title : 仕様書名（不明またはis_title_page=falseなら null）
                - publisher : 仕様書の発行元名（不明またはis_title_page=falseなら null）
                - created_date : 作成年月（YYYY-MM または YYYY-MM-DD、不明またはis_title_page=falseなら null）
                - version : 仕様書のバージョン情報（不明またはis_title_page=falseなら null）
                """

    # モデルのプロンプトベース案
    # def build_prompt(self, text):
    #     return f"""
    #             以下は仕様書の冒頭部分です。

    #             {text}
    #             """

    # モデルをollamaで呼び出す
    def call_ollama(self, prompt):
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "system": self.system_prompt(),
                "stream": False,
                "format": "json"
            }
        )

        # HTTPエラー確認
        if response.status_code != 200:
            raise RuntimeError(f"HTTP Error: {response.status_code} - {response.text}")

        data = response.json()

        # Ollamaエラー確認
        if "error" in data:
            raise RuntimeError(f"Ollama Error: {data['error']}")

        if "response" not in data:
            raise RuntimeError(f"Unexpected response format: {data}")

        return data["response"]
    
    def safe_json_parse(self, text):
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError("No JSON found in model output")

        return json.loads(match.group())
        
    def run(self, docx_path):
        doc = Document(docx_path)
        text = self.extract_first_pages(doc)
        raw_response = self.call_ollama(text)
        return json.loads(raw_response)
    
if __name__ == "__main__":
    import os
    import glob

    doc_path = os.path.join(os.path.dirname(__file__), "data", "ORG")
    files = glob.glob(os.path.join(doc_path, "**/*.docx"))

    for file in files:
        # エージェント使用
        agent = CoverAgent()
        result = agent.run(file)
        
        # テスト
        basename = os.path.basename(file)
        print("=== RESULT ===")
        print(basename)
        print(json.dumps(result, ensure_ascii=False, indent=2))