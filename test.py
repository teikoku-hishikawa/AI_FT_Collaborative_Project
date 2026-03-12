import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

payload = {
    "model": "qwen3:8b",
    "prompt": "日本語で『OllamaとPythonの接続テスト成功』とだけ返してください。",
    "stream": False
}

response = requests.post(
    OLLAMA_URL,
    json=payload,
    timeout=180
)

response.raise_for_status()

result = response.json()
print(result["response"])
