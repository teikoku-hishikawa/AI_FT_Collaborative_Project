import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

class Embedder:

    def __init__(self):

        self.model_name = "intfloat/multilingual-e5-base"

        print("Loading embedding model...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded on {self.device}")

    # 平均プーリング
    def average_pool(self, last_hidden_states, attention_mask):

        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )

        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    # embedding生成
    def embed(self, texts):

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():

            outputs = self.model(**inputs)

            embeddings = self.average_pool(
                outputs.last_hidden_state,
                inputs["attention_mask"]
            )

        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    # jsonl → embedding付与
    def process_jsonl(self, chunks, output_path, batch_size=16):
       
        records = []

        for chunk in chunks:
            record = {
                "id": chunk["chunk_id"],
                "text": chunk["embedding_text"],
                "metadata": {
                    "document_id": chunk.get("document_id"),
                    "document_title": chunk.get("document_title"),
                    "publisher": chunk.get("publisher"),
                    "version": chunk.get("version"),
                    "chapter": chunk.get("chapter_title"),
                    "section": chunk.get("section_title"),
                    "item": chunk.get("item_title"),
                    "subitem": chunk.get("subitem_title"),
                    "page": chunk.get("logical_page_number"),
                    "content_type": chunk.get("content_type")
                }
            }
            
            records.append(record)

        texts = [r["text"] for r in records]

        embeddings = []

        print("Generating embeddings...")
        for i in tqdm(range(0, len(texts), batch_size)):

            batch = texts[i:i+batch_size]

            emb = self.embed(batch)

            embeddings.extend(emb)

        # embeddingを追加
        for r, e in zip(records, embeddings):

            r["embedding"] = e.tolist()

        with open(output_path, "w", encoding="utf-8") as f:

            for r in records:

                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import os
    from .word_chunkloader import chunkloader
    
    # Word2Chunk
    print("===Word to Chunk Loading===")
    loader = chunkloader()
    chunks = loader.main()

    # 出力先
    output_jsonl = os.path.join(os.path.dirname(__file__), "data", "chunks_embedded.jsonl")
    
    embedder = Embedder()
    embedder.process_jsonl(chunks, output_jsonl)