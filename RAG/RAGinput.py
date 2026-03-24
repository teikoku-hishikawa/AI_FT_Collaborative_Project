import json
import faiss
import ollama
import numpy as np

from .embedding_model import embed_query

from tqdm import tqdm
from collections import defaultdict
from rank_bm25 import BM25Okapi
from janome.tokenizer import Tokenizer


class RAG:

    def __init__(self, jsonl, max_contexts=5, score_threshold=0.7):

        self.max_contexts = max_contexts
        self.score_threshold = score_threshold

        self.records = []
        vectors = []

        # ------------------------
        # load data
        # ------------------------
        
        with open(jsonl, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                self.records.append(r)
                vectors.append(r["embedding"])

        self.vectors = np.array(vectors).astype("float32")

        # cosine similarity化
        faiss.normalize_L2(self.vectors)

        # ------------------------
        # FAISS index
        # ------------------------

        dim = self.vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.vectors)

        # ------------------------
        # Parent map
        # ------------------------

        self.parent_map = defaultdict(list)
        for i, r in enumerate(self.records):

            m = r["metadata"]

            parent_id = f"{m.get('chapter','')}_{m.get('section','')}_{m.get('item','')}_{m.get('subitem','')}"
            self.parent_map[parent_id].append((i, r))

        # ------------------------
        # BM25
        # ------------------------

        self.tokenizer = Tokenizer()

        corpus = [r["text"] for r in self.records]
        tokenized = [self.tokenize(c) for c in corpus]

        self.bm25 = BM25Okapi(tokenized)

    # ------------------------
    # tokenize
    # ------------------------

    def tokenize(self, text):

        tokens = []

        for token in self.tokenizer.tokenize(text):

            base = token.base_form
            if base == "*":
                base = token.surface

            tokens.append(base)

        return tokens

    # ------------------------
    # Hybrid Search
    # ------------------------

    def search(self, query):

        q = embed_query(query).astype("float32")

        faiss.normalize_L2(q)

        vec_scores, vec_ids = self.index.search(q, 200)

        vec_ids = vec_ids[0]
        vec_scores = vec_scores[0]

        vec_ids = [i for i in vec_ids if i != -1]

        vec_score_map = {i: s for i, s in zip(vec_ids, vec_scores)}

        # BM25

        tokenized_query = self.tokenize(query)

        bm_scores = self.bm25.get_scores(tokenized_query)

        bm_ids = np.argsort(bm_scores)[::-1][:200]

        bm_max = max(bm_scores) if max(bm_scores) > 0 else 1

        # ------------------------
        # merge
        # ------------------------

        candidate_ids = set(vec_ids) | set(bm_ids)

        parent_scores = defaultdict(list)

        debug_chunks = []

        for idx in candidate_ids:
            
            record = self.records[idx]

            vec_score = vec_score_map.get(idx, 0)

            bm_score = bm_scores[idx] / bm_max

            score = vec_score * 0.8 + bm_score * 0.2

            debug_chunks.append((score, vec_score, bm_score, record))

            if score < self.score_threshold:
                continue

            m = record["metadata"]

            parent_id = f"{m.get('chapter','')}_{m.get('section','')}_{m.get('item','')}_{m.get('subitem','')}"

            parent_scores[parent_id].append(score)

        # ------------------------
        # debug
        # ------------------------

        debug_chunks.sort(key=lambda x: x[0], reverse=True)

        '''
        print("\n=== Candidate Scores ===\n")

        for score, vec, bm, r in debug_chunks[:10]:

            m = r["metadata"]

            print("Hybrid:", round(score,3),
                  "Vec:", round(vec,3),
                  "BM25:", round(bm,3))

            print(
                m.get("chapter",""),
                m.get("section",""),
                m.get("item",""),
                m.get("subitem","")
            )

            print(r["text"][:120])
            print()
        '''

        if len(parent_scores) == 0:
            return []

        # ------------------------
        # parent ranking
        # ------------------------

        def parent_score(scores):

            top = sorted(scores, reverse=True)[:3]

            return np.mean(top)

        ranked_parents = sorted(
            parent_scores.items(),
            key=lambda x: parent_score(x[1]),
            reverse=True
        )

        # ------------------------
        # context
        # ------------------------

        results = []
        for parent_id, scores in ranked_parents:

            pscore = parent_score(scores)

            for idx, record in self.parent_map[parent_id]:

                results.append((pscore, record))

                if len(results) >= self.max_contexts:
                    return results

        return results

    # ------------------------
    # generate
    # ------------------------

    def generate_prompt(self, query, contexts, max_seq_length=None):

        if len(contexts) == 0:
            return "仕様書内に該当する情報が見つかりませんでした。"

        context_text = ""

        prompt = (
            "以下の参考情報から、質問に答えてください。\n\n"
            f"質問：{query}\n\n"
            "参考情報：\n"
        )

        for i, (score, c) in enumerate(contexts):
            context_text = f"（参考{i+1}）" + c["metadata"]["document_title"] + "\n"
            lines = c["text"].splitlines()
            context = "\n".join(lines[3:]) + "\n\n"
            context_text = context_text + context

            check_prompt = (
                f"{prompt}"
                f"{context_text}"
            )

            if max_seq_length:
                if len(check_prompt) > max_seq_length - 14:
                    break
                else:
                    prompt = check_prompt
            else:
                prompt = check_prompt
                
        prompt = (
            f"{prompt}"
            "回答の方針：nan\n\n"
            "回答："
        )
        
        return prompt

    def generate_answer(self, prompt):
        
        response = ollama.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]
    
if __name__ == "__main__": 
    import os
    
    rag = RAG( 
        os.path.join(os.path.dirname(__file__), "data", "chunks_embedded.jsonl"),
        max_contexts=5, 
        score_threshold=0.7 
        ) 
    
    while True:
        query = input("\n質問 > ") 
        results = rag.search(query) 
        prompt = rag.generate_prompt(query, results) 
        answer = rag.generate_answer(prompt) 
        print("\n=== LLM回答 ===\n") 
        print(answer)