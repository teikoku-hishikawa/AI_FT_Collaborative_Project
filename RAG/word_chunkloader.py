import os
import re
import glob
import uuid
import unicodedata
import json
import pandas as pd

from tqdm import tqdm

from title_AIAgent import CoverAgent
from word_structure import StructureParser

doc_path_def = os.path.join(os.path.dirname(__file__), "data", "ORG")

class chunkloader:

    def __init__(self):

        #　不明な仕様書カウンター
        self.unknown_document_count = 0
        pass
    
    # Wordの保存フォルダ（doc_dir）から各種Wordを参照　→　chunk化
    def word2chunk(self, doc_dir):
        # 表紙確認
        files = glob.glob(os.path.join(doc_dir, "*.docx"))
        
        # 表紙から仕様書名と発行元、発行年月を取得
        cover_info = None
        for file in files:
            cover_agent = CoverAgent()
            tmp = cover_agent.run(file)

            if tmp.get("document_title"):
                cover_info = tmp
                break

        # 表紙が確認できない場合
        if cover_info is None:
            cover_info = {
                "document_title": os.path.basename(doc_dir),
                "publisher": None,
                "created_date": None,
                "version": None
            }

        # メタデータ作成
        spec_meta = {
            "chunk_id": "",
            "document_id": str(uuid.uuid4()),
            "document_title": cover_info.get("document_title"),
            "publisher": cover_info.get("publisher"),
            "created_date": cover_info.get("created_date"),
            "source_type": "docx",
            "version": cover_info.get("version"),
        }

        # JSON形式で出力
        directory_chunks = []
        for file in files:
            parser = StructureParser()
            chunk = parser.main(file, spec_meta)

            directory_chunks.extend(chunk)

        return directory_chunks
    
    # chunk_idをセット
    def chunkID(self, chunks):
        # ID追加
        save_base = ""
        new_chunks = []
        for i, chunk in enumerate(chunks):
            # 仕様書名
            document_name = ""
            if chunk.get("document_title") :
                publisher = chunk.get("publisher")
                document_title = chunk.get("document_title") 
                version = chunk.get("version") 
                created_date = chunk.get("created_date") 
                document_name = f"{publisher}_{document_title}_{version}_{created_date}"
            else:
                self.unknown_document_count += 1
                document_name = f"UnknownDocument_{self.unknown_document_count}"

            # 章番号
            chapter_num = 'Nun'
            cur_chapter = chunk.get("chapter_title")
            if cur_chapter:
                for pat in StructureParser().CHAPTER_PATTERNS:
                    m = re.match(pat, cur_chapter)
                    if m:
                        chapter_num = m.group(m.lastindex-1)
                        break
            
            # 節番号
            section_num = 'Nun'
            cur_section = chunk.get("section_title")
            if cur_section:
                for pat in StructureParser().SECTION_PATTERNS:
                    m = re.match(pat, cur_section)
                    if m:
                        section_num = m.group(m.lastindex-1)
                        break
            
            # 項番号
            item_num = 'Nun'
            cur_item = chunk.get("item_title")
            if cur_item:
                for pat in StructureParser().SECTION_PATTERNS:
                    m = re.match(pat, cur_item)
                    if m:
                        item_num = m.group(m.lastindex-1)
                        break

            # 号番号
            subitem_num = 'Nun'
            cur_subitem = chunk.get("subitem_title")
            if cur_subitem:
                for pat in StructureParser().SECTION_PATTERNS:
                    m = re.match(pat, cur_subitem)
                    if m:
                        subitem_num = m.group(m.lastindex-1)
                        break

            # ページ番号
            page_num = chunk.get("logical_page_number")
            
            # chunk_id
            chunk_id_base = f'{document_name}-{chapter_num}-{section_num}-{item_num}-{subitem_num}-{page_num}'
            if not chunk_id_base == save_base:
                chunk_id = f'{chunk_id_base}-1'
                save_base = chunk_id_base
                counter = 2
            else:
                chunk_id = f'{chunk_id_base}-{counter}'
                counter += 1
            
            chunk_id = unicodedata.normalize('NFKC', chunk_id)
            
            # 更新
            chunk['chunk_id'] = chunk_id
            new_chunks.append(chunk)
        
        return new_chunks
    
    # contextの改行をなくす
    def context_cleen(self, chunks):

        new_chunks=[]
        for chunk in chunks:
            content_type = chunk.get("content_type")
            content = chunk.get("context")

            # bodyの改行を未記載に置換
            if content_type == "body":
                if content:
                    # 改行 → スペース
                    content = re.sub(r"[\r\n]+", " ", content)

                    # 余分スペース削除
                    content = re.sub(r"\s+", " ", content).strip()
            
            # 更新
            chunk["context"] = content
            new_chunks.append(chunk)

        return new_chunks

    # Json配列化
    def enrich_chunks(self, chunks):

        new_chunks = []
        for chunk in chunks:
            # hierarchy_path
            chapter = chunk.get("chapter_title") or ""
            section = chunk.get("section_title") or ""
            figure  = chunk.get("figure_title") or ""

            parts = [p for p in [chapter, section, figure] if p]
            hierarchy_path = " / ".join(parts)

            chunk["hierarchy_path"] = hierarchy_path

            # embedding text
            summary = chunk.get("chunk_summary","")
            context = chunk.get("context","")

            embedding_text = f"""passage:
{hierarchy_path}
{summary}
{context}
"""

            chunk["embedding_text"] = embedding_text
            new_chunks.append(chunk)

        return new_chunks
    
    # jsonlファイル作成
    def save_jsonl(self, chunks, path):

        with open(path, "w", encoding="utf-8") as f:

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

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # チャンクデータ作成
    def main(self, doc_path = doc_path_def, json_path = ""):
        # word参照
        doc_dirs = glob.glob(os.path.join(doc_path, "*/"))
        all_chunks = []
        pbar = tqdm(doc_dirs)
        for doc_dir in pbar:
            # 作業中のディレクトリ確認
            basename = os.path.basename(doc_dir.rstrip("/"))
            pbar.set_description(f"Processing_{basename}")

            # チャンク確認
            dir_chunks = self.word2chunk(doc_dir)
            all_chunks.extend(dir_chunks)

        # ID更新
        all_chunks = self.chunkID(all_chunks)

        # bodyの改行をなくす
        # all_chunks = self.context_cleen(all_chunks)

        # hierarchy_path + embedding_text
        all_chunks = self.enrich_chunks(all_chunks)

        #json印刷
        if ".josnl" in json_path:
            self.save_jsonl(all_chunks, self.json_path)

        return all_chunks

if __name__ == "__main__":
    
    json_path = os.path.join(os.path.dirname(__file__), "data", "chunkloader.josnl")

    loader = chunkloader()
    chunks = loader.main()