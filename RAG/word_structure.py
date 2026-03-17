import re
from collections import Counter

from docx import Document
from docx.document import Document as _Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from PDF_loader import pageNumber_set

class StructureParser:

    def __init__(self,
                 min_ratio=0.1,       # 出現頻度閾値
                 indent_tolerance=3   # インデント誤差(pt)
                 ):
        self.min_ratio = min_ratio
        self.indent_tolerance = indent_tolerance

        # 章パターン
        self.CHAPTER_PATTERNS = [
            r"^\s*(\d+)\s*章", # 1章 タイトル
            r"^\s*第\s*(\d+)\s*章", # 第1章 タイトル
            r'^([0-9０-９]+)[ 　]+(.+)$' #1 タイトル
        ]

        # 節パターン
        self.SECTION_PATTERNS = [
            r"^\s*(\d+\.\d+)",
            r"^\s*第\s*(\d+)\s*節",
            r"^([0-9０-９]+)[－\-ー]([0-9０-９]+)[ 　]+(.+)"
        ]

        # 項パターン
        self.ITEM_PATTERNS = [
            r"^\s*(\d+\.\d+\.\d+)",
            r"^[\(（]([0-9０-９]+)[\)）]\s*(.+)", # (1) 項
        ]

        # 号パターン
        self.SUBITEM_PATTERNS = [
            r"^\s*(\d+\.\d+\.\d+\.\d+)",
            r"^\(\s*([a-zA-Z])\s*\)",
            r"^([0-9０-９]+)[\)）]\s*(.+)"      # 1) 号
        ]

        # 表タイトルパターン
        self.TABLE_TITLE_PATTERN = re.compile(r"^表\d*\s*.+")
        # 図タイトルパターン
        self.LOGICAL_PAGE_PATTERN = re.compile(
            r"^[\s\-－]*(\d+)[\s\-－]*$"
        )

    # ---------------------------
    # 見出し判定
    # ---------------------------
    def classify_heading(self, text):

        for pat in self.CHAPTER_PATTERNS:
            if re.match(pat, text):
                return "chapter"

        for pat in self.SECTION_PATTERNS:
            if re.match(pat, text):
                return "section"
            
        for pat in self.ITEM_PATTERNS:
            if re.match(pat, text):
                return "item"
        
        for pat in self.SUBITEM_PATTERNS:
            if re.match(pat, text):
                return "subitem"

        return None

    # ---------------------------
    # ブロック取得関数
    # ---------------------------
    def iter_block_items(self, parent):
        if isinstance(parent, _Document):
            parent_elm = parent.element.body
        else:
            parent_elm = parent._element

        for child in parent_elm.iterchildren():
            if child.tag.endswith('}p'):
                yield Paragraph(child, parent)
            elif child.tag.endswith('}tbl'):
                yield Table(child, parent)

    # チャンクの要約（chunk_summary）
    def chunk_summary_set(self, cur_chapter, cur_section, cur_item, cur_subitem, content_type):
        # 章内容確認
        chapter_summary =  ""
        if cur_chapter:
            for pat in self.CHAPTER_PATTERNS:
                m = re.match(pat, cur_chapter)
                if m:
                    chapter_summary = m.group(m.lastindex)
                    break
        
        # 節内容確認
        section_summary =  ""
        if cur_section:
            for pat in self.SECTION_PATTERNS:
                m = re.match(pat, cur_section)
                if m:
                    section_summary = m.group(m.lastindex)
                    break
        
        # 項内容確認
        item_summary =  ""
        if cur_item:
            for pat in self.ITEM_PATTERNS:
                m = re.match(pat, cur_item)
                if m:
                    item_summary =  m.group(m.lastindex)
                    break
        
        # 号内容確認
        subitem_summary =  ""
        if cur_subitem:
            for pat in self.SUBITEM_PATTERNS:
                m = re.match(pat, cur_subitem)
                if m:
                    subitem_summary =  m.group(m.lastindex)
                    break
        
        # コンテキストの形式
        type_map = {
            "body": "文章",
            "table": "表",
            "figure": "図"
        }
        text = type_map.get(content_type, "内容")

        # 要約文作成
        output = ""
        summarys = [chapter_summary, section_summary, item_summary, subitem_summary]

        for summary in summarys:
            if summary:
                output = f'{output}{summary}／'
        
        if output:
            output = f'{output}{text}'
        else:
            output = "Unknown"

        return output
    
    # チャンク取得
    def parse(self, docx_path, spec_meta):

        doc = Document(docx_path)

        chunks = []
        cur_chapter = None
        cur_section = None
        cur_item = None
        cur_subitem = None
        cur_logical_page = None
        buf = []
        last_paragraph_text = None

        for block in self.iter_block_items(doc):

            # ---------------------------
            # Paragraph
            # ---------------------------
            if isinstance(block, Paragraph):

                text = block.text.strip()
                if not text:
                    continue

                # 文頭ページ番号検出
                page_match = self.LOGICAL_PAGE_PATTERN.search(text)
                if page_match and len(text) <= 10:
                    # cur_logical_page = page_match.group(1)

                    # ページ切替時に flush
                    if buf:
                        chunks.append({
                            **spec_meta,
                            "file_name": docx_path,
                            "chapter_title": cur_chapter,
                            "section_title": cur_section,
                            "item_title": cur_item,
                            "subitem_title": cur_subitem,
                            "content_type": "body",
                            "figure_title": None,
                            "context": "\n".join(buf),
                            "chunk_summary":self.chunk_summary_set(cur_chapter,cur_section,cur_item,cur_subitem,"body"),
                            "logical_page_number": cur_logical_page
                        })
                        buf = []

                    cur_logical_page = page_match.group(1)
                    continue

                heading_type = self.classify_heading(text)

                if heading_type == "chapter":

                    if buf:
                        chunks.append({
                            **spec_meta,
                            "file_name": docx_path,
                            "chapter_title": cur_chapter,
                            "section_title": cur_section,
                            "item_title": cur_item,
                            "subitem_title": cur_subitem,
                            "content_type": "body",
                            "figure_title": None,
                            "context": "\n".join(buf),
                            "chunk_summary":self.chunk_summary_set(cur_chapter,cur_section,cur_item,cur_subitem,"body"),
                            "logical_page_number": cur_logical_page
                        })
                        buf = []

                    cur_chapter = text
                    cur_section = None
                    cur_item = None
                    cur_subitem = None
                    continue

                if heading_type == "section":

                    if buf:
                        chunks.append({
                            **spec_meta,
                            "file_name": docx_path,
                            "chapter_title": cur_chapter,
                            "section_title": cur_section,
                            "item_title": cur_item,
                            "subitem_title": cur_subitem,
                            "content_type": "body",
                            "figure_title": None,
                            "context": "\n".join(buf),
                            "chunk_summary":self.chunk_summary_set(cur_chapter,cur_section,cur_item,cur_subitem,"body"),
                            "logical_page_number": cur_logical_page
                        })
                        buf = []

                    cur_section = text
                    cur_item = None
                    cur_subitem = None
                    continue

                if heading_type == "item":

                    if buf:
                        chunks.append({
                            **spec_meta,
                            "file_name": docx_path,
                            "chapter_title": cur_chapter,
                            "section_title": cur_section,
                            "item_title": cur_item,
                            "subitem_title": cur_subitem,
                            "content_type": "body",
                            "figure_title": None,
                            "context": "\n".join(buf),
                            "chunk_summary":self.chunk_summary_set(cur_chapter,cur_section,cur_item,cur_subitem,"body"),
                            "logical_page_number": cur_logical_page
                        })
                        buf = []

                    cur_item = text
                    cur_subitem = None
                    continue

                if heading_type == "subitem":

                    if buf:
                        chunks.append({
                            **spec_meta,
                            "file_name": docx_path,
                            "chapter_title": cur_chapter,
                            "section_title": cur_section,
                            "item_title": cur_item,
                            "subitem_title": cur_subitem,
                            "content_type": "body",
                            "figure_title": None,
                            "context": "\n".join(buf),
                            "chunk_summary":self.chunk_summary_set(cur_chapter,cur_section,cur_item,cur_subitem,"body"),
                            "logical_page_number": cur_logical_page
                        })
                        buf = []

                    cur_subitem = text
                    continue

                buf.append(text)
                last_paragraph_text = text

            # ---------------------------
            # Table
            # ---------------------------
            elif isinstance(block, Table):

                # 表タイトル推定（直前段落）
                table_title = None
                # if last_paragraph_text and self.TABLE_TITLE_PATTERN.match(last_paragraph_text):
                #     table_title = last_paragraph_text
                if buf and self.TABLE_TITLE_PATTERN.match(buf[-1]):
                    table_title = buf.pop()

                # 表データ取得
                table_text = []
                for row in block.rows:
                    row_text = [cell.text.strip() for cell in row.cells]
                    table_text.append(" | ".join(row_text))

                chunks.append({
                    **spec_meta,
                    "file_name": docx_path,
                    "chapter_title": cur_chapter,
                    "section_title": cur_section,
                    "item_title": cur_item,
                    "subitem_title": cur_subitem,
                    "content_type": "table",
                    "figure_title": table_title,
                    "context": "\n".join(table_text),
                    "chunk_summary":self.chunk_summary_set(cur_chapter,cur_section,cur_item,cur_subitem,"table"),
                    "logical_page_number": cur_logical_page
                })

        if buf:
            chunks.append({
                **spec_meta,
                "file_name": docx_path,
                "chapter_title": cur_chapter,
                "section_title": cur_section,
                "item_title": cur_item,
                "subitem_title": cur_subitem,
                "content_type": "body",
                "figure_title": None,
                "context": "\n".join(buf),
                "chunk_summary":self.chunk_summary_set(cur_chapter,cur_section,cur_item,cur_subitem,"body"),
                "logical_page_number": cur_logical_page
            })

        return chunks
    
    def main(self, docx_path, spec_meta):

        # 1. 構造解析
        chunks = self.parse(docx_path, spec_meta)

        # 2. PDFからページ番号取得
        chunks = pageNumber_set(docx_path, chunks)

        return chunks
    
if __name__ == "__main__":
    import os
    import glob
    import uuid
    import pandas as pd

    from tqdm import tqdm
    from title_AIAgent import CoverAgent

    doc_path = os.path.join(os.path.dirname(__file__), "data", "ORG")
    csv_dir  = os.path.join(os.path.dirname(__file__), "data", "CSV")
    files = glob.glob(os.path.join(doc_path, "**/*.docx"))

    all_chunks = []
    for file in tqdm(files):
        # Wordパス参照

        cover_agent = CoverAgent()
        cover_info = cover_agent.run(file)

        # メタデータ作成
        spec_meta = {
            "document_id": str(uuid.uuid4()),
            "document_title": cover_info.get("document_title"),
            "publisher": cover_info.get("publisher"),
            "created_date": cover_info.get("created_date"),
            "source_type": "docx",
            "version": cover_info.get("version")
        }

        parser = StructureParser()
        chunk = parser.main(file, spec_meta)

        all_chunks += chunk

    df = pd.DataFrame(all_chunks)
    df.to_csv(f"{csv_dir}/chunkloader_test.csv", index=False, encoding="utf-8-sig")