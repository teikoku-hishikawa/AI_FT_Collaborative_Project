import subprocess
import os
import pdfplumber
import re

# WordをPDF変換
def convert_docx_to_pdf(docx_path, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(docx_path)

    result = subprocess.run(
        [
        "soffice",
        "--headless",
        "--convert-to", "pdf",
        "--outdir", output_dir,
        docx_path
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # エラーのときだけログを表示
    if result.returncode != 0:
        print("PDF変換エラー:")
        print(result.stderr)
        raise RuntimeError("PDF conversion failed")

    pdf_path = os.path.join(
        output_dir,
        os.path.splitext(os.path.basename(docx_path))[0] + ".pdf"
    )

    return pdf_path

# PDFテキストページを取得
def extract_pdf_pages(pdf_path):
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({
                "page_number": i + 1,
                "text": text
            })

    return pages

# 正規化して照合
def normalize(text):
    text = text or ""
    text = re.sub(r"\s+", "", text)
    text = text.replace("－", "-")
    text = text.replace("ー", "-")
    text = text.replace("―", "-")
    return text

# ページ照合
def assign_page_by_heading(chunks, pdf_pages):

    normalized_pages = [
        {
            "page_number": p["page_number"],
            "text": normalize(p["text"])
        }
        for p in pdf_pages
    ]

    for chunk in chunks:

        chunk["word_page_number"] = None

        # 優先：節タイトル
        key = chunk.get("section_title")

        # なければ章タイトル
        if not key:
            key = chunk.get("chapter_title")

        if not key:
            continue

        key_norm = normalize(key)

        for page in normalized_pages:
            if key_norm in page["text"]:
                chunk["word_page_number"] = page["page_number"]
                break

    return chunks

def extract_logical_page_from_pdf(pdf_pages):

    LOGICAL_PAGE_PATTERN = re.compile(r"\b\d+\b")

    page_map = {}

    for page in pdf_pages:

        lines = (page["text"] or "").split("\n")

        # 下から3行を見る（フッター想定）
        footer_candidates = lines[-3:]

        logical_page = None

        for line in reversed(footer_candidates):
            numbers = LOGICAL_PAGE_PATTERN.findall(line)
            if numbers:
                logical_page = numbers[-1]  # 最後の数字を採用
                break

        page_map[page["page_number"]] = logical_page

    return page_map

def assign_logical_page_number(chunks, logical_page_map):

    for chunk in chunks:

        word_page = chunk.get("word_page_number")

        if word_page:
            chunk["logical_page_number"] = logical_page_map.get(word_page)
        else:
            chunk["logical_page_number"] = None

    return chunks

def pageNumber_set(docx_path, chunks):
    # 1. PDF変換
    pdf_path = convert_docx_to_pdf(docx_path)

    # 2. PDFページ抽出
    pdf_pages = extract_pdf_pages(pdf_path)

    # 3. ページ番号（ファイル単位）付与
    Ex_chunks = assign_page_by_heading(chunks, pdf_pages)

    # 4．フッターのページ番号確認
    logical_page_map = extract_logical_page_from_pdf(pdf_pages)

    # 5．フッターのページ番号付与
    Ex_chunks = assign_logical_page_number(Ex_chunks, logical_page_map)

    return Ex_chunks

