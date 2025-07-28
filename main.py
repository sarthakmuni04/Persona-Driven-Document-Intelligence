import os
import json
import datetime
import fitz
import faiss
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer

INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"
EMBED_MODEL_NAME = 'all-mpnet-base-v2'
SUMM_MODEL_NAME = 't5-small'
TOP_K_SECTIONS = 20

embed_model = SentenceTransformer(EMBED_MODEL_NAME)
summ_model = T5ForConditionalGeneration.from_pretrained(SUMM_MODEL_NAME)
summ_tokenizer = T5Tokenizer.from_pretrained(SUMM_MODEL_NAME)

def extract_section_title(page):
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                txt = span.get("text", "").strip()
                size = span.get("size", 0)
                font = span.get("font", "")
                if size >= 12 and re.search(r"[A-Za-z0-9]", txt) and len(txt.split()) >= 3 and ("Bold" in font or "bold" in font):
                    return txt
    return None

def parse_pdf_sections(path):
    sections = []
    pdf = fitz.open(path)
    current = {
        'document': os.path.basename(path),
        'section_title': os.path.splitext(os.path.basename(path))[0],
        'text': "",
        'page_number': 1
    }
    for i, page in enumerate(pdf):
        text = page.get_text("text").strip()
        heading = extract_section_title(page)
        if heading:
            if current['text'].strip():
                sections.append(current)
            current = {
                'document': os.path.basename(path),
                'section_title': heading,
                'text': text,
                'page_number': i + 1
            }
        else:
            if text:
                current['text'] += "\n" + text
    if current['text'].strip():
        sections.append(current)
    return sections

def build_faiss_index(texts):
    embs = embed_model.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embs)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx

def summarize_text(text, persona="analyst", job="understand the section", max_length=128):
    prompt = f"summarize for a {persona} looking to {job}: {text}"
    inputs = summ_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    outs = summ_model.generate(**inputs, max_length=max_length, num_beams=2)
    return summ_tokenizer.decode(outs[0], skip_special_tokens=True)

def process_pdf(file_path):
    filename = os.path.basename(file_path)
    persona = "analyst"
    job = "understand the section"

    sections = parse_pdf_sections(file_path)
    texts = [sec['text'] for sec in sections]
    index = build_faiss_index(texts)
    q_text = f"{persona}. Task: {job}"
    q_emb = embed_model.encode([q_text], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    distances, indices = index.search(q_emb, TOP_K_SECTIONS)
    extracted, analysis = [], []

    for rank, idx in enumerate(indices[0], start=1):
        sec = sections[idx]
        extracted.append({
            'document': sec['document'],
            'section_title': sec['section_title'],
            'importance_rank': rank,
            'page_number': sec['page_number']
        })
        analysis.append({
            'document': sec['document'],
            'page_number': sec['page_number'],
            'refined_text': summarize_text(sec['text'], persona, job)
        })

    output = {
        'metadata': {
            'input_document': filename,
            'persona': persona,
            'job_to_be_done': job,
            'processing_timestamp': datetime.datetime.utcnow().isoformat()
        },
        'extracted_sections': extracted,
        'subsection_analysis': analysis
    }

    with open(os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + ".json"), 'w') as f:
        json.dump(output, f, indent=2)

def main():
    for file in os.listdir(INPUT_DIR):
        if file.endswith(".pdf"):
            process_pdf(os.path.join(INPUT_DIR, file))
    print("✅ All PDFs processed!")

if __name__ == "__main__":
    main()
