import os
import glob
import base64
import subprocess
import requests
import re
import fitz
import itertools
import shutil
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
INPUT_FOLDER = "./all"
OUTPUT_FOLDER = "./processed_txt_v2"
TEMP_FOLDER = "./temp_conversion"
IMAGE_OUTPUT_FOLDER = "./processed_images"
CHECKPOINT_FOLDER = "./page_checkpoints" 

WORKER_NODES = [
    "http://192.168.0.111:80/api/generate",
    "http://192.168.0.100:80/api/generate"
]
node_cycle = itertools.cycle(WORKER_NODES)
MODEL = "minicpm-v"

for folder in [OUTPUT_FOLDER, TEMP_FOLDER, IMAGE_OUTPUT_FOLDER, CHECKPOINT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def sanitize_text(text):
    if not text: return ""
    text = re.sub(r'(.)\1{10,}', r'\1\1\1', text) 
    text = re.sub(r'(\b\w+\s+)\1{4,}', r'\1', text) 
    return text.strip()

def convert_doc_to_pdf(input_path):
    filename = os.path.basename(input_path)
    subprocess.run(['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', TEMP_FOLDER, input_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.join(TEMP_FOLDER, os.path.splitext(filename)[0] + ".pdf")

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG") 
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_minicpm_ocr(base64_img):
    prompt = "You are a high-precision OCR engine. Transcribe the text in this image verbatim. Preserve lists and headers. Do not add conversational filler. Just output the text."
    payload = {"model": MODEL, "prompt": prompt, "images": [base64_img], "stream": False, "options": {"temperature": 0.0, "num_predict": 500, "top_k": 1, "num_ctx": 4096}}
    try:
        response = requests.post(next(node_cycle), json=payload, timeout=240)
        if response.status_code == 200:
            return sanitize_text(response.json().get('response', '').replace("：", ": ").replace("，", ", "))
    except Exception as e: 
        print(f"  [!] Vision OCR Error: {e}")
    return ""

def process_single_page(page, i, base_name, chunk_dir):
    """Worker function to process a single page on a separate thread."""
    chunk_file = os.path.join(chunk_dir, f"page_{i+1:04d}.txt")

    # --- TEXT CHECKPOINT LOGIC ---
    if os.path.exists(chunk_file):
        print(f"    Page {i+1} already processed (Loaded from checkpoint).", flush=True)
        with open(chunk_file, "r", encoding="utf-8") as f:
            return i, f.read()

    native_text = page.get_text("text")
    img_path = os.path.join(IMAGE_OUTPUT_FOLDER, f"{base_name}_Page_{i+1:02d}.png")
    
    # --- IMAGE OVERWRITE PREVENTION ---
    if not os.path.exists(img_path):
        pix = page.get_pixmap(dpi=150)
        img_display = Image.open(BytesIO(pix.tobytes("png")))
        img_display.save(img_path, "PNG")

    if len(native_text.strip()) > 50:
        method, final_text = "NATIVE", native_text
    else:
        method = "MINICPM_OCR"
        # Extract 300 DPI to RAM for OCR, but do not save to disk
        img_b64 = image_to_base64(Image.open(BytesIO(page.get_pixmap(dpi=300).tobytes("png"))))
        final_text = get_minicpm_ocr(img_b64)

    output = f"\n--- PAGE {i+1} [{method}] ---\n### IMAGE PATH\n{img_path}\n\n### TEXT CONTENT\n{sanitize_text(final_text)}\n----------------------------------------\n"
    
    with open(chunk_file, "w", encoding="utf-8") as f:
        f.write(output)
        
    print(f"    Page {i+1} ({method}) done.", flush=True)
    return i, output

def process_pdf(pdf_path, original_filename):
    doc = fitz.open(pdf_path)
    base_name = os.path.splitext(original_filename)[0]
    
    chunk_dir = os.path.join(CHECKPOINT_FOLDER, base_name)
    os.makedirs(chunk_dir, exist_ok=True)
    
    page_results = [None] * len(doc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_single_page, page, i, base_name, chunk_dir): i for i, page in enumerate(doc)}
        for future in as_completed(futures):
            try:
                idx, content = future.result()
                page_results[idx] = content
            except Exception as e:
                print(f"  [!] Error on a page: {e}")

    shutil.rmtree(chunk_dir, ignore_errors=True)
    return "".join(filter(None, page_results))

def main():
    files = []
    for ext in ['*.pdf', '*.doc', '*.docx']: 
        files.extend(glob.glob(os.path.join(INPUT_FOLDER, '**', ext), recursive=True))
    
    for file_path in files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_FOLDER, f"{filename}.txt")
        
        if os.path.exists(output_path): 
            continue

        print(f"\n[NEW FILE DETECTED] Processing: {filename}")
        pdf_path, is_temp = file_path, False
        if file_path.lower().endswith(('.doc', '.docx')):
            pdf_path = convert_doc_to_pdf(file_path)
            is_temp = True

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"FILENAME: {filename}\nPATH: {file_path}\n" + process_pdf(pdf_path, filename))
        except Exception as e: 
            print(f"  [!] Error processing {filename}: {e}")
        
        if is_temp and os.path.exists(pdf_path): 
            os.remove(pdf_path)

if __name__ == "__main__":
    main()
