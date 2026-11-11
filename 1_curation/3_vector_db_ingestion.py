import os
import glob
import re
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FOLDER = "./processed_txt_v2"
DB_DIR = "./chroma_db"
COLLECTION_NAME = "pathology_textbooks"

# Offloading embedding workload to the RTX 4060 node via Ollama
OLLAMA_API_URL = "http://192.168.0.117:11434/api/embeddings"
EMBEDDING_MODEL_NAME = "bge-large"

# --- DEFENSIVE CHARACTER-BASED CHUNKING ---
# 1000 characters guarantees staying well under the 512 token limit, regardless of vocabulary
MAX_CHARS_PER_CHUNK = 1000
CHARACTER_OVERLAP = 200

print(f"🦥 Connecting to Ollama embedding service on RTX 4060 ({OLLAMA_API_URL})...")
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url=OLLAMA_API_URL,
    model_name=EMBEDDING_MODEL_NAME
)

print("🗄️ Connecting to persistent ChromaDB instance...")
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME, 
    embedding_function=ollama_ef,
    metadata={"hnsw:space": "cosine"} 
)

def parse_master_file(file_path):
    """Parses Phase 1 master txt into safe character-bounded chunk blocks with metadata."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    filename = os.path.basename(file_path)
    
    # Regex to capture individual page blocks from Phase 1 output
    page_blocks = re.findall(
        r"--- PAGE (\d+) \[(NATIVE|MINICPM_OCR)\] ---\n(.*?)(?=\n----------------------------------------\n|\Z)", 
        content, 
        re.DOTALL
    )
    
    parsed_chunks = []
    for page_num, method, block_body in page_blocks:
        # Extract the image path
        img_match = re.search(r"### IMAGE PATH\n([^\n]+)", block_body)
        img_path = img_match.group(1).strip() if img_match else "NONE"
        
        # Extract the text content
        text_match = re.search(r"### TEXT CONTENT\n(.*)", block_body, re.DOTALL)
        text_content = text_match.group(1).strip() if text_match else ""
        
        if not text_content:
            continue
            
        # --- SAFE CHARACTER-BASED SLIDING WINDOW ---
        chunks = []
        text_len = len(text_content)
        
        if text_len <= MAX_CHARS_PER_CHUNK:
            chunks.append(text_content)
        else:
            start = 0
            while start < text_len:
                end = start + MAX_CHARS_PER_CHUNK
                chunk_str = text_content[start:end]
                chunks.append(chunk_str)
                
                # Advance window by step size (Size - Overlap)
                start += (MAX_CHARS_PER_CHUNK - CHARACTER_OVERLAP)
                
                # Prevent infinite loops if layout anomalies create weird index boundaries
                if (MAX_CHARS_PER_CHUNK - CHARACTER_OVERLAP) <= 0:
                    break

        # Append each chunk with identical page-level metadata
        for chunk_idx, chunk_text in enumerate(chunks):
            # Enforce a final safety truncation to prevent any hidden byte leakage
            safe_text = chunk_text[:MAX_CHARS_PER_CHUNK].strip()
            if not safe_text:
                continue
                
            parsed_chunks.append({
                "id": f"{filename}_page_{page_num.zfill(4)}_chunk_{chunk_idx:03d}",
                "text": safe_text,
                "metadata": {
                    "source_document": filename.replace(".txt", ""),
                    "page_number": int(page_num),
                    "extraction_method": method,
                    "image_path": img_path,
                    "chunk_index": chunk_idx
                }
            })
        
    return parsed_chunks

def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"[!] Input folder {INPUT_FOLDER} does not exist.")
        return

    txt_files = glob.glob(os.path.join(INPUT_FOLDER, "*.txt"))
    print(f"📚 Found {len(txt_files)} master textbook files to parse and embed.")

    for file_path in txt_files:
        filename = os.path.basename(file_path)
        print(f"\nParsing: {filename}...")
        
        chunks = parse_master_file(file_path)
        if not chunks:
            print(f"  ⚠️ No valid text structures found in {filename}. Skipping.")
            continue
            
        print(f"  🚀 Ingesting {len(chunks)} character-safe chunks into Vector DB via RTX 4060...")
        
        batch_size = 50
        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i:i + batch_size]
            
            try:
                collection.add(
                    ids=[item["id"] for item in batch],
                    documents=[item["text"] for item in batch],
                    metadatas=[item["metadata"] for item in batch]
                )
            except Exception as e:
                print(f"\n  [!] Error processing batch around item {i}: {e}")
                print("  Retrying batch sequentially...")
                for item in batch:
                    try:
                        collection.add(
                            ids=[item["id"]],
                            documents=[item["text"]],
                            metadatas=[item["metadata"]]
                        )
                    except Exception as single_e:
                        print(f"  [!] Skipping corrupted item {item['id']}: {single_e}")
            
    print("\n✅ Ingestion complete! Database is saved to:", os.path.abspath(DB_DIR))

if __name__ == "__main__":
    main()
