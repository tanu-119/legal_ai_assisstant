import json
import os

def split_json(filename, chunk_size_mb=75):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Calculate chunks
    total_size = os.path.getsize(filename)
    num_chunks = (total_size // (chunk_size_mb * 1024 * 1024)) + 1
    items_per_chunk = len(data) // int(num_chunks) + 1
    
    for i in range(0, len(data), items_per_chunk):
        chunk = data[i:i + items_per_chunk]
        chunk_num = (i // items_per_chunk) + 1
        chunk_name = f"judgements_part_{chunk_num}.json"
        with open(chunk_name, 'w', encoding='utf-8') as f:
            json.dump(chunk, f)
        print(f"Created {chunk_name}")

if __name__ == "__main__":
    split_json('judgements.json')