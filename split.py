import json
import os

# Splits your 300MB file into ~75MB chunks
with open('judgements.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

chunk_size = len(data) // 4
for i in range(4):
    chunk = data[i * chunk_size : (i + 1) * chunk_size]
    with open(f'judgements_part_{i+1}.json', 'w', encoding='utf-8') as f:
        json.dump(chunk, f)