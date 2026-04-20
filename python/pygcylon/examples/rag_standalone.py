##
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##

'''
Standalone RAG pipeline test (no MPI required)

python python/pygcylon/examples/rag_standalone.py 200
'''

import sys
import time

import cudf
import numpy as np

from llama_index.core.embeddings import MockEmbedding
from llama_index.core.node_parser import TokenTextSplitter
import chromadb

num_docs = int(sys.argv[1]) if len(sys.argv) > 1 else 200
np.random.seed(0)
words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
         "machine", "learning", "neural", "network", "GPU", "distributed"]
texts = [" ".join(np.random.choice(words, np.random.randint(50, 200))) + "."
         for _ in range(num_docs)]

docs = cudf.DataFrame({
    "doc_id": list(range(num_docs)),
    "text": cudf.Series(texts, dtype="str"),
})

print(f"RAG pipeline (standalone): {num_docs} docs")
print("=" * 60)
print("\n[Pipeline] load -> split -> embed -> l2_normalize -> upsert")

embed_model = MockEmbedding(embed_dim=384)
splitter = TokenTextSplitter(chunk_size=50, chunk_overlap=10)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("rag_standalone")

timings = {}

# Split
t0 = time.perf_counter()
doc_ids, chunk_idxs, chunk_texts = [], [], []
for i in range(len(docs)):
    text = docs["text"].iloc[i]
    doc_id = int(docs["doc_id"].iloc[i])
    chunks = splitter.split_text(text)
    for cidx, chunk in enumerate(chunks):
        doc_ids.append(doc_id)
        chunk_idxs.append(cidx)
        chunk_texts.append(chunk)
chunks_df = cudf.DataFrame({
    "doc_id": doc_ids,
    "chunk_idx": chunk_idxs,
    "chunk_text": cudf.Series(chunk_texts, dtype="str"),
})
timings["split"] = time.perf_counter() - t0

# Embed
t0 = time.perf_counter()
texts_list = chunks_df["chunk_text"].to_pandas().tolist()
embeddings = []
batch_size = 64
for i in range(0, len(texts_list), batch_size):
    batch = texts_list[i:i + batch_size]
    batch_embs = embed_model.get_text_embedding_batch(batch)
    embeddings.extend(batch_embs)
timings["embed"] = time.perf_counter() - t0

# L2 Normalize
t0 = time.perf_counter()
normalized = []
for emb in embeddings:
    arr = np.array(emb, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    normalized.append(arr.tolist())
timings["l2_norm"] = time.perf_counter() - t0

# Upsert
t0 = time.perf_counter()
n = len(chunks_df)
ids = [f"{doc_ids[i]}_{chunk_idxs[i]}" for i in range(n)]
metadatas = [{"doc_id": doc_ids[i], "chunk_idx": chunk_idxs[i]} for i in range(n)]
upsert_batch = 256
for i in range(0, n, upsert_batch):
    end = min(i + upsert_batch, n)
    collection.upsert(
        ids=ids[i:end],
        embeddings=normalized[i:end],
        documents=texts_list[i:end],
        metadatas=metadatas[i:end],
    )
timings["upsert"] = time.perf_counter() - t0

total = sum(timings.values())
sample_norm = float(np.linalg.norm(normalized[0])) if normalized else 0.0

print(f"  split:     {timings['split']:.4f}s  ({n} chunks)")
print(f"  embed:     {timings['embed']:.4f}s  (dim=384)")
print(f"  l2_norm:   {timings['l2_norm']:.4f}s  (norm={sample_norm:.4f})")
print(f"  upsert:    {timings['upsert']:.4f}s  ({n} vectors)")
print(f"  --------------------------")
print(f"  total:     {total:.4f}s")
print(f"  throughput: {num_docs / total:.1f} docs/s")
print(f"\n[Verify] collection.count() = {collection.count()}")
