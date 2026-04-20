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
RAG ingestion pipeline demo: Load -> Split -> Embed -> L2 Normalize -> Upsert

# local (single GPU):
python python/pygcylon/examples/rag.py

# distributed:
mpirun -n 2 --mca opal_cuda_support 1 \
    python python/pygcylon/examples/rag.py
'''

import sys
import time

import cudf
import numpy as np
import pycylon as cy
import pygcylon as gcy
from pygcylon.rag import token_split, embed, l2_normalize, upsert

from llama_index.core.embeddings import MockEmbedding
import chromadb

env = cy.CylonEnv(config=cy.MPIConfig(), distributed=True)
rank, ws = env.rank, env.world_size

num_docs = int(sys.argv[1]) if len(sys.argv) > 1 else 200
np.random.seed(0)
words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
         "machine", "learning", "neural", "network", "GPU", "distributed"]
texts = [" ".join(np.random.choice(words, np.random.randint(50, 200))) + "."
         for _ in range(num_docs)]
docs = gcy.DataFrame.from_cudf(cudf.DataFrame({
    "doc_id": list(range(num_docs)),
    "text": cudf.Series(texts, dtype="str"),
}))

if rank == 0:
    print(f"RAG pipeline: {ws} worker(s), {num_docs} docs")
    print("=" * 60)
    print("\n[Pipeline] load -> split -> embed -> l2_normalize -> upsert")

embed_model = MockEmbedding(embed_dim=384)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(f"rag_demo_{rank}")

timings = {}

t0 = time.perf_counter()
chunks = token_split(docs, env, chunk_size=50)
timings["split"] = time.perf_counter() - t0

t0 = time.perf_counter()
embedded = embed(chunks, embed_model, env, batch_size=64)
timings["embed"] = time.perf_counter() - t0

t0 = time.perf_counter()
normalized = l2_normalize(embedded, env)
timings["l2_norm"] = time.perf_counter() - t0

t0 = time.perf_counter()
n_upserted = upsert(normalized, collection, env, batch_size=256)
timings["upsert"] = time.perf_counter() - t0

total = sum(timings.values())

if rank == 0:
    n_chunks = len(normalized.to_cudf())
    sample_emb = normalized.to_cudf()["embedding"].iloc[0]
    emb_norm = float(np.linalg.norm(sample_emb)) if sample_emb is not None else 0.0
    print(f"  split:     {timings['split']:.4f}s  ({n_chunks} chunks)")
    print(f"  embed:     {timings['embed']:.4f}s  (dim=384)")
    print(f"  l2_norm:   {timings['l2_norm']:.4f}s  (norm={emb_norm:.4f})")
    print(f"  upsert:    {timings['upsert']:.4f}s  ({n_upserted} vectors)")
    print(f"  --------------------------")
    print(f"  total:     {total:.4f}s")
    print(f"  throughput: {num_docs / total:.1f} docs/s")
    print(f"\n[Verify] collection.count() = {collection.count()}")

env.finalize()
