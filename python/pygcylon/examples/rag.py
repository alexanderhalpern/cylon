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
from pygcylon.rag import token_split, sentence_split, recursive_split, normalize

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

ops = {
    "token_split": lambda: token_split(docs, env, chunk_size=50),
    "sentence_split": lambda: sentence_split(docs, env, chunk_size=200),
    "recursive_split": lambda: recursive_split(docs, env, chunk_size=200),
    "normalize": lambda: normalize(docs, env),
}

if rank == 0:
    print(f"RAG ops benchmark: {ws} worker(s), {num_docs} docs")
    print(f"{'op':<25} {'time(s)':<10} {'docs/s':<10}")
    print("-" * 45)

for name, fn in ops.items():
    fn()
    t0 = time.perf_counter()
    result = fn()
    t = time.perf_counter() - t0
    if rank == 0:
        n = len(result.to_cudf())
        print(f"{name:<25} {t:<10.4f} {num_docs/t:<10.1f}  ({n} chunks)")

env.finalize()
