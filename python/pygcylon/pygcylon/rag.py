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

from __future__ import annotations

import cudf
import numpy as np

import pygcylon as gcy
from pygcylon.comms import shuffle
from pycylon.frame import CylonEnv

from llama_index.core.node_parser import TokenTextSplitter as _LITokenSplitter
from llama_index.core.node_parser import SentenceSplitter as _LISentenceSplitter
from llama_index.core.node_parser import SemanticSplitterNodeParser as _LISemanticSplitter


def _distribute(docs, env):
    if env is not None and env.world_size > 1:
        return shuffle(docs, env=env, on="doc_id")
    return docs


def _apply_splitter(docs, env, split_fn):
    """Distribute docs across workers, apply split_fn per doc, collect results.

    split_fn(text: str) -> list[str]
    """
    local = _distribute(docs, env).to_cudf()
    doc_ids, chunk_idxs, chunk_texts = [], [], []

    doc_ids_host = local["doc_id"].values_host
    texts_host = local["text"].to_pandas()
    for row in range(len(local)):
        text = texts_host.iloc[row]
        if not text:
            continue
        doc_id = int(doc_ids_host[row])
        chunks = split_fn(text)
        for cidx, chunk in enumerate(chunks):
            doc_ids.append(doc_id)
            chunk_idxs.append(cidx)
            chunk_texts.append(chunk)

    result = cudf.DataFrame({
        "doc_id": doc_ids,
        "chunk_idx": chunk_idxs,
        "chunk_text": cudf.Series(chunk_texts, dtype="str"),
    })
    return gcy.DataFrame.from_cudf(result)


def token_split(docs, env=None, chunk_size=1024, chunk_overlap=20, **kwargs):
    """Distributed TokenTextSplitter using LlamaIndex."""
    splitter = _LITokenSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
    return _apply_splitter(docs, env, splitter.split_text)


def sentence_split(docs, env=None, chunk_size=1024, chunk_overlap=200, **kwargs):
    """Distributed SentenceSplitter using LlamaIndex."""
    splitter = _LISentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
    return _apply_splitter(docs, env, splitter.split_text)


def recursive_split(docs, env=None, chunk_size=1000, chunk_overlap=200, separators=None):
    """Distributed RecursiveCharacterTextSplitter using LangChain via LlamaIndex."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from llama_index.core.node_parser import LangchainNodeParser

    lc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators or ["\n\n", "\n", " ", ""],
    )
    return _apply_splitter(docs, env, lc_splitter.split_text)


def semantic_split(docs, embed_model, env=None, buffer_size=1,
                   breakpoint_percentile_threshold=95, **kwargs):
    """Distributed SemanticSplitterNodeParser using LlamaIndex.

    embed_model: a LlamaIndex BaseEmbedding instance
    """
    splitter = _LISemanticSplitter.from_defaults(
        embed_model=embed_model,
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        **kwargs,
    )

    local = _distribute(docs, env).to_cudf()
    doc_ids, chunk_idxs, chunk_texts = [], [], []

    doc_ids_host = local["doc_id"].values_host
    texts_host = local["text"].to_pandas()
    for row in range(len(local)):
        text = texts_host.iloc[row]
        if not text:
            continue
        doc_id = int(doc_ids_host[row])

        from llama_index.core.schema import Document
        li_doc = Document(text=text)
        nodes = splitter.build_semantic_nodes_from_documents([li_doc])
        for cidx, node in enumerate(nodes):
            doc_ids.append(doc_id)
            chunk_idxs.append(cidx)
            chunk_texts.append(node.get_content())

    result = cudf.DataFrame({
        "doc_id": doc_ids,
        "chunk_idx": chunk_idxs,
        "chunk_text": cudf.Series(chunk_texts, dtype="str"),
    })
    return gcy.DataFrame.from_cudf(result)


def normalize(docs, env=None, unicode_form="NFC", lowercase=True,
              collapse_whitespace=True):
    """Text normalization using cuDF GPU string ops."""
    local = _distribute(docs, env).to_cudf().copy()
    col = local["text"]

    if unicode_form:
        col = col.str.normalize(unicode_form)
    if lowercase:
        col = col.str.lower()
    if collapse_whitespace:
        col = col.str.normalize_spaces().str.strip()

    local["text"] = col
    return gcy.DataFrame.from_cudf(local)


def embed(chunks, embed_model, env=None, batch_size=64):
    """Distributed batch embedding using a LlamaIndex BaseEmbedding model.

    Parameters
    ----------
    chunks : gcy.DataFrame
        Output of any splitter: columns doc_id, chunk_idx, chunk_text.
    embed_model : llama_index.core.embeddings.BaseEmbedding
        Any LlamaIndex-compatible embedding model.
    env : CylonEnv, optional
        Distributed environment. Single-process if None.
    batch_size : int
        Texts per embedding API call.

    Returns
    -------
    gcy.DataFrame
        Input columns plus an ``embedding`` list column (float32).
    """
    local = _distribute(chunks, env).to_cudf()
    texts = local["chunk_text"].to_pandas().tolist()

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embs = embed_model.get_text_embedding_batch(batch)
        embeddings.extend(batch_embs)

    local["embedding"] = cudf.Series(embeddings)
    return gcy.DataFrame.from_cudf(local)


def l2_normalize(embedded, env=None):
    """L2 normalize embedding vectors for cosine similarity."""
    local = _distribute(embedded, env).to_cudf()
    embs = local["embedding"].to_pandas().tolist()

    normalized = []
    for emb in embs:
        arr = np.array(emb, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        normalized.append(arr.tolist())

    local["embedding"] = cudf.Series(normalized)
    return gcy.DataFrame.from_cudf(local)


def upsert(embedded, collection, env=None, batch_size=256):
    """Batch upsert embeddings to a ChromaDB collection.

    Parameters
    ----------
    embedded : gcy.DataFrame
        Output of embed() or l2_normalize(): columns doc_id, chunk_idx, chunk_text, embedding.
    collection : chromadb.Collection
        A ChromaDB collection to upsert into.
    env : CylonEnv, optional
        Distributed environment. Single-process if None.
    batch_size : int
        Vectors per upsert call.

    Returns
    -------
    int
        Number of vectors upserted by this worker.
    """
    local = _distribute(embedded, env).to_cudf()
    n = len(local)
    if n == 0:
        return 0

    ids = [f"{int(local['doc_id'].iloc[i])}_{int(local['chunk_idx'].iloc[i])}" for i in range(n)]
    texts = local["chunk_text"].to_pandas().tolist()
    embs = local["embedding"].to_pandas().tolist()
    metadatas = [{"doc_id": int(local["doc_id"].iloc[i]), "chunk_idx": int(local["chunk_idx"].iloc[i])} for i in range(n)]

    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        collection.upsert(
            ids=ids[i:end],
            embeddings=embs[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end],
        )

    return n
