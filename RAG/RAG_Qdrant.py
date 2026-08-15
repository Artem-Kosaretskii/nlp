import re
import os
import time
import uuid
import torch
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from typing import List, Optional, Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from sklearn.feature_extraction.text import TfidfVectorizer


class RAG:

    def __init__(
        self,
        dense_model: Any,
        sparse_model: Any,
        coll_name: str = '',
        emb_dim: int = 1024,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        device: Optional[str] = 'cuda',
        db_url: str = "http://localhost:6333",
        multi: bool = True
    ):
        self.dbclient = QdrantClient(url=db_url)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.index = None
        self.multivector = multi
        if len(coll_name) != 0:
            self.collection_name = coll_name
        else:
            self.collection_name = f'arxiv_{int(time.time())}'
            if self.multivector:
                vector_params = {
                    "dense_vector": models.VectorParams(
                        size=emb_dim,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM,
                        ),
                        hnsw_config=models.HnswConfigDiff(m=0),
                        on_disk=True,
                )}
            else:
                vector_params = {
                    "dense_vector": models.VectorParams(
                        size=emb_dim,
                        distance=models.Distance.COSINE,
                )}
            self.collection = self.dbclient.create_collection(
                collection_name=self.collection_name,
                vectors_config=vector_params,
                sparse_vectors_config={"bm25_sparse_vector": models.SparseVectorParams(modifier=models.Modifier.IDF)}
            )
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model)
        self.dense_model = SentenceTransformer(model_name_or_path=dense_model)


    def build_db(self, docs: pd.DataFrame,  batch_size: int = 8, limit: int = 0) -> None:
        double_p_1 = '([\s\n]{1,}[\s]{1,})|(\n)|(;)'
        double_p_2 = '(,{2,})|(\.{2,})|(-{2,})'
        build_range = docs.shape[0] if limit == 0 else limit
        for i in range(build_range):
            points = []
            if i % 500 == 0:
                print(f'line {i}')
            line = docs.iloc[i, :]
            abstract = line['abstract']
            tte = re.sub(double_p_1, ' ', abstract)
            tte = re.sub(double_p_2, '', tte)
            tte = tte.lower()
            chunks = self.text_splitter.split_documents([Document(tte)])
            if self.multivector:
                dense_vector = []
                sparse = list(self.sparse_model.embed(tte))[0]
                for chunk in chunks:
                    dense_vector.append(self.dense_model.encode(chunk.page_content))
                point = PointStruct(
                    id=uuid.uuid4().hex,
                    payload={
                        'content': abstract,
                        'metadata': {
                            'id': line['id'],
                            'title': line['title'],
                            'authors': line['authors']
                        }
                    },
                    vector={
                        "dense_vector": dense_vector,
                        "bm25_sparse_vector": {'indices': sparse.indices, 'values': sparse.values},
                    }
                )
                points.append(point)
            else:
                for chunk in chunks:
                    sparse = list(self.sparse_model.embed(chunk.page_content))[0]
                    point = PointStruct(
                        id=uuid.uuid4().hex,
                        payload={
                            'content': abstract,
                            'metadata': {
                                'id': line['id'],
                                'title': line['title'],
                                'authors': line['authors']
                            }
                        },
                        vector={
                            "dense_vector": self.dense_model.encode(chunk.page_content),
                            "bm25_sparse_vector": {'indices': sparse.indices, 'values': sparse.values},
                        }
                    )
                    points.append(point)
            self.dbclient.upload_points(collection_name=self.collection_name, points=points, batch_size=batch_size)


def main(mmr_count=True, verbose=False, test_mode=False, visualise=False):
    load_dotenv()
    test_path = "./nlp_s3_data/test_sample.csv"
    data_path = "./nlp_s3_data/arxiv-metadata-s.json"
    ch_size = 256
    ch_over = 64
    emb_dim = 384 # 1024
    multivector = True
    sparse_model = "Qdrant/bm25"
    # dense_model = "Qwen/Qwen3-Embedding-0.6B" # 1024
    dense_model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' # 384
    K = 5 # for MRR@K
    dbsf = True

    rag = RAG(
        chunk_size=ch_size,
        chunk_overlap=ch_over,
        emb_dim=emb_dim,
        multi=multivector,
        dense_model=dense_model,
        sparse_model=sparse_model
    )

    docs = pd.read_json(data_path)
    test_docs = pd.read_csv(test_path)

    if visualise:
        import matplotlib.pyplot as plt
        print(f'\nData columns: {", ".join(list(docs.columns))}')
        print(f'Articles number: {len(docs)}')
        sorted_docs = docs.sort_values(by="abstract", key=lambda x: x.str.len())
        print(f"Shortest abstract: {len(sorted_docs.iloc[0, :]['abstract'])}")
        print(f"Longest abstract: {len(sorted_docs.iloc[-1, :]['abstract'])}")
        print(f'\nTest data columns: {", ".join(list(test_docs.columns))}')
        sorted_queries = test_docs.sort_values(by="query", key=lambda x: x.str.len())
        print(f"Shortest query: {len(sorted_queries.iloc[0, :]['query'])}")
        print(f"Longest query: {len(sorted_queries.iloc[-1, :]['query'])}")
        lengths = sorted_docs.abstract.str.len()
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].scatter(lengths.values, lengths.index, s=2, alpha=0.5)
        lengths = sorted_queries['query'].str.len()
        axes[1].scatter(lengths.values, lengths.index, s=2, alpha=0.5)
        plt.show()

    if not test_mode:
        rag.build_db(docs)
    else:
        test_docs.set_index(test_docs.id.values, inplace=True)
        docs.set_index(docs.id.values, inplace=True)
        exist = docs.loc[docs.index.isin(test_docs.index)]
        not_exist = docs.loc[~docs.index.isin(test_docs.index)]
        rag.build_db(exist)
        rag.build_db(not_exist.iloc[:5000, :])

    print(f'DB size: {rag.dbclient.count(collection_name=rag.collection_name)}')

    q_sim = 0.
    c_sim = 0.
    mmr_sum = 0.
    for i in range(test_docs.shape[0]):
        line = test_docs.iloc[i, :]
        query = line['query'].lower()
        id = line['id']
        abstract = line['abstract']
        results = rag.dbclient.query_points(
            collection_name=rag.collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model=dense_model
                    ),
                    using="dense_vector",
                    limit=K
                ),
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model=sparse_model
                    ),
                    using="bm25_sparse_vector",
                    limit=K
                )
            ],
            query=(models.FusionQuery(fusion=models.Fusion.DBSF) if dbsf == True else models.FusionQuery(fusion=models.Fusion.RRF)),
            limit=K,
            with_payload=True
        )

        print('\n')
        print('-' * 50)
        print(f'id: {id}, query: {query}')
        mmr_current = 0.
        best_sim = 0.
        counter = 1
        for p in results.points:
            print(f'score: {p.score}')
            print(f"retrieved metadata id: {p.payload['metadata']['id']}")
            vect = TfidfVectorizer(min_df=1, stop_words="english")
            tfidf = vect.fit_transform([abstract, p.payload['content']])
            pws = np.average((tfidf * tfidf.T).data)
            if pws > best_sim:
                best_sim = pws
                mmr_current = counter
            counter += 1
            if verbose:
                print(p.payload['content'])
                print(f'pairwise_similarity: {pws:.4f}')
            if not mmr_count:
                q_sim += p.score
                c_sim += pws
                break
        mmr_sum += 1 / mmr_current

    if mmr_count:
        print(f'MMR@{K} : {(mmr_sum / test_docs.shape[0]):.4f}')
    else:
        print(f'Average cosine similarity query_text/retrieved_content : {(q_sim / test_docs.shape[0]):.4f}')
        print(f'Average cosine similarity test_content/retrieved_content : {(c_sim / test_docs.shape[0]):.4f}')


if __name__ == '__main__':
        main(mmr_count=True, verbose=False, test_mode=False)
