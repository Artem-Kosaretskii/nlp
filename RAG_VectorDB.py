import numpy as np
import faiss
import os
# faiss-gpu package


def get_index_size(index):
    faiss.write_index(index, 'temp.index')
    index_size = os.path.getsize('temp.index')
    os.remove('temp.index')
    return index_size


def main():
    d = 256
    nb = 100000
    nq = 10000

    np.random.seed(1234)
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.

    index = faiss.IndexFlatL2(d)
    print(index.is_trained)
    index.add(xb)
    print(index.ntotal)

    k = 4
    D, I = index.search(xb[:5], k)
    print(f'FlatL2:\n {I}')
    print(D)

    nlist = 100

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)

    assert not index.is_trained
    index.train(xb)
    assert index.is_trained

    xq = xb[:5]
    index.add(xb)
    D, I = index.search(xq, k)
    print(f'IFV:\n {I[-5:]}')
    index.nprobe = 10
    D, I = index.search(xq, k)
    print(f'IFV nprobe = 10:\n {I[-5:]}')

    M = 64
    ef_search = 16
    ef_construction = 32
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(xb)
    D, I = index.search(xq, k)
    print(f'HNSW:\n {I[:5]}')


    index_l2 = faiss.IndexFlatL2(d)
    index_l2.add(xb)
    index_l2_size = get_index_size(index_l2)
    M = 16
    assert d % M == 0
    nbits = 8
    index_pq = faiss.IndexPQ(d, M, nbits)
    index_pq.train(xb)
    index_pq_size = get_index_size(index_pq)
    print(f"Ratio PQ/L2 (M=16, nbits=8): {index_pq_size / index_l2_size:.4f}")

    index_l2 = faiss.IndexFlatL2(d)
    index_l2.add(xb)
    index_l2_size = get_index_size(index_l2)
    M = 32
    assert d % M == 0
    nbits = 8
    index_pq = faiss.IndexPQ(d, M, nbits)
    index_pq.train(xb)
    index_pq_size = get_index_size(index_pq)
    print(f"Ratio PQ/L2 (M=32, nbits=8): {index_pq_size / index_l2_size:.4f}")


if __name__ == '__main__':
    main()
