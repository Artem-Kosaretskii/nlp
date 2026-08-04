import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer


def get_sim_matrix(a, b):
    sim_matrix = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            sim = 1 - cosine(a[i], b[j])
            sim_matrix[i,j] = sim
    return sim_matrix


def semantic_alignmen(src, trg, model):
    srcs = src.split(".")
    trgs = trg.split(".")

    src_embeds = model.encode(srcs)
    trg_embeds = model.encode(trgs)
    sim_matrix = get_sim_matrix(src_embeds, trg_embeds)
    np.testing.assert_array_equal(sim_matrix.argmax(1), np.arange(len(src_embeds)))


def main():
    model_name = 'distiluse-base-multilingual-cased'
    model = SentenceTransformer(model_name)
    src = "there must be a source text"
    trg = "there must be a target text"
    semantic_alignmen(src, trg, model)


if __name__ == 'main':
    main()
