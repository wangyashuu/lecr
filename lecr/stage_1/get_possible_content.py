import pandas as pd

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
import cupy
from cuml.neighbors import NearestNeighbors


def get_neighbors(
    query_embeddings,
    corpus_embeddings,
    top_k=10,
):
    ## get n_neighbors
    query_embeddings_cupy = cupy.array(query_embeddings)
    corpus_embeddings_cupy = cupy.array(corpus_embeddings)
    neighbors_model = NearestNeighbors(n_neighbors=top_k, metric="cosine")
    neighbors_model.fit(corpus_embeddings_cupy)
    kneighbors_indices = neighbors_model.kneighbors(
        query_embeddings_cupy, return_distance=False
    )
    kneighbors_indices = [
        indices.get() for i, indices in enumerate(kneighbors_indices)
    ]
    return kneighbors_indices


def search(query_embeddings, corpus_embeddings, top_k, semantic=True):
    if semantic:
        results = semantic_search(
            query_embeddings, corpus_embeddings, top_k=top_k
        )
        indices = [[item["corpus_id"] for item in r] for r in results]
        return indices
    else:
        return get_neighbors(query_embeddings, corpus_embeddings, top_k)


def get_possible_content(
    topics,
    content,
    correlations,
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    top_k=10,
):
    ## get embeddings
    topic_sentences = topics.loc[correlations.index]["title"]
    content_sentences = content["title"]
    model = SentenceTransformer(model_name)
    topic_embeddings = model.encode(
        topic_sentences, show_progress_bar=True, convert_to_tensor=True
    )
    content_embeddings = model.encode(
        content_sentences, show_progress_bar=True, convert_to_tensor=True
    )
    top_k_indices = search(topic_embeddings, content_embeddings, top_k=top_k)
    content_ids = [
        [content.iloc[idx].name for idx in indices]
        for indices in top_k_indices
    ]
    predictions = pd.DataFrame(
        {"topic_id": correlations.index, "content_ids": content_ids}
    ).set_index("topic_id")
    return predictions
