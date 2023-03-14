import pandas as pd

from sentence_transformers import SentenceTransformer
import cupy
from cuml.neighbors import NearestNeighbors


def get_neighbors(
    topics,
    content,
    correlations,
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    n_neighbors=10,
):
    ## get embeddings
    topic_sentences = topics.loc[correlations.index]["title"]
    content_sentences = content["title"]
    model = SentenceTransformer(model_name)
    topic_embeddings = model.encode(topic_sentences, show_progress_bar=True)
    content_embeddings = model.encode(
        content_sentences, show_progress_bar=True
    )

    ## get n_neighbors
    topic_embeddings_cupy = cupy.array(topic_embeddings)
    content_embeddings_cupy = cupy.array(content_embeddings)
    neighbors_model = NearestNeighbors(
        n_neighbors=n_neighbors, metric="cosine"
    )
    neighbors_model.fit(content_embeddings_cupy)
    kneighbors_indices = neighbors_model.kneighbors(
        topic_embeddings_cupy, return_distance=False
    )
    content_ids = [
        [content.iloc[idx].name for idx in indices.get()]
        for i, indices in enumerate(kneighbors_indices)
    ]
    predictions = pd.DataFrame(
        {"topic_id": correlations.index, "content_ids": content_ids}
    ).set_index("topic_id")
    return predictions
