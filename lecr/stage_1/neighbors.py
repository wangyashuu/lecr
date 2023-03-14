import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from box import Box

from prepare.read import read_original_data
from get_neighbors import get_neighbors

config = Box(
    dict(
        batch_size=64,
        n_neighbors=10,
        # model_name=(
        #     "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        # ),
        model_name="./out/output_v1/saved",
        dataset=dict(test_size=0.1, random_state=2023),
    )
)


## measure


def fbeta_score(y_truth, y_pred, beta=2, eps=1e-10):
    tp = set(y_truth) & set(y_pred)
    precision = len(tp) / len(y_pred)
    recall = len(tp) / len(y_truth)
    f = (
        (1 + beta**2)
        * (precision * recall)
        / ((beta**2) * precision + recall + eps)
    )
    return f


def compute_f2_for(csv1, csv2):
    return [
        fbeta_score(r["content_ids"], csv2.loc[id]["content_ids"])
        for id, r in csv1.iterrows()
    ]


def train(config, input_path):
    topics, content, correlations = read_original_data(input_path)

    neighbors = get_neighbors(
        topics,
        content,
        correlations,
        model_name=config.model_name,
        n_neighbors=config.n_neighbors,
    )

    ## compute train, test score
    correlations.content_ids = correlations.content_ids.str.split()
    scores = compute_f2_for(correlations, neighbors)
    # train_topics, test_topics = train_test_split(
    #     topics,
    #     test_size=config.dataset.test_size,
    #     random_state=config.dataset.random_state,
    # )

    # train_correlations = correlations[
    #     correlations.index.isin(train_topics.index)
    # ]
    # test_correlations = correlations[
    #     correlations.index.isin(test_topics.index)
    # ]
    # print("corr size:", len(train_correlations), "/", len(test_correlations))

    # train_score = compute_f2_for(train_correlations, predictions)
    # test_score = compute_f2_for(test_correlations, predictions)

    return train_score, test_score


train(config, input_path="./input")
