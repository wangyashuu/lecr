from box import Box
import numpy as np
from lecr.prepare.read import read_original_data
from lecr.eval.f2 import (
    compute_f2scores_for,
    compute_precisions_for,
    compute_recalls_for,
)
from sklearn.model_selection import train_test_split
from lecr.stage_1.get_possible_content import get_possible_content


default_config = Box(
    dict(
        batch_size=128,
        top_k=100,
        # model_name=(
        #     "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        # ),
        model_name="./output_v15/saved",
        dataset=dict(test_size=0.1, random_state=2023),
    )
)


def train(config, input_path):
    config = default_config + config
    topics, content, correlations = read_original_data(input_path)

    neighbors = get_possible_content(
        topics,
        content,
        correlations,
        model_name=config.model_name,
        top_k=config.top_k,
    )

    f2scores = compute_f2scores_for(correlations, neighbors)
    precisions = compute_precisions_for(correlations, neighbors)
    recalls = compute_recalls_for(correlations, neighbors)
    f2, precision, recall = (
        np.mean(f2scores),
        np.mean(precisions),
        np.mean(recalls),
    )
    print(f"all f2={f2}, precision={precision}, recall={recall}")

    train_topics, test_topics = train_test_split(
        topics,
        test_size=config.dataset.test_size,
        random_state=config.dataset.random_state,
    )

    train_correlations = correlations[
        correlations.index.isin(train_topics.index)
    ]
    test_correlations = correlations[
        correlations.index.isin(test_topics.index)
    ]
    print("corr size:", len(train_correlations), "/", len(test_correlations))

    train_f2scores = compute_f2scores_for(train_correlations, neighbors)
    train_precisions = compute_precisions_for(train_correlations, neighbors)
    train_recalls = compute_recalls_for(train_correlations, neighbors)
    train_f2, train_precision, train_recall = (
        np.mean(train_f2scores),
        np.mean(train_precisions),
        np.mean(train_recalls),
    )
    print(
        f"tr f2={train_f2} precision={train_precision} recall={train_recall}"
    )
    test_f2scores = compute_f2scores_for(test_correlations, neighbors)
    test_precisions = compute_precisions_for(test_correlations, neighbors)
    test_recalls = compute_recalls_for(test_correlations, neighbors)
    test_f2, test_precision, test_recall = (
        np.mean(test_f2scores),
        np.mean(test_precisions),
        np.mean(test_recalls),
    )
    print(f"ev f2={test_f2} precision={test_precision} recall={test_recall}")

    # return train_score, test_score


train(Box({}), input_path="./input")
