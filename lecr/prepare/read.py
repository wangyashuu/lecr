import os
import pandas as pd
import numpy as np
import ast


def read_original_data(data_path):
    original_data_path = os.path.join(
        data_path, "learning-equality-curriculum-recommendations/"
    )
    topics = (
        pd.read_csv(original_data_path + "topics.csv")
        .set_index("id")[["title", "description"]]
        .fillna("")
    )
    content = (
        pd.read_csv(original_data_path + "content.csv")
        .set_index("id")[["title", "description"]]
        .fillna("")
    )
    correlations = pd.read_csv(
        original_data_path + "correlations.csv"
    ).set_index("topic_id")[["content_ids"]]
    return topics, content, correlations


def read_generated_data(data_path, include_rands=False):
    generated_data_path = os.path.join(data_path, "learning-equality/v1/")
    correlated = pd.read_csv(
        generated_data_path + "correlated.csv",
        converters={"content_ids": ast.literal_eval},
    )
    uncorrelated_neighbors = pd.read_csv(
        generated_data_path + "uncorrelated_neighbors.csv",
        converters={"content_ids": ast.literal_eval},
    ).set_index("topic_id")
    if include_rands:
        uncorrelated_rands = pd.read_csv(
            generated_data_path + "uncorrelated_rands.csv",
            converters={"content_ids": ast.literal_eval},
        ).set_index("topic_id")

        uncorrelated = (
            pd.concat([uncorrelated_neighbors, uncorrelated_rands])
            .groupby(["topic_id"])
            .agg({"content_ids": lambda x: list(set(sum(x.tolist(), [])))})
        )
    else:
        uncorrelated = uncorrelated_neighbors
    return correlated, uncorrelated


def read_formated_data(data_path, triplet=False, include_rands=False):
    correlated, uncorrelated = read_generated_data(data_path, include_rands)

    if triplet:
        triplet_relations = (
            correlated.merge(
                uncorrelated,
                on=["topic_id"],
                how="outer",
                suffixes=("_corr", "_uncorr"),
            )
            .explode("content_ids_corr")
            .rename(columns={"content_ids_corr": "content_id_corr"})
            .explode("content_ids_uncorr")
            .rename(columns={"content_ids_uncorr": "content_id_uncorr"})
            .dropna()
            .reset_index(drop=True)
        )
        return triplet_relations

    contrastive_relations = (
        correlated.explode("content_ids")
        .rename(columns={"content_ids": "content_id"})
        .merge(
            uncorrelated.explode("content_ids").rename(
                columns={"content_ids": "content_id"}
            ),
            on=["topic_id", "content_id"],
            how="outer",
            indicator="correlated",
        )
        .reset_index(drop=True)
    )

    contrastive_relations["correlated"] = contrastive_relations[
        "correlated"
    ].map({"left_only": 1, "right_only": 0})
    return contrastive_relations
