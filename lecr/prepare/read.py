import os
import pandas as pd
import numpy as np
import ast


def read_original_data(data_path, detailed=False):
    original_data_path = os.path.join(
        data_path, "learning-equality-curriculum-recommendations/"
    )
    topics = pd.read_csv(original_data_path + "topics.csv").set_index("id")
    content = pd.read_csv(original_data_path + "content.csv").set_index("id")

    if detailed:
        topic_cols = [
            topics[["title", "description"]],
            topics[["has_content"]].astype(int),
        ]
        topic_cols += [
            pd.get_dummies(topics[col], prefix=col)
            for col in ["channel", "category", "language"]
        ]
        topics = pd.concat(topic_cols, axis=1)
        content_cols = [content[["title", "description"]]]
        content_cols += [
            pd.get_dummies(content[col], prefix=col)
            for col in ["kind", "language"]
        ]
        content = pd.concat(content_cols, axis=1)

    else:
        topics = topics[["title", "description"]]
        content = content[["title", "description"]]

    topics.fillna({"title": "", "description": ""}, inplace=True)
    content.fillna({"title": "", "description": ""}, inplace=True)

    correlations = pd.read_csv(
        original_data_path + "correlations.csv"
    ).set_index("topic_id")[["content_ids"]]
    correlations.content_ids = correlations.content_ids.str.split()
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


def to_triplet_relations(correlated, uncorrelated):
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


def to_contrastive_relations(correlated, uncorrelated):
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


def read_formated_data(
    data_path,
    format="mnr",
    include_rands=False,
    train_topics=None,
    test_topics=None,
):
    correlated, uncorrelated = read_generated_data(data_path, include_rands)
    topics, content, _ = read_original_data(data_path)
    if format == "mnr":
        all_set = correlated.explode("content_ids").rename(
            columns={"content_ids": "content_id"}
        )
        if train_topics is not None and test_topics is not None:
            eval_set = correlated.merge(
                uncorrelated,
                on=["topic_id"],
                # how="outer",
                suffixes=("_corr", "_uncorr"),
            )
            train_set = all_set[all_set["topic_id"].isin(train_topics.index)]
            test_set = eval_set[eval_set["topic_id"].isin(test_topics.index)]
            samples = pd.DataFrame(
                {
                    "query": topics.loc[test_set["topic_id"]]["title"].tolist(),
                    "positive": test_set["content_ids_corr"].map(
                        lambda ids: content.loc[ids]["title"].tolist()
                    ),
                    "negative": test_set["content_ids_uncorr"].map(
                        lambda ids: content.loc[ids]["title"].tolist()
                    ),
                }
            )
            samples = samples.to_dict('records')
            return train_set, samples
        return all_set
    elif format == "triplet":
        all_set = to_triplet_relations(correlated, uncorrelated)
        if train_topics is not None and test_topics is not None:
            train_set = all_set[all_set["topic_id"].isin(train_topics.index)]
            test_set = all_set[all_set["topic_id"].isin(test_topics.index)]
            test_data = [
                topics.loc[test_set["topic_id"]]["title"].tolist(),
                content.loc[test_set["content_id_corr"]]["title"].tolist(),
                content.loc[test_set["content_id_uncorr"]]["title"].tolist(),
            ]
            return train_set, test_data
        return all_set
    else:
        all_set = to_contrastive_relations(correlated, uncorrelated)
        if train_topics is not None and test_topics is not None:
            train_set = all_set[all_set["topic_id"].isin(train_topics.index)]
            test_set = all_set[all_set["topic_id"].isin(test_topics.index)]
            test_data = [
                topics.loc[test_set["topic_id"]]["title"].tolist(),
                content.loc[test_set["content_id"]]["title"].tolist(),
                test_set["correlated"].tolist(),
            ]
            return train_set, test_data
        return all_set


def get_formated_neighbors(data_path, neighbors):
    generated_data_path = os.path.join(data_path, "learning-equality/v1/")
    correlated = pd.read_csv(
        generated_data_path + "correlated.csv",
        converters={"content_ids": ast.literal_eval},
    )

    contrastive_relations = (
        correlated.explode("content_ids")
        .rename(columns={"content_ids": "content_id"})
        .merge(
            neighbors.explode("content_ids").rename(
                columns={"content_ids": "content_id"}
            ),
            on=["topic_id", "content_id"],
            how="outer",
            indicator="correlated",
        )
        .reset_index(drop=True)
    )
    contrastive_relations = contrastive_relations[
        contrastive_relations["correlated"] != "left_only"
    ]

    contrastive_relations["correlated"] = contrastive_relations[
        "correlated"
    ].map({"right_only": 0, "both": 1})
    contrastive_relations = contrastive_relations.reset_index(drop=True)

    return contrastive_relations
