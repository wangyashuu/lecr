import os
import pandas as pd
import numpy as np

data_path = "./input"

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

correlations = pd.read_csv(original_data_path + "correlations.csv").set_index(
    "topic_id"
)[["content_ids"]]

unsupervised = pd.read_csv(data_path + "/public_generated/unsupervised.csv")

generated_data_path = original_data_path = os.path.join(
    data_path, "learning-equality/"
)

## corr
train_rate = 0.7
total_size = len(correlations)

train_size = int(total_size * train_rate)
train_indices = np.random.choice(total_size, size=train_size)
val_indices = np.delete(np.arange(total_size), train_indices)


def explode_correlations(correlations):
    correlations.content_ids = correlations.content_ids.str.split()
    correlations = correlations.explode("content_ids").rename(
        columns={"content_ids": "content_id"}
    )
    return correlations


exploded_correlations = explode_correlations(correlations.copy())


def get_data(correlations_df, content, neighbors, indices):
    rand_data = []
    uns_data = []
    corr_data = []
    content_len = len(content)

    for i, idx in enumerate(indices):
        tid = correlations_df.iloc[idx]["topic_id"]

        rand_cindices = np.random.choice(content_len, size=5)
        rand_crows = content.iloc[rand_cids]
        rand_data += [
            [tid, content.iloc[cidx]["id"]] for cidx in rand_cindices
        ]

        rows = neighbors[neighbors["topics_ids"] == tid]
        uns_data += [
            [row["topics_ids"], row["content_ids"]]
            for _, row in rows.iterrows()
        ]

        corr_data += [
            [tid, cid]
            for cid in correlations_df.iloc[idx]["content_ids"].split(" ")
        ]

        if (i + 1) % 1000 == 0:
            print("processing:", i)
    return rand_data, uns_data, corr_data


train_rand, train_uns, train_corr = get_data(
    exploded_correlations, content, unsupervised, train_indices
)

val_rand, val_uns, val_corr = get_data(
    exploded_correlations, content, unsupervised, val_indices
)
train_rand.to_csv(generated_data_path + "v0/train_rand.csv")
train_uns.to_csv(generated_data_path + "v0/train_uns.csv")
train_corr.to_csv(generated_data_path + "v0/train_corr.csv")
val_rand.to_csv(generated_data_path + "v0/val_rand.csv")
val_uns.to_csv(generated_data_path + "v0/val_uns.csv")
val_corr.to_csv(generated_data_path + "v0/val_corr.csv")

## uncorr
uncorr_tids = list(set(topics["id"]) - set(correlations["topic_id"]))

train_rate = 0.7
total_size = len(uncorr_tids)

train_size = int(total_size * train_rate)
train_indices = np.random.choice(total_size, size=train_size)
val_indices = np.delete(np.arange(total_size), train_indices)


def get_uncorr_data(uncorr_tids, content, neighbors, indices):
    rand_data = []
    uns_data = []
    content_len = len(content)

    for i, idx in enumerate(indices):
        tid = uncorr_tids[idx]

        rand_cindices = np.random.choice(content_len, size=5)
        rand_crows = content.iloc[rand_cids]
        rand_data += [
            [tid, content.iloc[cidx]["id"]] for cidx in rand_cindices
        ]

        rows = neighbors[neighbors["topics_ids"] == tid]
        uns_data += [
            [row["topics_ids"], row["content_ids"]]
            for _, row in rows.iterrows()
        ]
        if (i + 1) % 1000 == 0:
            print("processing:", i)
    return rand_data, uns_data


train_unrand_data, train_ununs_data = get_uncorr_data(
    uncorr_tids, content, unsupervised, train_indices
)

train_unrand_data = pd.DataFrame(
    train_unrand_data, columns=["topic_id", "content_id"]
)
train_ununs_data = pd.DataFrame(
    train_ununs_data, columns=["topic_id", "content_id"]
)
train_uncorr_data = train_unrand_data.merge(
    train_ununs_data,
    on=["topic_id", "content_id"],
    how="outer",
)
train_uncorr_data.to_csv("train_uncorr_data.csv")

val_unrand_data, val_ununs_data = get_uncorr_data(
    uncorr_tids, content, unsupervised, val_indices
)

val_unrand_data = pd.DataFrame(
    val_unrand_data, columns=["topic_id", "content_id"]
)
val_ununs_data = pd.DataFrame(
    val_ununs_data, columns=["topic_id", "content_id"]
)
val_uncorr_data = val_unrand_data.merge(
    val_ununs_data,
    on=["topic_id", "content_id"],
    how="outer",
)
val_uncorr_data.to_csv("val_uncorr_data.csv")
