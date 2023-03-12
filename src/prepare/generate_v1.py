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

generated_data_path = os.path.join(data_path, "learning-equality/")

## correlated
correlations = (
    pd.read_csv(original_data_path + "correlations.csv")
    .reset_index()[["topic_id", "content_ids"]]
    .set_index("topic_id")
)
correlations.content_ids = correlations.content_ids.str.split()
correlations.to_csv(generated_data_path + "v1/correlated.csv")


## uncorrelated_neighbors
unsupervised = pd.read_csv(data_path + "/public_generated/unsupervised.csv")
uncorrelated_neighbors = (
    unsupervised[unsupervised["target"] == 0]
    .rename({"topics_ids": "topic_id"}, axis=1)[["topic_id", "content_ids"]]
    .groupby(["topic_id"])
    .agg({"content_ids": lambda x: x.tolist()})
)
uncorrelated_neighbors.to_csv(
    generated_data_path + "v1/uncorrelated_neighbors.csv"
)

## uncorrelated_rands
rand_len = 5
content_len = len(content)
uncorrelated_rands = []
for idx in range(len(topics)):
    tid = topics.iloc[idx].name
    cids = (
        correlations.loc[tid]["content_ids"]
        if tid in correlations.index
        else []
    )
    rand_cindices = np.random.choice(content_len, size=5)
    rand_cids = [
        content.iloc[cidx].name
        for cidx in rand_cindices
        if content.iloc[cidx].name not in cids
    ]
    uncorrelated_rands.append([tid, rand_cids])

uncorrelated_rands = pd.DataFrame(
    uncorrelated_rands, columns=["topic_id", "content_ids"]
).set_index("topic_id")
uncorrelated_rands.to_csv(generated_data_path + "v1/uncorrelated_rands.csv")
