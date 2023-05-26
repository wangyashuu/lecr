def get_tp(y_truth, y_pred):
    tp = set(y_truth) & set(y_pred)
    return tp


def fbeta_score(y_truth, y_pred, beta=2, eps=1e-16):
    tp = get_tp(y_truth, y_pred)
    precision = len(tp) / (len(y_pred) + eps)
    recall = len(tp) / len(y_truth)
    f = (
        (1 + beta**2)
        * (precision * recall)
        / ((beta**2) * precision + recall + eps)
    )
    return f


def precision_score(y_truth, y_pred, eps=1e-16):
    tp = get_tp(y_truth, y_pred)
    precision = len(tp) / (len(y_pred) + eps)
    return precision


def recall_score(y_truth, y_pred):
    tp = get_tp(y_truth, y_pred)
    recall = len(tp) / len(y_truth)
    return recall


def compute_precisions_for(csv1, csv2):
    return [
        precision_score(r["content_ids"], csv2.loc[id]["content_ids"])
        for id, r in csv1.iterrows()
    ]


def compute_recalls_for(csv1, csv2):
    return [
        recall_score(r["content_ids"], csv2.loc[id]["content_ids"])
        for id, r in csv1.iterrows()
    ]


def compute_f2scores_for(csv1, csv2):
    return [
        fbeta_score(r["content_ids"], csv2.loc[id]["content_ids"])
        for id, r in csv1.iterrows()
    ]
