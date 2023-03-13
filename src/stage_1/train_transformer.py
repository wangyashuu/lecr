from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import (
    BinaryClassificationEvaluator,
    TripletEvaluator,
)


from prepare.read import read_original_data, read_generated_data


# https://huggingface.co/blog/how-to-train-sentence-transformers


class TCRelationDataset(Dataset):
    def __init__(self, dataset, text_dicts, triplet=False):
        self.dataset = dataset
        self.text_dicts = text_dicts
        self.triplet = triplet

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tid, column_val1, column_val2 = self.dataset.iloc[idx]
        t = self.text_dicts["topic"][tid]["title"]
        x1 = self.text_dicts["content"][column_val1]["title"]
        if not self.triplet:
            y = column_val2
            return InputExample(texts=[t, x1], label=y)
        x2 = self.text_dicts["content"][column_val2]["title"]
        return InputExample(texts=[t, x1, x2])


def create_datasets(config, input_path):
    topics, content, _ = read_original_data(input_path)
    all_set = read_generated_data(input_path, triplet=config.triplet)
    # all_set = all_set.iloc[:3000]
    print("dataset size:", len(all_set))
    text_dicts = dict(topic=topics.loc, content=content.loc)
    train_set, test_set = train_test_split(
        all_set, test_size=config.test_size, stratify=all_set["topic_id"]
    )  #
    train_dset = TCRelationDataset(
        train_set,
        text_dicts=text_dicts,
        triplet=config.triplet,
    )
    test_data_for_evaluator = (
        [
            topics.loc[test_set["topic_id"]]["title"].tolist(),
            content.loc[test_set["content_id_corr"]]["title"].tolist(),
            content.loc[test_set["content_id_uncorr"]]["title"].tolist(),
        ]
        if config.triplet
        else [
            topics.loc[test_set["topic_id"]]["title"].tolist(),
            content.loc[test_set["content_id"]]["title"].tolist(),
            test_set["correlated"].tolist(),
        ]
    )
    return train_dset, test_data_for_evaluator


def train(config, input_path):
    train_set, test_data = create_datasets(
        config.dataset,
        input_path=input_path,
    )
    train_loader = DataLoader(train_set, **config.train_loader)
    # test_loader = DataLoader(test_set, **config.val_loader)

    # 10% of train data
    warmup_steps = int(len(train_loader) * config.trainer.max_epochs * 0.1)
    model = SentenceTransformer(config.model_name, device="cuda")
    train_loss = getattr(
        losses,
        config.loss.name,
        "TripletLoss" if config.dataset.triplet else "ContrastiveLoss",
    )(model, **config.loss.params)

    evaluator_params = dict(
        name=config.loss.name,
        batch_size=config.val_loader.batch_size,
        show_progress_bar=True,
        write_csv=True,
    )

    evaluator = (
        TripletEvaluator(
            *test_data, main_distance_function=0, **evaluator_params
        )
        if config.dataset.triplet
        else BinaryClassificationEvaluator(*test_data, **evaluator_params)
    )

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=config.trainer.max_epochs,
        use_amp=config.trainer.use_amp,
        warmup_steps=warmup_steps,
        output_path="./output/saved",
        checkpoint_path="./output/checkpoint",
        evaluator=evaluator,
        evaluation_steps=500,
    )
    return model
