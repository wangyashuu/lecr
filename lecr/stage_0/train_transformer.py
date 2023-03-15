from sklearn.model_selection import train_test_split
import numpy as np
from box import Box
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
)


from prepare.read import read_original_data, read_formated_data


# https://huggingface.co/blog/how-to-train-sentence-transformers


default_config = Box(
    dict(
        model_name=(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        ),
        dataset=dict(format="mnr", test_size=0.1, random_state=2023),
        train_loader=dict(
            shuffle=True, pin_memory=True, batch_size=128, num_workers=16
        ),
        val_loader=dict(
            shuffle=False, pin_memory=True, batch_size=128, num_workers=16
        ),
        optimizer=dict(name="AdamW", params=dict(lr=3e-5)),
        trainer=dict(max_epochs=1, use_amp=True, evaluation_steps=3200),
        output_path="./output_v9",
        seed=2023,
    )
)


class TCRelationDataset(Dataset):
    def __init__(self, dataset, text_dicts, format, random_switch=True):
        self.dataset = dataset
        self.text_dicts = text_dicts
        self.format = format
        self.random_switch = random_switch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.format == "mnr":
            tid, column_val1 = self.dataset.iloc[idx]
            x0 = self.text_dicts["topic"][tid]["title"]
            x1 = self.text_dicts["content"][column_val1]["title"]
            return InputExample(texts=[x0, x1])
        elif self.format == "triplet":
            tid, column_val1, column_val2 = self.dataset.iloc[idx]
            x0 = self.text_dicts["topic"][tid]["title"]
            x1 = self.text_dicts["content"][column_val1]["title"]
            x2 = self.text_dicts["content"][column_val2]["title"]
            return InputExample(texts=[x0, x1, x2])
        else:
            tid, column_val1, column_val2 = self.dataset.iloc[idx]
            x0 = self.text_dicts["topic"][tid]["title"]
            x1 = self.text_dicts["content"][column_val1]["title"]
            if self.random_switch and np.random.rand() > 0.5:
                x0, x1 = x1, x0
            y = column_val2
            return InputExample(texts=[x0, x1], label=y)


def create_datasets(config, input_path):
    topics, content, _ = read_original_data(input_path)
    train_topics, test_topics = train_test_split(
        topics, test_size=config.test_size, random_state=config.random_state
    )
    print("topics size: ", len(train_topics), "/", len(test_topics))

    train_set, test_data = read_formated_data(
        input_path,
        format=config.format,
        train_topics=train_topics,
        test_topics=test_topics,
    )
    print("train size: ", len(train_set))

    train_dset = TCRelationDataset(
        train_set,
        text_dicts=dict(topic=topics.loc, content=content.loc),
        format=config.format,
        random_switch=config.random_switch,
    )
    return train_dset, test_data


def train(config, input_path):
    config = default_config + config
    train_set, test_data = create_datasets(
        config.dataset,
        input_path=input_path,
    )
    train_loader = DataLoader(train_set, **config.train_loader)
    # test_loader = DataLoader(test_set, **config.val_loader)

    # 10% of train data
    warmup_steps = int(len(train_loader) * config.trainer.max_epochs * 0.1)
    model = SentenceTransformer(config.model_name)
    evaluator_params = dict(
        name=config.dataset.format,
        batch_size=config.val_loader.batch_size,
        show_progress_bar=True,
        write_csv=True,
    )
    if config.dataset.format == "mnr":
        train_loss = losses.MultipleNegativesRankingLoss(model)
        evaluator = evaluation.RerankingEvaluator(test_data, **evaluator_params)
    elif config.dataset.format == "triplet":
        train_loss = losses.TripletLoss(model)
        evaluator = evaluation.TripletEvaluator(*test_data, **evaluator_params)
    else:
        train_loss = losses.ContrastiveLoss(model)
        evaluator = evaluation.BinaryClassificationEvaluator(
            *test_data, **evaluator_params
        )

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=config.trainer.max_epochs,
        use_amp=config.trainer.use_amp,
        warmup_steps=warmup_steps,
        output_path=config.output_path + "/saved",
        checkpoint_path=config.output_path + "/checkpoint",
        evaluator=evaluator,
        evaluation_steps=config.trainer.evaluation_steps,
        optimizer_params=config.optimizer.params,
    )
    return model
