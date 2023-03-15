import os
import ast
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from lightning import (
    LightningModule,
    Trainer,
    seed_everything,
)
from sentence_transformers import SentenceTransformer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies.ddp import DDPStrategy
import pandas as pd
import numpy as np
from box import Box
import wandb
from sklearn.model_selection import train_test_split

from lecr.prepare.read import read_original_data, get_formated_neighbors
from lecr.stage_1.get_possible_content import get_possible_content
from lecr.eval.f2 import compute_f2scores_for


default_config = Box(
    dict(
        model_name="/root/lecr/output_v15/saved",
        nn_name=None,
        dataset=dict(test_size=0.1, top_k=100, random_state=2023),
        train_loader=dict(
            shuffle=True, pin_memory=True, batch_size=4096, num_workers=16
        ),
        val_loader=dict(
            shuffle=False, pin_memory=True, batch_size=4096, num_workers=16
        ),
        trainer=dict(
            accelerator="gpu",
            devices=[0, 1],
            max_epochs=1,
            # precision=16,
        ),
        optimizers=[dict(name="AdamW", lr=0.001, weight_decay=0)],
        schedulers=[dict(name="ExponentialLR", gamma=0.99)],
        seed=2023,
        logging=dict(save_dir="./out/output_nn_v0"),
    )
)


class TCRelationDataset(Dataset):
    def __init__(self, dataset, dicts):
        self.dataset = dataset
        self.dicts = dicts

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tid, cid, correlated = self.dataset.iloc[idx]
        trow = self.dicts["topic"][tid]
        crow = self.dicts["content"][cid]
        return (
            trow[2:].to_numpy().astype("float32"),
            trow["title"],
            crow[2:].to_numpy().astype("float32"),
            crow["title"],
            float(correlated),
        )


def get_possible_content_cache(
    input_path, model_name, top_k, topics, content, correlations
):
    generated_data_path = os.path.join(input_path, "learning-equality", "v2")
    formated_model_name = model_name.replace(".", "").replace("/", "_")
    fpath = os.path.join(
        generated_data_path, f"{formated_model_name}-top_{top_k}.csv"
    )
    if os.path.isfile(fpath):
        possible_set = pd.read_csv(
            fpath,
            converters={"content_ids": ast.literal_eval},
        )
        return possible_set

    possible_set = get_possible_content(
        topics, content, correlations, model_name=model_name, top_k=top_k
    )
    possible_set.to_csv(fpath)
    return possible_set


def create_datasets(config, input_path, model_name):
    topics, content, correlations = read_original_data(
        input_path, detailed=True
    )
    train_topics, test_topics = train_test_split(
        topics, test_size=config.test_size, random_state=config.random_state
    )
    test_correlations = correlations[
        correlations.index.isin(test_topics.index)
    ]

    possible_set = get_possible_content_cache(
        input_path,
        model_name=model_name,
        top_k=config.top_k,
        topics=topics,
        content=content,
        correlations=correlations,
    )
    all_set = get_formated_neighbors(input_path, possible_set)
    train_set = all_set[all_set["topic_id"].isin(train_topics.index)]
    test_set = all_set[all_set["topic_id"].isin(test_topics.index)]

    dicts = dict(topic=topics.loc, content=content.loc)
    return [
        TCRelationDataset(s, dicts=dicts) for s in [train_set, test_set]
    ], test_correlations


def create_optimizers(model, confs):
    optimizers = []
    for c in confs:
        m = getattr(model, c.model) if "model" in c else model
        optimizer_func = getattr(torch.optim, c.pop("name", "Adam"))
        o = optimizer_func(m.parameters(), **c)
        optimizers.append(o)
    return optimizers


def create_schedulers(optimizers, confs):
    schedulers = []
    for o, c in zip(optimizers, confs):
        scheduler_func = getattr(
            torch.optim.lr_scheduler, c.pop("name", "ExponentialLR")
        )
        schedulers.append(scheduler_func(o, **c))
    return schedulers


class StepModule(LightningModule):
    def __init__(self, model, optimizers, schedulers, loss, text_encode):
        super().__init__()
        # self.hold_graph = self.params['retain_first_backpass'] or False
        self.model = model
        self._optimizers = optimizers
        self._schedulers = schedulers
        self._loss = loss
        self._text_encode = text_encode

    def training_step(self, batch, batch_idx):
        loss = self.forward_step(batch)
        self.log_dict({f"train/loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward_step(batch)
        self.log_dict({f"val/loss": loss}, sync_dist=True)
        return loss

    def forward_step(self, batch):
        t, t_titles, c, c_titles, y = batch
        y = y.reshape(-1, 1)
        t_embeddings = self._text_encode(t_titles)
        c_embeddings = self._text_encode(c_titles)
        representations = torch.hstack([t, t_embeddings, c, c_embeddings])
        y_hat = self.model(representations)
        loss_val = self._loss(y_hat, y)
        return loss_val

    def configure_optimizers(self):
        optimizers = self._optimizers
        schedulers = self._schedulers
        return optimizers, schedulers


def create_model(nn_name, input_size=1771):
    model = nn.Sequential(
        nn.Linear(input_size, 2048),
        nn.ReLU(),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Linear(512, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    if nn_name is not None and os.path.isfile(nn_name):
        model.load_state_dict(torch.load(nn_name))
    return model


def train(config, input_path):
    config = default_config + config
    seed_everything(config.seed, True)
    # wandb auto set project name based on git (not documented)
    # see: https://github.com/wandb/wandb/blob/cce611e2e518951064833b80aee975fa139a85ee/wandb/cli/cli.py#L872
    wandb_logger = WandbLogger(
        save_dir=config.logging.save_dir,
        group=f"stage2",
        project="lecr",
        config=config,
    )

    sets, test_correlations = create_datasets(
        config.dataset, input_path=input_path, model_name=config.model_name
    )
    train_set, test_set = sets
    train_loader = DataLoader(train_set, **config.train_loader)
    test_loader = DataLoader(test_set, **config.val_loader)

    encoder = SentenceTransformer(config.model_name)

    def text_encode(*args, **kwargs):
        return encoder.encode(*args, convert_to_tensor=True, **kwargs)

    criterion = nn.BCEWithLogitsLoss()
    backbone = create_model(config.nn_name)
    optimizers = create_optimizers(backbone, config.optimizers)
    schedulers = create_schedulers(optimizers, config.schedulers)

    step_module = StepModule(
        backbone, optimizers, schedulers, criterion, text_encode
    )

    runner = Trainer(
        logger=wandb_logger,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=2,
                dirpath=os.path.join(config.logging.save_dir, "checkpoints"),
                monitor="val/loss",
                save_last=True,
            ),
        ],
        strategy=DDPStrategy(find_unused_parameters=False),
        # detect_anomaly=True,
        **config.trainer,
    )
    runner.fit(step_module, train_loader)
    torch.save(
        backbone.state_dict(), os.path.join(config.logging.save_dir, "saved")
    )
    predictions = get_predictions(
        test_correlations,
        test_loader,
        text_encode,
        backbone,
        device=encoder.device,
    )
    scores = compute_f2scores_for(test_correlations, predictions)
    score = np.mean(scores)
    print("score", score)
    wandb.finish()
    return backbone


@torch.no_grad()
def get_predictions(
    target_correlations, val_loader, text_encode, model, device, threshold=0.5
):
    y_hat = []
    model.to(device)
    model.eval()
    for batch in val_loader:
        t, t_titles, c, c_titles, y = batch
        t_embeddings = text_encode(t_titles)
        c_embeddings = text_encode(c_titles)
        representations = torch.hstack(
            [t.to(device), t_embeddings, c.to(device), c_embeddings]
        )
        y_hat.append(torch.sigmoid(model(representations)))
    y_hat = torch.vstack(y_hat).cpu().detach().numpy().squeeze()
    val_dataset = val_loader.dataset.dataset

    predictions = pd.DataFrame(
        {
            "topic_id": val_dataset["topic_id"],
            "content_id": val_dataset["content_id"],
            "correlated": y_hat > threshold,
        }
    )
    predictions = (
        predictions[predictions["correlated"]]
        .groupby(["topic_id"])
        .agg({"content_id": lambda x: x.tolist()})
        .rename({"content_id": "content_ids"}, axis=1)
    )
    content_ids = [
        predictions.loc[id]["content_ids"] if id in predictions.index else []
        for id, row in target_correlations.iterrows()
    ]
    correlated_predictions = pd.DataFrame(
        {
            "topic_id": target_correlations.index.tolist(),
            "content_ids": content_ids,
        }
    ).set_index("topic_id")

    return correlated_predictions


train(Box({}), input_path="./input")
