import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from lightning import (
    LightningModule,
    Trainer,
    seed_everything,
)
from sklearn.model_selection import train_test_split
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies.ddp import DDPStrategy
from transformers import AutoTokenizer, AutoModel
import wandb


from prepare.read import read_original_data, read_generated_data

# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html
# https://huggingface.co/docs/transformers/training
# https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2


class TCRelationDataset(Dataset):
    def __init__(self, dataset, text_dicts, transform_x=None, triplet=False):
        self.dataset = dataset
        self.text_dicts = text_dicts
        self.transform_x = transform_x
        self.triplet = triplet
        self.transform_y = transform_x
        if not triplet:
            self.transform_y = lambda y: y if y == 1 else -1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tid, column_val1, column_val2 = self.dataset.iloc[idx]
        t = self.text_dicts["topic"][tid]["title"]
        x1 = self.text_dicts["content"][column_val1]["title"]

        if not self.triplet:
            y = column_val2
            return (
                self.transform_x(t),
                self.transform_x(x1),
                self.transform_y(y),
            )
        x2 = self.text_dicts["content"][column_val2]["title"]
        return self.transform_x(t), self.transform_x(x1), self.transform_x(x2)


def create_datasets(config, input_path, tokenizer):
    transform_x = lambda x: tokenizer(
        x, padding="max_length", truncation=True, return_tensors="pt"
    )
    topics, content, _ = read_original_data(input_path)
    all_set = read_generated_data(input_path, triplet=config.triplet)
    text_dicts = dict(topic=topics.loc, content=content.loc)
    sets = train_test_split(
        all_set, test_size=config.test_size, stratify=all_set["topic_id"]
    )
    return [
        TCRelationDataset(
            s,
            text_dicts=text_dicts,
            transform_x=transform_x,
            triplet=config.triplet,
        )
        for s in sets
    ]


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


def create_loss(config):
    return getattr(nn, config.name)(**config.get("params", {}))


class StepModule(LightningModule):
    def __init__(self, model, optimizers, schedulers, loss, triplet):
        super().__init__()
        # self.hold_graph = self.params['retain_first_backpass'] or False
        self.model = model
        self._optimizers = optimizers
        self._schedulers = schedulers
        self._loss = loss
        self._triplet = triplet

    def training_step(self, batch, batch_idx):
        loss_vals = self.forward_step(batch)
        self.log_dict({f"train/{k}": v for k, v in loss_vals.items()})
        loss = loss_vals.get("loss") or next(iter(loss_vals.values()))
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        loss_vals = self.forward_step(batch)
        self.log_dict(
            {f"val/{k}": v for k, v in loss_vals.items()}, sync_dist=True
        )
        loss = loss_vals.get("loss") or next(iter(loss_vals.values()))
        return dict(loss=loss)

    def forward_step(self, batch):
        loss_vals = []
        x0, x1, x2 = batch
        out0 = mean_pooling(self.model(x0), x0)
        out1 = mean_pooling(self.model(x1), x1)
        loss_val = (
            self._loss(out0, out1, mean_pooling(self.model(x2), x2))
            if self._triplet
            else self._loss(out0, out1, x2)
        )
        loss_vals.append(loss_val)
        return loss_vals

    def configure_optimizers(self):
        optimizers = self._optimizers
        schedulers = self._schedulers
        return optimizers, schedulers


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, model_input):
    attention_mask = model_input["attention_mask"]
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def train(config, input_path):
    seed_everything(config.seed, True)
    # wandb auto set project name based on git (not documented)
    # see: https://github.com/wandb/wandb/blob/cce611e2e518951064833b80aee975fa139a85ee/wandb/cli/cli.py#L872
    wandb_logger = WandbLogger(
        save_dir=config.logging.save_dir,
        group=f"{config.model_name}",
        config=config,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    backbone = AutoModel.from_pretrained(config.model_name)

    train_set, test_set = create_datasets(
        config.dataset, input_path=input_path, tokenizer=tokenizer
    )
    train_loader = DataLoader(train_set, **config.train_loader)
    test_loader = DataLoader(test_set, **config.val_loader)

    optimizers = create_optimizers(backbone, config.optimizers)
    schedulers = create_schedulers(optimizers, config.schedulers)
    loss = create_loss(config.loss)
    step_module = StepModule(
        backbone, optimizers, schedulers, loss, triplet=config.dataset.triplet
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
    runner.fit(step_module, train_loader, test_loader)
    wandb.finish()
    return backbone
