import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from lightning import (
    LightningModule,
    Trainer,
    seed_everything,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies.ddp import DDPStrategy
from transformers import AutoTokenizer, AutoModel
from box import Box
import wandb
from sklearn.model_selection import train_test_split

from prepare.read import read_original_data, read_formated_data


# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html
# https://huggingface.co/docs/transformers/training
# https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2


default_config = Box(
    dict(
        model_name=(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        ),
        dataset=dict(triplet=False, test_size=0.2, random_state=2023),
        train_loader=dict(
            shuffle=True, pin_memory=True, batch_size=4, num_workers=16
        ),
        val_loader=dict(
            shuffle=False, pin_memory=True, batch_size=4, num_workers=16
        ),
        loss=dict(
            name="cosine_embedding_loss", params=dict(margin=1)
        ),  # CosineEmbeddingLoss
        trainer=dict(
            accelerator="gpu",
            devices=[0, 1, 2, 3, 4],
            max_epochs=1,
            # precision=16,
        ),
        optimizers=[dict(name="AdamW", lr=0.00001, weight_decay=0)],
        schedulers=[dict(name="ExponentialLR", gamma=0.99)],
        seed=2023,
        logging=dict(save_dir="./output/logging"),
    )
)


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
    def transform_x(x):
        out = tokenizer(
            x, padding="max_length", truncation=True, return_tensors="pt"
        )
        return {k: torch.squeeze(out[k]) for k in out}

    topics, content, _ = read_original_data(input_path)
    train_topics, test_topics = train_test_split(
        topics, test_size=config.test_size, random_state=config.random_state
    )

    all_set = read_formated_data(input_path, triplet=config.triplet)
    train_set = all_set[all_set["topic_id"].isin(train_topics.index)]
    test_set = all_set[all_set["topic_id"].isin(test_topics.index)]

    text_dicts = dict(topic=topics.loc, content=content.loc)
    return [
        TCRelationDataset(
            s,
            text_dicts=text_dicts,
            transform_x=transform_x,
            triplet=config.triplet,
        )
        for s in [train_set, test_set]
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
    def loss_function(*args):
        return getattr(nn.functional, config.name)(
            *args, **config.get("params", {})
        )

    return loss_function


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
        loss = self.forward_step(batch)
        self.log_dict({f"train/loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward_step(batch)
        self.log_dict({f"val/loss": loss}, sync_dist=True)
        return loss

    def forward_step(self, batch):
        x0, x1, x2 = batch
        loss_val = (
            self._loss(
                mean_pooling(self.model(**x0), x0),
                mean_pooling(self.model(**x1), x1),
                mean_pooling(self.model(**x2), x2),
            )
            if self._tripletautwa
            else self._loss(
                mean_pooling(self.model(**x0), x0),
                mean_pooling(self.model(**x1), x1),
                x2,
            )
        )
        return loss_val

    def configure_optimizers(self):
        optimizers = self._optimizers
        schedulers = self._schedulers
        return optimizers, schedulers


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, model_input):
    attention_mask = model_input["attention_mask"]
    token_embeddings = model_output.last_hidden_state  # model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def train(config, input_path):
    config = default_config + config
    seed_everything(config.seed, True)
    # wandb auto set project name based on git (not documented)
    # see: https://github.com/wandb/wandb/blob/cce611e2e518951064833b80aee975fa139a85ee/wandb/cli/cli.py#L872
    wandb_logger = WandbLogger(
        save_dir=config.logging.save_dir,
        group=f"stage1",
        project="lecr",
        config=config,
    )
    # model_config = AutoConfig.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    backbone = AutoModel.from_pretrained(
        config.model_name, add_pooling_layer=False
    )

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
