from box import Box
from train import train


config = dict()

default_config = Box(
    dict(
        model_name=(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        ),
        metrics=[],
        dataset=dict(triplet=False, test_size=0.3),
        train_loader=dict(shuffle=True, batch_size=16, num_workers=8),
        val_loader=dict(shuffle=False, batch_size=32, num_workers=8),
        loss=dict(name="CosineEmbeddingLoss"),
        trainer=dict(
            accelerator="gpu", devices=[0, 1, 2, 3, 4, 5, 6, 7], max_epochs=1
        ),
        optimizers=[dict(name="AdamW", lr=0.001, weight_decay=0)],
        schedulers=[dict(name="ExponentialLR", gamma=0.99)],
        seed=2023,
        logging=dict(save_dir="logging"),
    )
)
config = default_config + Box(config)
input_path = "./input"

train(config, input_path=input_path)
