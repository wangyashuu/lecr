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
        train_loader=dict(
            shuffle=True, pin_memory=True, batch_size=2, num_workers=4
        ),
        val_loader=dict(
            shuffle=False, pin_memory=True, batch_size=4, num_workers=4
        ),
        loss=dict(name="cosine_embedding_loss"),  # CosineEmbeddingLoss
        trainer=dict(
            accelerator="gpu",
            devices=[3, 4, 5, 6, 7],
            max_epochs=16,
            precision=16,
            # amp_backend="apex",
        ),
        optimizers=[dict(name="AdamW", lr=0.00001, weight_decay=0)],
        schedulers=[dict(name="ExponentialLR", gamma=0.99)],
        seed=2023,
        logging=dict(save_dir="logging"),
    )
)
config = default_config + Box(config)
input_path = "./input"

train(config, input_path=input_path)
