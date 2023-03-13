from box import Box

from stage_1.train_huggingface import train as train_huggingface
from stage_1.train_transformer import train as train_transformer

import os
print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])

huggingface_config = Box(
    dict(
        model_name=(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        ),
        dataset=dict(triplet=False, test_size=0.2),
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


transformer_config = Box(
    dict(
        model_name=(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        ),
        dataset=dict(triplet=False, test_size=0.1),
        train_loader=dict(
            shuffle=True, pin_memory=True, batch_size=64, num_workers=16
        ),
        val_loader=dict(
            shuffle=False, pin_memory=True, batch_size=96, num_workers=16
        ),
        loss=dict(name="ContrastiveLoss", params=dict()),
        # loss=dict(name="TripletLoss", params=dict()), #
        trainer=dict(max_epochs=1, use_amp=True),
        seed=2023,
    )
)

config = dict()
input_path = "./input"

train_transformer(transformer_config + Box(config), input_path=input_path)
# train_huggingface(huggingface_config + Box(config), input_path=input_path)