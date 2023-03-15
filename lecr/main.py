from box import Box

from stage_0.train_huggingface import train as train_huggingface
from stage_0.train_transformer import train as train_transformer

import os

print(
    "CUDA_VISIBLE_DEVICES:",
    os.environ["CUDA_VISIBLE_DEVICES"]
    if "CUDA_VISIBLE_DEVICES" in os.environ
    else "",
)


# config = Box(dict())
input_path = "./input"

config = Box(
    dict(
        # model_name="/root/lecr/output_v15/saved",
        dataset=dict(
            triplet=False, random_switch=True, test_size=0.1, random_state=2023
        ),
        train_loader=dict(
            shuffle=True, pin_memory=True, batch_size=32, num_workers=16
        ),
        val_loader=dict(
            shuffle=False, pin_memory=True, batch_size=64, num_workers=16
        ),
        optimizer=dict(name="AdamW", params=dict(lr=3e-5)),
        trainer=dict(max_epochs=64, use_amp=True, evaluation_steps=3200),
        output_path="./output_v18",
        seed=2023,
    )
)

train_transformer(config, input_path=input_path)
# train_huggingface(huggingface_config + Box(config), input_path=input_path)
