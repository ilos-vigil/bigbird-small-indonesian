import math
from multiprocessing import cpu_count
from datasets import load_from_disk
from transformers import (
    BigBirdConfig,
    BigBirdTokenizerFast,
    BigBirdForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    get_cosine_schedule_with_warmup
)
from torch.optim import AdamW

SEED = 42
MODEL_DIR = 'model-bigbird-small-indonesian'
TOKENIZER_DIR = './tokenizer-bigbird-small-indonesian'
DS_TRAIN_DIR = './dataset/lm_train'
DS_EVAL_DIR = './dataset/lm_test'
CHECKPOINT_DIR = f'./checkpoint-{MODEL_DIR}'
TENSORBOARD_DIR = './tensorboard'

MAX_LENGTH = 4096
EPOCH = 8
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 64
LEARNING_RATE = 1e-4


def init_model():
    config = BigBirdConfig(
        vocab_size = 30_000,
        hidden_size = 512,
        num_hidden_layers = 4,
        num_attention_heads = 8,
        intermediate_size = 2048,  # BERT/RoBERTa use 4x hidden_size
        max_position_embeddings = MAX_LENGTH,
        is_encoder_decoder=False,
        attention_type='block_sparse'
    )
    tokenizer = BigBirdTokenizerFast.from_pretrained(TOKENIZER_DIR)
    model = BigBirdForMaskedLM(config)
    
    return config, tokenizer, model


def train(tokenizer, model, ds_train, ds_eval):
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
    optim = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    # lr_scheduler = get_cosine_schedule_with_warmup(optim)
    training_args = TrainingArguments(
        # checkpoint
        output_dir=CHECKPOINT_DIR,
        save_strategy='steps',
        save_steps=50,
        save_total_limit=2000,
        # log
        report_to='tensorboard',
        logging_dir=TENSORBOARD_DIR,
        logging_strategy='steps',
        logging_first_step=True,
        logging_steps=10,
        # train
        num_train_epochs=EPOCH,
        weight_decay=0.01,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.06,
        bf16=True,
        dataloader_num_workers=cpu_count(),
        # misc.
        evaluation_strategy='epoch',
        seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=data_collator,
        optimizers=(optim, None)
    )
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(MODEL_DIR)


if __name__ == '__main__':
    config, tokenizer, model = init_model()
    ds_train = load_from_disk(DS_TRAIN_DIR)
    ds_eval = load_from_disk(DS_EVAL_DIR)
    train(tokenizer, model, ds_train, ds_eval)
