import math
from multiprocessing import cpu_count
from datasets import load_from_disk
from transformers import (
    BigBirdTokenizerFast,
    EncoderDecoderModel,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from torch.optim import AdamW


SEED = 42
TOKENIZER_DIR = './tokenizer-bigbird-small-indonesian'
PRETRAINED_DIR = './checkpoint-model-bigbird-small-indonesian/checkpoint-6450'
MODEL_DIR = './model-bigbird-small-indonesian-summarization'
TEMP_SEQ2SEQ_DIR = './model_ed'
DS_TRAIN_DIR = './dataset/finetune_train'
CHECKPOINT_DIR = f'./checkpoint-{MODEL_DIR}'
TENSORBOARD_DIR = './tensorboard-summarization'

MAX_LENGTH = 4096
EPOCH = 1
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 16
LEARNING_RATE = 1e-4


def init_model():
    tokenizer = BigBirdTokenizerFast.from_pretrained(TOKENIZER_DIR)
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(PRETRAINED_DIR, PRETRAINED_DIR)
    model.save_pretrained(TEMP_SEQ2SEQ_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(TEMP_SEQ2SEQ_DIR)

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # May need adjustment
    model.config.max_length = 512
    model.config.min_length = 32
    model.config.no_repeat_ngram_size = 3
    # model.config.early_stopping = True  # enable in future
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    return tokenizer, model

def train(tokenizer, model, ds_train):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    optim = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    training_args = Seq2SeqTrainingArguments(
        # checkpoint
        output_dir=CHECKPOINT_DIR,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2000,
        # log
        report_to='tensorboard',
        logging_dir=TENSORBOARD_DIR,
        logging_strategy='steps',
        logging_first_step=True,
        logging_steps=25,
        # train param
        num_train_epochs=EPOCH,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_accumulation_steps=GRADIENT_ACCUMULATION,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_ratio=0.06,
        bf16=True,  # increase train speed
        dataloader_num_workers=cpu_count(),
        # misc.
        seed=SEED,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        data_collator=data_collator,
        optimizers=(optim, None)
    )

    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(MODEL_DIR)

tokenizer, model = init_model()
ds_train = load_from_disk(DS_TRAIN_DIR)
train(tokenizer, model, ds_train)
