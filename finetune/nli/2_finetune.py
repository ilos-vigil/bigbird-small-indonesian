import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import (
    BigBirdTokenizerFast,
    BigBirdForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from multiprocessing import cpu_count

SEED = 42
CACHE_DIR = './dataset'

PRETRAINED_MODEL = 'ilos-vigil/bigbird-small-indonesian'
NLI_MODEL = 'bigbird-small-indonesian-nli'
DS_TRAIN_DIR = './dataset/nli_train'
DS_VAL_DIR = './dataset/nli_val'
DS_TEST_LAY_DIR = './dataset/nli_test_lay'
DS_TEST_EXP_DIR = './dataset/nli_test_expert'
CHECKPOINT_DIR = f'./checkpoint-{NLI_MODEL}'
TENSORBOARD_DIR = './tensorboard-nli'

EPOCH = 16
BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 1e-5


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# Matching with the used dataset
id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
accuracy = evaluate.load('accuracy')

tokenizer = BigBirdTokenizerFast.from_pretrained(PRETRAINED_MODEL)
model = BigBirdForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL, num_labels=3, id2label=id2label, label2id=label2id
)
ds_train = load_from_disk(DS_TRAIN_DIR)
ds_val = load_from_disk(DS_VAL_DIR)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments(
    # checkpoint
    output_dir=CHECKPOINT_DIR,
    save_strategy='epoch',
    save_total_limit=EPOCH,
    # log
    report_to='tensorboard',
    logging_dir=TENSORBOARD_DIR,
    logging_strategy='steps',
    logging_first_step=True,
    logging_steps=50,
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
    eval_dataset=ds_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train(resume_from_checkpoint=True)
trainer.save_model(NLI_MODEL)
