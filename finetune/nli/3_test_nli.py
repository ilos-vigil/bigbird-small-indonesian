import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from multiprocessing import cpu_count


BATCH_SIZE = 32
CACHE_DIR = './dataset'
MODELS = [
    'ilos-vigil/bigbird-small-indonesian-nli',
    'joeddav/xlm-roberta-large-xnli',
    'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
]

ds_indonli = load_dataset('indonli', cache_dir=CACHE_DIR)
accuracy = evaluate.load('accuracy')


def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples['premise'], examples['hypothesis'],
        truncation=True, return_token_type_ids=True
    )

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def test_nli(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir='/tmp/test_nli',
        report_to='none',
        per_device_eval_batch_size=BATCH_SIZE,
        bf16=True,
        dataloader_num_workers=cpu_count()
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    tokenized_indonli = ds_indonli.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
        num_proc=1,
        remove_columns=['premise', 'hypothesis'],  # all except 'label'
        fn_kwargs={'tokenizer': tokenizer}
    )


    test_lay_result = trainer.predict(tokenized_indonli['test_lay'])
    test_exp_result = trainer.predict(tokenized_indonli['test_expert'])
    tokenized_indonli.cleanup_cache_files()

    print('Model name:', model_name)
    print(test_lay_result.metrics)
    print(test_exp_result.metrics)
    print('='*79)

for m in MODELS:
    test_nli(m)
