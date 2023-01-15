from datasets import (
    load_dataset,
    concatenate_datasets
)
from multiprocessing import cpu_count
from transformers import BigBirdTokenizerFast


CACHE_DIR = './dataset'
MODEL_NAME = 'ilos-vigil/bigbird-small-indonesian'

DS_TRAIN_DIR = './dataset/nli_train'
DS_VAL_DIR = './dataset/nli_val'
DS_TEST_LAY_DIR = './dataset/nli_test_lay'
DS_TEST_EXP_DIR = './dataset/nli_test_expert'


def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples['premise'], examples['hypothesis'],
        truncation=True, return_token_type_ids=True
    )


tokenizer = BigBirdTokenizerFast.from_pretrained(MODEL_NAME)
ds_indonli = load_dataset('indonli', cache_dir=CACHE_DIR)
ds_multinli = load_dataset(
    'MoritzLaurer/multilingual-NLI-26lang-2mil7',
    split='id_anli+id_fever+id_ling+id_mnli+id_wanli',
    cache_dir=CACHE_DIR
)

tokenized_indonli = ds_indonli.map(
    preprocess_function,
    batched=True,
    load_from_cache_file=True,
    num_proc=1,
    # num_proc=cpu_count() - 1,
    remove_columns=['premise', 'hypothesis'],  # all except 'label'
    fn_kwargs={'tokenizer': tokenizer}
)
tokenized_multinli = ds_multinli.map(
    preprocess_function,
    batched=True,
    load_from_cache_file=True,
    num_proc=cpu_count() - 1,
    remove_columns=['premise_original', 'hypothesis_original', 'premise', 'hypothesis'],  # all except 'label'
    fn_kwargs={'tokenizer': tokenizer}
)
tokenized_multinli.features['label'] = tokenized_indonli['train'].features['label']

nli_train = concatenate_datasets([
    tokenized_indonli['train'],
    tokenized_multinli
])

nli_train.save_to_disk(DS_TRAIN_DIR)
tokenized_indonli['validation'].save_to_disk(DS_VAL_DIR)
tokenized_indonli['test_lay'].save_to_disk(DS_TEST_LAY_DIR)
tokenized_indonli['test_expert'].save_to_disk(DS_TEST_EXP_DIR)

tokenized_indonli.cleanup_cache_files()
tokenized_multinli.cleanup_cache_files()
nli_train.cleanup_cache_files()
