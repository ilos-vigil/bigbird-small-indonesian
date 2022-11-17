from datasets import load_dataset, concatenate_datasets
from multiprocessing import cpu_count
from transformers import BigBirdTokenizerFast

SEED = 42
CACHE_DIR = './dataset'
TOKENIZER_DIR = './tokenizer-bigbird-small-indonesian'
DS_TRAIN_DIR = './dataset/lm_train'
DS_EVAL_DIR = './dataset/lm_test'

MAX_LENGTH = 4096
TEST_SIZE = 0.05


def preprocess_function(examples, column, tokenizer):
    return tokenizer(examples[column], truncation=True)

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // MAX_LENGTH) * MAX_LENGTH
    result = {
        k: [t[i : i + MAX_LENGTH] for i in range(0, total_length, MAX_LENGTH)]
        for k, t in concatenated_examples.items()

    }
    result["labels"] = result["input_ids"].copy()

    return result


tokenizer = BigBirdTokenizerFast.from_pretrained(TOKENIZER_DIR)
ds_oscar = load_dataset("oscar", "unshuffled_deduplicated_id", cache_dir=CACHE_DIR)
ds_wiki = load_dataset("wikipedia", language="id", date="20221020", beam_runner="DirectRunner", cache_dir=CACHE_DIR)
ds_news = load_dataset("id_newspapers_2018", cache_dir=CACHE_DIR)

tokenized_oscar = ds_oscar.map(
    preprocess_function,
    batched=True,
    load_from_cache_file=True,
    num_proc=cpu_count() - 1,
    remove_columns=ds_oscar["train"].column_names,
    fn_kwargs={'column': 'text', 'tokenizer': tokenizer}
)
tokenized_wiki = ds_wiki.map(
    preprocess_function,
    batched=True,
    load_from_cache_file=True,
    num_proc=cpu_count() - 1,
    remove_columns=ds_wiki["train"].column_names,
    fn_kwargs={'column': 'text', 'tokenizer': tokenizer}
)
tokenized_news = ds_news.map(
    preprocess_function,
    batched=True,
    load_from_cache_file=True,
    num_proc=cpu_count() - 1,
    remove_columns=ds_news["train"].column_names,
    fn_kwargs={'column': 'content', 'tokenizer': tokenizer}
)

lm_oscar = tokenized_oscar.map(
    group_texts,
    batched=True,
    load_from_cache_file=True,
    num_proc=cpu_count() - 1,
)
lm_wiki = tokenized_wiki.map(
    group_texts,
    batched=True,
    load_from_cache_file=True,
    num_proc=cpu_count() - 1,
)
lm_news = tokenized_news.map(
    group_texts,
    batched=True,
    load_from_cache_file=True,
    num_proc=cpu_count() - 1,
)

lm_oscar = lm_oscar['train'].train_test_split(TEST_SIZE, seed=SEED)
lm_wiki = lm_wiki['train'].train_test_split(TEST_SIZE, seed=SEED)
lm_news = lm_news['train'].train_test_split(TEST_SIZE, seed=SEED)

lm_train = concatenate_datasets([
    lm_oscar['train'],
    lm_wiki['train'],
    lm_news['train'],
])
lm_test = concatenate_datasets([
    lm_oscar['test'],
    lm_wiki['test'],
    lm_news['test'],
])

lm_train.save_to_disk(DS_TRAIN_DIR)
lm_test.save_to_disk(DS_EVAL_DIR)

tokenized_oscar.cleanup_cache_files()
tokenized_wiki.cleanup_cache_files()
tokenized_news.cleanup_cache_files()
lm_oscar.cleanup_cache_files()
lm_wiki.cleanup_cache_files()
lm_news.cleanup_cache_files()
