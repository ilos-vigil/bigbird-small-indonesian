from datasets import load_dataset, concatenate_datasets
from multiprocessing import cpu_count
from transformers import BigBirdTokenizerFast


SEED = 42
CACHE_DIR = './dataset'
TOKENIZER_DIR = './tokenizer-bigbird-small-indonesian'
DS_TRAIN_DIR = './dataset/finetune_train'
DS_EVAL_DIR = './dataset/finetune_test'

MAX_LENGTH_INPUT = 4096
MAX_LENGTH_OUTPUT = 512
TEST_SIZE = 0.1


def preprocess_function_lingua(examples, tokenizer):
    section_name = []
    document = []
    summary = []
    for example in examples['article']:
        for text in example['section_name']:
            section_name.append(text)
        for text in example['document']:
            document.append(text)
        for text in example['summary']:
            summary.append(text)
    inputs = [sn + '\n\n' + d for sn, d in zip(section_name, document)]
    labels = tokenizer(text_target=summary, max_length=MAX_LENGTH_OUTPUT, truncation=True, padding='max_length')

    batch = tokenizer(inputs, max_length=MAX_LENGTH_INPUT, truncation=True, padding='max_length')
    batch["decoder_input_ids"] = labels['input_ids']
    batch["decoder_attention_mask"] = labels['attention_mask']
    batch['labels'] = labels['input_ids']
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
    ]

    return batch

def preprocess_function(examples, input_column, output_column, tokenizer):
    inputs = [doc for doc in examples[input_column]]
    labels = tokenizer(text_target=examples[output_column], max_length=MAX_LENGTH_OUTPUT, truncation=True, padding='max_length')

    batch = tokenizer(inputs, max_length=MAX_LENGTH_INPUT, truncation=True, padding='max_length')  # has input_ids, attention_mask
    batch["decoder_input_ids"] = labels['input_ids']
    batch["decoder_attention_mask"] = labels['attention_mask']
    batch['labels'] = labels['input_ids']
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
    ]

    return batch


tokenizer = BigBirdTokenizerFast.from_pretrained(TOKENIZER_DIR)
ds_lingua = load_dataset('wiki_lingua', 'indonesian', cache_dir=CACHE_DIR)
ds_xlsum = load_dataset('csebuetnlp/xlsum', 'indonesian', cache_dir=CACHE_DIR)
ds_liputan6 = load_dataset('id_liputan6', 'canonical', data_dir=f'{CACHE_DIR}/sum_liputan6/data', ignore_verifications=True, cache_dir=CACHE_DIR)

tokenized_lingua = ds_lingua.map(
    preprocess_function_lingua,
    batched=True,
    load_from_cache_file=True,
    # num_proc=cpu_count() - 1,
    remove_columns=ds_lingua["train"].column_names,
    fn_kwargs={'tokenizer': tokenizer}
)
tokenized_xlsum = ds_xlsum.map(
    preprocess_function,
    batched=True,
    load_from_cache_file=True,
    # num_proc=cpu_count() - 1,
    remove_columns=ds_xlsum["train"].column_names,
    fn_kwargs={'input_column': 'text', 'output_column': 'summary', 'tokenizer': tokenizer}
)
tokenized_liputan6 = ds_liputan6.map(
    preprocess_function,
    batched=True,
    load_from_cache_file=True,
    num_proc=cpu_count() - 1,
    remove_columns=ds_liputan6["train"].column_names,
    fn_kwargs={'input_column': 'clean_article', 'output_column': 'clean_summary', 'tokenizer': tokenizer}
)

tokenized_lingua = tokenized_lingua['train'].train_test_split(TEST_SIZE, seed=SEED)

finetune_train = concatenate_datasets([
    tokenized_lingua['train'],
    tokenized_xlsum['train'],
    tokenized_liputan6['train'],
    tokenized_xlsum['validation'],
    tokenized_liputan6['validation'],
])
finetune_test = concatenate_datasets([
    tokenized_lingua['test'],
    tokenized_xlsum['test'],
    tokenized_liputan6['test'],
])

finetune_train.save_to_disk(DS_TRAIN_DIR)
finetune_test.save_to_disk(DS_EVAL_DIR)

tokenized_lingua.cleanup_cache_files()
tokenized_xlsum.cleanup_cache_files()
tokenized_liputan6.cleanup_cache_files()
