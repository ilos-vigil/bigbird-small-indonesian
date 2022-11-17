from datasets import load_dataset
from transformers import BigBirdTokenizerFast

CACHE_DIR = './dataset'
TOKENIZER_DIR = 'tokenizer-bigbird-small-indonesian'

ds_oscar = load_dataset("oscar", "unshuffled_deduplicated_id", cache_dir=CACHE_DIR)
ds_wiki = load_dataset("wikipedia", language="id", date="20221020", beam_runner="DirectRunner", cache_dir=CACHE_DIR)
ds_news = load_dataset("id_newspapers_2018", cache_dir=CACHE_DIR)

def batch_ds(batch_size:int = 1024):
    for i in range(0, len(ds_oscar), batch_size):
        yield ds_oscar['train'][i:i+batch_size]['text']
    for i in range(0, len(ds_wiki), batch_size):
        yield ds_wiki['train'][i:i+batch_size]['text']
    for i in range(0, len(ds_news), batch_size):
        yield ds_news['train'][i:i+batch_size]['content']

tokenizer = BigBirdTokenizerFast.from_pretrained('google/bigbird-roberta-base')
tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_ds(), vocab_size=30_000)
tokenizer.name_or_path = TOKENIZER_DIR

tokenizer.save_pretrained(TOKENIZER_DIR)
