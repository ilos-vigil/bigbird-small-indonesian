from datasets import load_dataset

CACHE_DIR = './dataset'

# Pretrain Language Model
load_dataset('oscar', 'unshuffled_deduplicated_id', cache_dir=CACHE_DIR)
load_dataset('id_newspapers_2018', cache_dir=CACHE_DIR)
load_dataset('wikipedia', language='id', date='20221020', beam_runner='DirectRunner', cache_dir=CACHE_DIR)

# Finetune - Summarization
load_dataset('wiki_lingua', 'indonesian', cache_dir=CACHE_DIR)
load_dataset('csebuetnlp/xlsum', 'indonesian', cache_dir=CACHE_DIR)
load_dataset('id_liputan6', 'canonical', data_dir=f'{CACHE_DIR}/sum_liputan6/data', ignore_verifications=True, cache_dir=CACHE_DIR)
