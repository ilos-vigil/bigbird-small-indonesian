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

# Finetune - NLI
load_dataset('indonli', cache_dir=CACHE_DIR)
# Even though only id_* split is chosen, datasets will download whole whole dataset, see https://github.com/huggingface/datasets/issues/5243
load_dataset('MoritzLaurer/multilingual-NLI-26lang-2mil7', split='id_anli+id_fever+id_ling+id_mnli+id_wanli', cache_dir=CACHE_DIR)
