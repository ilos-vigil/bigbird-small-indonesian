from transformers import (
    BigBirdConfig,
    BigBirdTokenizerFast,
    BigBirdForMaskedLM,
    pipeline
)
from pprint import pprint

MODEL_DIR = './model-bigbird-small-indonesian/checkpoint-1000'
TOKENIZER_DIR = './tokenizer-bigbird-small-indonesian'

tokenizer = BigBirdTokenizerFast.from_pretrained(TOKENIZER_DIR)
model = BigBirdForMaskedLM.from_pretrained(MODEL_DIR)
fm = pipeline(task='fill-mask', model=model, tokenizer=tokenizer, device='cpu')

print(f'{model.num_parameters()}=')
print(f'{model.num_parameters(exclude_embeddings=True)}=')

# fill-mask pipeline only support single [MASK] token
pprint(fm('Kucing itu sedang bermain dengan [MASK].'))
pprint(fm('Saya [MASK] makan nasi goreng.'))
pprint(fm('Negara [MASK] merdeka pada tahun 1945.'))
pprint(fm('Dia sedang bekerja [MASK] supermarket'))
pprint(fm('Buah tomat [MASK] untuk kesehatan.'))
pprint(fm('[MASK] sebagai hewan peliharaan mulai popular sejak tahun 2010.'))
