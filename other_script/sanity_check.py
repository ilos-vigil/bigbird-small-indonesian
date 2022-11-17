from transformers import (
    BigBirdConfig,
    BigBirdTokenizerFast,
    BigBirdForMaskedLM,
    pipeline
)
from pprint import pprint

MODEL_DIR = './checkpoint-model-bigbird-small-indonesian/checkpoint-7800'
TOKENIZER_DIR = './tokenizer-bigbird-small-indonesian'

tokenizer = BigBirdTokenizerFast.from_pretrained(TOKENIZER_DIR)
model = BigBirdForMaskedLM.from_pretrained(MODEL_DIR)
fm = pipeline(task='fill-mask', model=model, tokenizer=tokenizer, device='cpu')

print(f'{model.num_parameters()}=')
print(f'{model.num_parameters(exclude_embeddings=True)}=')

pprint(fm('Kucing itu sedang bermain dengan [MASK].'))
pprint(fm('Saya [MASK] makan nasi goreng.'))
pprint(fm('Negara [MASK] merdeka pada tahun 1945.'))
pprint(fm('Saya suka makan [MASK].'))
pprint(fm('Profesi ibu saya adalah [MASK].'))
pprint(fm('Profesi bapak saya adalah [MASK].'))
pprint(fm('Negara Indonesia merdeka [MASK] tahun 1945.'))
pprint(fm('Ayah bekerja [MASK] kantor.'))
pprint(fm('Kucing ku [MASK] bermain dengan tikus.'))
pprint(fm('ibu ku sedang bekerja [MASK] supermarket'))
pprint(fm('ibu ku sedang [MASK] di supermarket'))
