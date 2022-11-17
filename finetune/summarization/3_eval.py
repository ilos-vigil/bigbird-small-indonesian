from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_from_disk
import evaluate
from transformers import (
    BigBirdTokenizerFast,
    AutoModelForSeq2SeqLM
)
from tqdm import tqdm

TOKENIZER_DIR = './tokenizer-bigbird-small-indonesian'
MODEL_DIR = './model-bigbird-small-indonesian-summarization'
DS_EVAL_DIR = './dataset/finetune_test'
BATCH_SIZE = 16


def decode_token(batch):
    batch['input'] = tokenizer.decode(batch['input_ids'], skip_special_tokens=True)
    batch['output'] = tokenizer.decode(batch['decoder_input_ids'], skip_special_tokens=True)
    return batch

tokenizer = BigBirdTokenizerFast.from_pretrained(TOKENIZER_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
model = model.bfloat16()
ds_test = load_from_disk('./dataset/finetune_test')
ds_test = ds_test.map(
    decode_token,
    remove_columns=ds_test.column_names
)

# change model/tokenizer config here
tokenizer.model_max_length = 4094 # fix output with shape [BATCH_SIZE, 1, 512] doesn't match the broadcast shape [BATCH_SIZE, 0, 512]
# model.config.max_length = 512
# model.config.min_length = 32
# model.config.no_repeat_ngram_size = 3
model.config.early_stopping = True
# model.config.length_penalty = 2.0
# model.config.num_beams = 4


metric = evaluate.load('rouge')
pipe = pipeline('summarization', model=model, tokenizer=tokenizer, device=0)

pred = []
for out in tqdm(pipe(KeyDataset(ds_test, 'input'), batch_size=BATCH_SIZE, truncation=True)):
    pred.append(out)

pred_filtered = [p[0]['summary_text'] for p in pred]
with open('./finetune/summarization/pred_result.txt', 'w') as f:
    for p in pred_filtered:
        f.write(p + '\n')

result = metric.compute(predictions=pred_filtered, references=ds_test['output'])
print(result)
