import torch
from transformers import BigBirdTokenizerFast, BigBirdForMaskedLM
from pprint import pprint

tokenizer = BigBirdTokenizerFast.from_pretrained('ilos-vigil/bigbird-small-indonesian')
model = BigBirdForMaskedLM.from_pretrained('ilos-vigil/bigbird-small-indonesian')
topk = 5
text = 'Saya [MASK] bermain [MASK] teman saya.'

tokenized_text = tokenizer(text, return_tensors='pt')
raw_output = model(**tokenized_text)
tokenized_output = torch.topk(raw_output.logits, topk, dim=2).indices
score_output = torch.softmax(raw_output.logits, dim=2)

result = []
for position_idx in range(tokenized_text['input_ids'][0].shape[0]):
    if tokenized_text['input_ids'][0][position_idx] == tokenizer.mask_token_id:
        outputs = []
        for token_idx in tokenized_output[0, position_idx]:
            output = {}
            output['score'] = score_output[0, position_idx, token_idx].item()
            output['token'] = token_idx.item()
            output['token_str'] = tokenizer.decode(output['token'])
            outputs.append(output)
        result.append(outputs)

pprint(result)