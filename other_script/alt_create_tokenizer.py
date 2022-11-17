from datasets import load_dataset
from tokenizers import SentencePieceBPETokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


CACHE_DIR = './dataset'
TOKENIZER_DIR = './tokenizer-bigbird-small-indonesian'
TYPE = 1  # 0 -> BPE, 1 -> SentencePieceBPE
SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<cls>", "<sep>", "<mask>"]  # match with reference bigbird-pegasus model on HuggingFace docs
special_token_map = {
    'pad_token': '<pad>',
    'unk_token': '<unk>',
    'bos_token': '<s>',
    'eos_token': '</s>',
    'sep_token': '[SEP]',
    'cls_token': '[CLS]',
    'mask_token': '[MASK]'
}

if TYPE == 0:
    tokenizer = BpeTrainer(BPE())
else:
    tokenizer = SentencePieceBPETokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()


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


if TYPE == 0:
    trainer = BpeTrainer(vocab_size=30000, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(batch_ds(), trainer)
else:
    tokenizer.train_from_iterator(
        batch_ds(),
        vocab_size=30_000,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS
    )

tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, model_max_length=4096, special_tokens=SPECIAL_TOKENS)
tokenizer.bos_token = "<s>"
tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
tokenizer.pad_token = "<pad>"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
tokenizer.eos_token = "</s>"
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
tokenizer.unk_token = "<unk>"
tokenizer.unk_token_id = tokenizer.convert_tokens_to_ids("<unk>")
tokenizer.cls_token = "<cls>"
tokenizer.cls_token_id = tokenizer.convert_tokens_to_ids("<cls>")
tokenizer.sep_token = "<sep>"
tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids("<sep>")
tokenizer.mask_token = "<mask>"
tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")
tokenizer.save_pretrained(TOKENIZER_DIR)
