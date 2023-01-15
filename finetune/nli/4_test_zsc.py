import pandas as pd
from itertools import product
from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report


BATCH_SIZE = 32
CACHE_DIR = './dataset'
MODELS = [
    'ilos-vigil/bigbird-small-indonesian-nli',
    'joeddav/xlm-roberta-large-xnli',
    'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
]


df_test = pd.read_csv('./finetune/nli/Twitter_Emotion_Dataset.csv')
X_test = df_test['tweet'].tolist()
y_test = df_test['label']

le = LabelEncoder()
y_test = le.fit_transform(y_test)
# le.classes_ -> ['anger', 'fear', 'happy', 'love', 'sadness']
label_emot = ['marah', 'takut', 'senang', 'cinta', 'sedih']


def test_zsc(model_name: str, use_template: bool, multilabel: bool):
    hypothesis_template = '{}.'
    if use_template:
        hypothesis_template = 'Kalimat ini mengekspresikan perasaan {}.'
    zsl = pipeline('zero-shot-classification', device=0, model=model_name, batch_size=BATCH_SIZE)

    y_pred = zsl(
        X_test,
        candidate_labels=label_emot,
        hypothesis_template=hypothesis_template,
        multi_label=multilabel
    )
    y_pred = [
        label_emot.index(y_pred[i]['labels'][0])
        for i
        in range(len(y_pred))
    ]

    print(f'Model: {model_name} | Template: {use_template} | Multilabel: {multilabel}')
    print('F1 Score:', f1_score(y_test, y_pred, average='micro'))
    print(classification_report(y_test, y_pred, target_names=label_emot))

    print('='*79)

for model_name, use_template, multilabel in product(MODELS, [True, False], [True, False]):
    test_zsc(model_name, use_template, multilabel)

