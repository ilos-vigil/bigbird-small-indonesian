import pandas as pd
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
csv_type = {
    'Wall time': 'str',
    'Step': 'int',
    'Value': 'float'
}

dfs = [
    ['eval/accuracy', pd.read_csv('./other_script/data2/eval_accuracy.csv', dtype=csv_type)],
    ['eval/loss',pd.read_csv('./other_script/data2/eval_loss.csv', dtype=csv_type)],
    ['train/epoch',pd.read_csv('./other_script/data2/train_epoch.csv', dtype=csv_type)],
    ['train/learning_rate',pd.read_csv('./other_script/data2/train_learning_rate.csv', dtype=csv_type)],
    ['train/loss',pd.read_csv('./other_script/data2/train_loss.csv', dtype=csv_type)],
]

for name, df in dfs:
    for i in range(df.shape[0]):
        writer.add_scalar(name, df.iat[i, 2], df.iat[i, 1])
