import pandas as pd
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
csv_type = {
    'Wall time': 'str',
    'Step': 'int',
    'Value': 'float'
}

df = [
    pd.read_csv('./other_script/data/0_eval_loss.csv', dtype=csv_type),
    pd.read_csv('./other_script/data/1_eval_loss.csv', dtype=csv_type),
    pd.read_csv('./other_script/data/0_train_epoch.csv', dtype=csv_type),
    pd.read_csv('./other_script/data/1_train_epoch.csv', dtype=csv_type),
    pd.read_csv('./other_script/data/0_train_learning_rate.csv', dtype=csv_type),
    pd.read_csv('./other_script/data/1_train_learning_rate.csv', dtype=csv_type),
    pd.read_csv('./other_script/data/0_train_loss.csv', dtype=csv_type),
    pd.read_csv('./other_script/data/1_train_loss.csv', dtype=csv_type),
]

# select <= 12900 for 0_
# +12900 for 1_

df[2] = df[2].query('Step < 12901')
df[4] = df[4].query('Step < 12901')
df[6] = df[6].query('Step < 12901')

df[1]['Step'] = df[1]['Step'].apply(lambda x: x+12900)
df[3]['Step'] = df[3]['Step'].apply(lambda x: x+12900)
df[5]['Step'] = df[5]['Step'].apply(lambda x: x+12900)
df[7]['Step'] = df[7]['Step'].apply(lambda x: x+12900)

df[3]['Value'] = df[3]['Value'].apply(lambda x: x+2.0)

# idx, 1-> Step, 2-> Value
for i in range(df[0].shape[0]):
    writer.add_scalar('eval/loss', df[0].iat[i, 2], df[0].iat[i, 1])
for i in range(df[1].shape[0]):
    writer.add_scalar('eval/loss', df[1].iat[i, 2], df[1].iat[i, 1])

for i in range(df[2].shape[0]):
    writer.add_scalar('train/epoch', df[2].iat[i, 2], df[2].iat[i, 1])
for i in range(df[3].shape[0]):
    writer.add_scalar('train/epoch', df[3].iat[i, 2], df[3].iat[i, 1])

for i in range(df[4].shape[0]):
    writer.add_scalar('train/learning_rate', df[4].iat[i, 2], df[4].iat[i, 1])
for i in range(df[5].shape[0]):
    writer.add_scalar('train/learning_rate', df[5].iat[i, 2], df[5].iat[i, 1])

for i in range(df[6].shape[0]):
    writer.add_scalar('train/loss', df[6].iat[i, 2], df[6].iat[i, 1])
for i in range(df[7].shape[0]):
    writer.add_scalar('train/loss', df[7].iat[i, 2], df[7].iat[i, 1])
