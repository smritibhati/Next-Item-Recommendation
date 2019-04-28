import pandas as pd
filename = 'ratings_Baby.csv'
df = pd.read_csv(filename,names=['userid','itemid','rating','timestamp'])
df = df.sort_values(['userid','timestamp'],ascending=[True,True])
df2 = df.groupby('userid').filter(lambda x : len(x) >= 10)
df2.to_csv('baby_pruned_2.csv')
