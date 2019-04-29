import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

filename = 'ratings_Musical_Instruments.csv'
df = pd.read_csv(filename,names=['userid','itemid','rating','timestamp'])
df = df.sort_values(['userid','timestamp'],ascending=[True,True])

df2 = df.groupby('userid').filter(lambda x : len(x) >= 10)
userset = np.unique(df2['userid'].values)



df2 = df2.drop (columns=['timestamp'])
dftrain = pd.DataFrame()
dftest = pd.DataFrame()
for i in userset:
    dftemp = df2[df2['userid'] == i]
    # print(dftemp)
    dftrain = dftrain.append(dftemp.iloc[:int(math.ceil(0.8*len(dftemp)))])
    dftest = dftest.append(dftemp.iloc[int(math.ceil(0.8*len(dftemp))):])

    



# df_xtrain, df_xtest = train_test_split(df2,shuffle = False, stratify = df['userid'])

# df_xtrain  = pd.DataFrame(df_xtrain)
# df_xtest = pd.DataFrame(df_xtest)

dftrain.to_csv('train.csv')
dftest.to_csv('test.csv')