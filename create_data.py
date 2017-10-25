import numpy as np
import sys;
import random; 
import pandas as pd
random.seed(sys.argv[2]);

df=pd.read_csv('train.csv');    

data = range(len(df))
random.shuffle(data)


val_data = df.iloc[data[:int( sys.argv[1])]]
train_data = df.iloc[data[int(sys.argv[1]):]]


train_data.to_csv('Train_Csvs/train_' + sys.argv[3] + '.csv', index=False )
val_data.to_csv('Train_Csvs/val_' + sys.argv[3] + '.csv', index=False )

fname = 'Train_Csvs/conf_' +  sys.argv[3] + '.txt'
np.savetxt(fname, ['Seed: ' + sys.argv[2] + '\nN_Validation: ' + sys.argv[1] + '\nNetwork: ' + sys.argv[3]], fmt='%s')


