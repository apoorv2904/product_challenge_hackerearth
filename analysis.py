import pandas as pd
import numpy as np

f1 = 'Predictions/sub_dense169_1.csv'
f2 = 'Predictions/sub_dense169_3.csv'

df_1 = pd.read_csv( f1 )
df_2 = pd.read_csv( f2 )

labels = df_1.label.unique()

y_df1 = df_1.label
y_df2 = df_2.label

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_df1, y_df2, labels=labels )

for i,label in enumerate( labels ):
    locs = np.where( cm[i,:] != 0 )[0]
    confusion_classes = labels[locs]
    confusion_count = cm[i,locs]
    confusion_percentage = cm[i,locs]/float(np.sum( cm[i,locs] ))
    print( np.sum( ( df_1.label==label).values & ( df_2.label==label).values ) / float(np.sum( cm[i,locs] )))
    print( label )
    print( confusion_classes )
    print( confusion_count )
    print( confusion_percentage )
    print( '************************ +++++ ************************')

