import pandas as pd
import numpy as np

import glob
import pickle
files = glob.glob("./VotingCandidates/*.csv")
print( files )

fnames = {}
predictions = {}
for f in files:
    fname=f
    predictions[f] = pd.read_csv( f )
    fnames[f] = fname

labels = predictions[files[0]].label.unique()
mapping = {}
inverse_mapping = {}
for i, label in enumerate(labels):
    mapping[label] = i
    inverse_mapping[i] = label

predictions_matrix = np.zeros( (len(predictions[files[0]]), len(files)))
for i,f in enumerate(files):
    predictions[f] = predictions[f].replace({'label': mapping})
    predictions_matrix[:,i] = predictions[f].label.values

majority_votes = np.zeros( predictions_matrix.shape[0] )
majority_count = np.zeros( predictions_matrix.shape[0] )

for row_idx in range(predictions_matrix.shape[0] ):
    preds = predictions_matrix[row_idx,:]
    #print( preds )
    majority_votes[row_idx] = np.argmax(np.bincount(np.asarray(preds,int)))
    majority_count[row_idx] = np.max(np.bincount(np.asarray(preds,int)))
    

# ---------------------------------------------------------------------------
# Following lines are for creating the submission file in the proper format
# ---------------------------------------------------------------------------

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

pred_labels = [inverse_mapping[k] for k in majority_votes]

## make submission
sub = pd.DataFrame({'image_id':test.image_id, 'label':pred_labels})
sub.to_csv('./VotingResults/sub_vgg_mean.csv', index=False) ## ~0.59
sub.to_csv('./VotingResults/ensemble_model.csv', index=False) ## ~0.59
sub = pd.DataFrame({'image_id':test.image_id, 'counts':majority_count})
sub.to_csv('./VotingResults/ensemble_model_counts.csv', index=False) ## ~0.59


confused_images = []
confusion_percentage = []
confusion_preds = []
for image_id, count, pred in zip( test.image_id, majority_count, pred_labels ):
    if (count/float(len(files))) < 0.7 :
	confused_images.append( image_id )
	confusion_percentage.append( count/float(len(files)))
	confusion_preds.append( pred )
	#print( pred )

confusion_percentage = np.asarray( confusion_percentage )
confusion_preds = np.asarray( confusion_preds )
confused_images = np.asarray( confused_images )

sub = pd.DataFrame({'image_id':confused_images, 'counts':confusion_percentage, 'label':confusion_preds})
sub.to_csv('./VotingResults/ensemble_confused_counts.csv', index=False) ## ~0.59

