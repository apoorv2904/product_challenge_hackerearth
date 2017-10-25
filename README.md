My Transfer Learning solution to Grocery Product Detection Challenge at HackerEarth
https://www.hackerearth.com/problem/machine-learning/identify-the-objects/

To run the code follow option 1 or 2

OPTION 1 <br />
 * Place Training Images in the folder "train_img" <br />
 * Place Training Images in the folder "test_img" <br />
 * Place pretrained modes in the folder "finetune_models/imagenet_models/" <br /> 
 	- Pretrained models can be found here <br />
	  https://github.com/flyyufelix/cnn_finetune <br />

OPTION 2 <br />
* Download the entire setup from the Google Drive Link <br />
  https://drive.google.com/open?id=0Bx84R0HH6GEIcXI4MTZOTm0yeHM <br />


To generate results, run the script "generate_submission.sh" <br />

The script does the following <br />
 * Trains 5 Densenet models using different validation subsets ( currently using DENSENET169 but this can changed ) <br />
 * Saves the predictions using the 5 models in the folder "Predictions" <br />
    - Uses Majority Voting to create a submisssion file "ensemble_model.csv"in the folder "VotingResults" <br />
    - Majority Voting can be done on other predictions by simpy placing the individual predictions in the folder VotingCandidates <br />
 * Additionally saves the features to build interesting models using stacking  <br />

