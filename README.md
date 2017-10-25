My Transfer Learning solution to Grocery Product Detection Challenge at HackerEarth
https://www.hackerearth.com/problem/machine-learning/identify-the-objects/

This is the barebone setup. To run the code follow option 1 or 2

OPTION 1
    - Place Training Images in the folder "train_img"
    - Place Training Images in the folder "test_img"
    - Place pretrained modes in the folder "finetune_models/imagenet_models/" 
	- Pretrained models can be found here
	  https://github.com/flyyufelix/cnn_finetune

OPTION 2
    - Download the entire setup from the Google Drive Link
      https://drive.google.com/open?id=0Bx84R0HH6GEIcXI4MTZOTm0yeHM


To generate results, run the script "generate_submission.sh"

The script does the following
    - Trains 5 Densenet models using different validation subsets ( currently using DENSENET169 but this can changed )
    - Saves the predictions using the 5 models in the folder "Predictions"
    - Uses Majority Voting to create a submisssion file "ensemble_model.csv"in the folder "VotingResults"
    	- Majority Voting can be done on other predictions by simpy placing the individual predictions in the folder VotingCandidates
    - Additionally saves the features to build interesting models using stacking 
