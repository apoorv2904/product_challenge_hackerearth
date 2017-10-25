rm Features/* Checkpoints/* Predictions/* Validation/* VotingCandidates/* VotingResults/* logs/*
bash run_exp.sh
cp Predictions/* VotingCandidates/
python majorityVoting.py


