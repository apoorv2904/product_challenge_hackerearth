N_val=320
exp_seed=0
for i in $(seq 1 1 5)
    do
        exp=$i
	echo "Training Model: $i"
	python groceries_train_clean.py --seed $exp --model_name DenseNet169 --batch_size 32
    done
