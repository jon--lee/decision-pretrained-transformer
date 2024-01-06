# Collect data
python3 collect_data.py --env linear_bandit --envs 1000000 --H 200 --dim 10 --var 0.3 --cov 0.0 --lin_d 2 --envs_eval 200

# Train
python3 train.py --env linear_bandit --envs 1000000 --H 200 --dim 10 --lin_d 2 --var 0.3 --cov 0.0 --lr 0.0001 --layer 4 --head 4 --seed 1

# Evaluate, choose an appropriate epoch
python3 eval.py --env linear_bandit --envs 1000000 --H 200 --dim 10 --lin_d 2 --var 0.3 --cov 0.0 --lr 0.0001 --layer 4 --head 4 --epoch 100 --n_eval 200 --seed 1
