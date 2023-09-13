# Collect data
python3 collect_data.py --env darkroom_heldout --envs 100000 --H 100 --dim 10

# Train
python3 train.py --env darkroom_heldout --envs 100000 --H 100 --dim 10 --lr 0.001 --layer 4 --head 4 --shuffle --seed 1

# Evaluate, choose an appropriate epoch
python3 eval.py --env darkroom_heldout --envs 100000 --H 100 --dim 10 --lr 0.001 --layer 4 --head 4 --shuffle --epoch 200 --seed 1
