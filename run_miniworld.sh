# Collect data by running the below up to 60,000
# xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3 collect_data.py --env miniworld --H 50 --env_id_start 0 --env_id_end 4999
# xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3 collect_data.py --env miniworld --H 50 --env_id_start 5000 --env_id_end 9999
# ...
# Here is the looped version. It is recommended the run data collection in parallel.
for i in {0..11}; do
    start=$(( i * 5000 ))
    end=$(( (i + 1) * 5000 - 1 ))
    
    xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3 collect_data.py --env miniworld --H 50 --env_id_start $start --env_id_end $end
done



# Train
python3 train.py --env miniworld --envs 60000 --H 50 --lr 0.0001 --layer 4 --head 4 --shuffle

# Evaluate, choose an appropriate epoch
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3 eval.py --env miniworld --envs 60000 --H 50 --lr 0.0001 --layer 4 --head 4 --shuffle --epoch 200
