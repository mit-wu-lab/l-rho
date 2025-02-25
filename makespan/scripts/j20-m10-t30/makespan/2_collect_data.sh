export DATA_IDX=0  # [INSERT DATA INDEX HERE] we have a slurm script that runs DATA_IDX from 0 to 500 in parallel
export TRAIN_DATA_DIR=train_data_dir/train_data/j20-m10-t30_mix-w80-s30-t60-st3   # directory to save the training (and validation) data
export JSS_DATA_DIR=train_data_dir/instance/j20-m10-t30_mix   # directory of the FJSP instances
export STATS_DIR=train_data_dir/stats/j20-m10-t30_mix-w80-s30-t60-st3   # directory to save he statistics

export N_CPUS=3
export NUM_ORACLE_TRIALS=3
export WINDOW=80    # RHO subproblem size (number of operations)
export STEP=30   # RHO execution step size (number of operations)
export TIME_LIMIT=60   # time limit for each RHO subproblem in seconds
export STOP_SEARCH_TIME=3   # early stop if performance plateaued for STOP_SEARCH_TIME seconds

python -u flexible_jss_main.py --train_data_dir $TRAIN_DATA_DIR --jss_data_dir $JSS_DATA_DIR --stats_dir $STATS_DIR \
        --data_idx $DATA_IDX --time_limit $TIME_LIMIT --stop_search_time $STOP_SEARCH_TIME \
        --oracle_time_limit $TIME_LIMIT --oracle_stop_search_time $STOP_SEARCH_TIME \
        --n_cpus $N_CPUS --num_oracle_trials $NUM_ORACLE_TRIALS --window $WINDOW --step $STEP 

