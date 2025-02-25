export DATA_IDX=0  # INSERT DATA INDEX HERE
export JSS_DATA_DIR=train_data_dir/instance/j20-m10-t30_mix   # directory of the FJSP instances
export STATS_DIR=train_data_dir/debug_stats/j20-m10-t30_mix-w80-s30-t60-st3   # directory to save he statistics

export N_CPUS=3
export NUM_ORACLE_TRIALS=3
export WINDOW=80    # RHO subproblem size (number of operations)
export STEP=30   # RHO execution step size (number of operations)
export TIME_LIMIT=60   # time limit for each RHO subproblem in seconds
export STOP_SEARCH_TIME=3   # early stop if performance plateaued for STOP_SEARCH_TIME seconds

export SCRIPT_ACTION=debug

python -u flexible_jss_main.py --jss_data_dir $JSS_DATA_DIR --stats_dir $STATS_DIR \
        --data_idx $DATA_IDX --time_limit $TIME_LIMIT --stop_search_time $STOP_SEARCH_TIME \
        --oracle_time_limit $TIME_LIMIT --oracle_stop_search_time $STOP_SEARCH_TIME \
        --n_cpus $N_CPUS --num_oracle_trials $NUM_ORACLE_TRIALS --window $WINDOW --step $STEP \
        --script_action $SCRIPT_ACTION

### then you can run gen_analysis.py with the appropriate parameters (similar to step 5) to compare the behavior of Default RHO and Oracle RHO.
