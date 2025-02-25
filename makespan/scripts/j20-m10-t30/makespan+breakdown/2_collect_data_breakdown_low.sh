export DATA_IDX=0  # [INSERT DATA INDEX HERE] we have a slurm script that runs DATA_IDX from 0 to 500 in parallel
export TRAIN_DATA_DIR=train_data_dir/machine_breakdown/train_data/j20-m10-t30_mix-w80-s30-t60-st3-low
export JSS_DATA_DIR=train_data_dir/instance/j20-m10-t30_mix
export STATS_DIR=train_data_dir/machine_breakdown/stats/j20-m10-t30_mix-w80-s30-t60-st3-low
export PLOTS_DIR=train_data_dir/machine_breakdown/plots/j20-m10-t30_mix-w80-s30-t60-st3-low
export ORACLE_TIME_LIMIT=60
export ORACLE_STOP_SEARCH_TIME=3
export N_CPUS=3
export NUM_ORACLE_TRIALS=3
export WINDOW=80
export STEP=30
export TIME_LIMIT=60
export STOP_SEARCH_TIME=3

### breakdown parameters (you should modify this based on the setting you use to generate the breakdown data)
export BREAKDOWN_DATA_DIR=train_data_dir/machine_breakdown/breakdown_data/j20-m10-t30_mix-low
export NUM_MACHINE_BREAKDOWN_P=0.2
export FIRST_BREAKDOWN_BUFFER_LB=50
export FIRST_BREAKDOWN_BUFFER_UB=150
export MACHINE_BREAKDOWN_DURATION_LB=100
export MACHINE_BREAKDOWN_DURATION_UB=100
export BREAKDOWN_BUFFER_LB=400
export BREAKDOWN_BUFFER_UB=600

python -u flexible_jss_main_machine_breakdown.py --train_data_dir $TRAIN_DATA_DIR --jss_data_dir $JSS_DATA_DIR --stats_dir $STATS_DIR \
        --data_idx $DATA_IDX --time_limit $TIME_LIMIT --stop_search_time $STOP_SEARCH_TIME \
        --oracle_time_limit $ORACLE_TIME_LIMIT --oracle_stop_search_time $ORACLE_STOP_SEARCH_TIME --n_cpus $N_CPUS \
        --num_oracle_trials $NUM_ORACLE_TRIALS --window $WINDOW --step $STEP  \
        --breakdown_data_dir $BREAKDOWN_DATA_DIR \
        --num_machine_breakdown_p $NUM_MACHINE_BREAKDOWN_P \
        --first_breakdown_buffer_lb $FIRST_BREAKDOWN_BUFFER_LB \
        --first_breakdown_buffer_ub $FIRST_BREAKDOWN_BUFFER_UB \
        --machine_breakdown_duration_lb $MACHINE_BREAKDOWN_DURATION_LB \
        --machine_breakdown_duration_ub $MACHINE_BREAKDOWN_DURATION_UB \
        --breakdown_buffer_lb $BREAKDOWN_BUFFER_LB \
        --breakdown_buffer_ub $BREAKDOWN_BUFFER_UB