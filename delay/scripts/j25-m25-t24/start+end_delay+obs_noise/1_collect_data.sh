# index of the FJSP instance to generate the training data (input and oracle labels)
export DATA_IDX=0   # [INSERT DATA INDEX HERE], we use a slurm script to generate data for DATA_IDX from 0 to 500 in parallel

# FJSP setting 
export N_MACHINES=25
export N_JOBS=25
export N_TASKS_PER_JOB=24
export OPTIM_OPTION='start_and_end_delay'
# end delay related parameters
export L_LOW_END=0
export L_HIGH_END=30
# observation noise parameters
export PERTURB_P=0.2

# Oracle's best parameter combination for the FJSP setting
export WINDOW=80
export STEP=25
export TIME_LIMIT=60
export EARLY_STOP_TIME=3

# directory to save the training data
export TRAIN_DATA_DIR=train_data_dir/train_data/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-optimtype_start_and_end_delay-tlo0-thi30-w80-s25-t60-st3-perturbo0-perturbp0.2
# FJSP instance directory (same as in start_and_end_delay (no obs noise) case; we generate obs. noise on the fly)
export FJSP_DATA_DIR=train_data_dir/instance/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-optimtype_start_and_end_delay-tlo0-thi30
# directory to save data collection statistics
export stats_dir=train_data_dir/stats/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-optimtype_start_and_end_delay-tlo0-thi30-w80-s25-t60-st3-tlo0-thi30-perturbo0-perturbp0.2


python flexible_jss_main.py --data_idx $DATA_IDX  \
        --n_machines $N_MACHINES --n_jobs $N_JOBS \
        --n_tasks_per_job $N_TASKS_PER_JOB --optim_option $OPTIM_OPTION \
        --l_low_end $L_LOW_END --l_high_end $L_HIGH_END \
        --perturb_data --perturb_p $PERTURB_P \
        --window $WINDOW --step $STEP --time_limit $TIME_LIMIT --stop_search_time $EARLY_STOP_TIME \
        --train_data_dir $TRAIN_DATA_DIR  --jss_data_dir $FJSP_DATA_DIR \
        --stats_dir $stats_dir --load_data