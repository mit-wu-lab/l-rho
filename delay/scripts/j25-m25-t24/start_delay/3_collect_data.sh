# index of the FJSP instance to generate the training data (input and oracle labels)
export DATA_IDX=0   # [INSERT DATA INDEX HERE], we use a slurm script to generate data for DATA_IDX from 0 to 500 in parallel

# FJSP setting 
export N_MACHINES=25
export N_JOBS=25
export N_TASKS_PER_JOB=24
export OPTIM_OPTION='start_delay'

# Oracle's best parameter combination for the FJSP setting [MODIFY THIS BASED ON THE RESULT OF STEP 2]
export WINDOW=80
export STEP=25
export TIME_LIMIT=60
export EARLY_STOP_TIME=3

# directory to save the training data
export TRAIN_DATA_DIR='train_data_dir/train_data/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-w80-s25-t60-st3'
# FJSP instance directory
export FJSP_DATA_DIR='train_data_dir/instance/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3'
# directory to save data collection statistics
export stats_dir='train_data_dir/stats/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-w80-s25-t60-st3'


python flexible_jss_main.py --data_idx $DATA_IDX  \
        --n_machines $N_MACHINES --n_jobs $N_JOBS \
        --n_tasks_per_job $N_TASKS_PER_JOB --optim_option $OPTIM_OPTION \
        --window $WINDOW --step $STEP --time_limit $TIME_LIMIT --stop_search_time $EARLY_STOP_TIME \
        --train_data_dir $TRAIN_DATA_DIR  --jss_data_dir $FJSP_DATA_DIR \
        --stats_dir $stats_dir --load_data