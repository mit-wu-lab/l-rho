### the test data indices; note that we use a slurm script to test FJSP instances with IDX from 500 to 600 in parallel
export TEST_INDEX_START=500
export TEST_INDEX_END=501

# training data indices
export DATA_INDEX_START=10
export DATA_INDEX_END=460

# pretrained model name
export MODEL_NAME=model_pw0.5
export POS_WEIGHT=0.5

# which epoch to load the pretrained model
export LOAD_MODEL_EPOCH=120

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
# (only used if BEST_PARAM_FILE below does not exist)
export WINDOW=80
export STEP=25
export TIME_LIMIT=60
export EARLY_STOP_TIME=3

# directory that contains the training data 
export TRAIN_DATA_DIR=train_data_dir/train_data/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-optimtype_start_and_end_delay-tlo0-thi30-w80-s25-t60-st3-perturbo0-perturbp0.2
# directory that contains the FJSP instances
export FJSP_DATA_DIR=train_data_dir/instance/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-optimtype_start_and_end_delay-tlo0-thi30
# directory to save the training logs
export LOG_DIR=train_test_dir/train_log/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-optimtype_start_and_end_delay-tlo0-thi30-w80-s25-t60-st3-perturbo0-perturbp0.2

# directory to save the test performance statistics 
export TEST_STATS_DIR=train_test_dir/test_stats/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-optimtype_start_and_end_delay-tlo0-thi30-w80-s25-t60-st3-perturbo0-perturbp0.2
# directory that contains the pretrained model
export MODEL_DIR=train_test_dir/model/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-optimtype_start_and_end_delay-tlo0-thi30-w80-s25-t60-st3-perturbo0-perturbp0.2
# path to the best RHO parameter file (obtained from the parameter search procedure; 
# by default, we use the same best parameter as in the case of start delay, but you can repeat the parameter grid search procedure to find the best parameter for this setting)
export BEST_PARAM_FILE=train_test_dir/param_search/params/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3/best_params.pkl

python flexible_jss_learn.py --data_index_start $DATA_INDEX_START --data_index_end $DATA_INDEX_END \
        --val_index_start $TEST_INDEX_START --val_index_end $TEST_INDEX_END --model_name $MODEL_NAME \
        --pos_weight $POS_WEIGHT --test --load_model_epoch $LOAD_MODEL_EPOCH  \
        --n_machines $N_MACHINES --n_jobs $N_JOBS \
        --n_tasks_per_job $N_TASKS_PER_JOB --optim_option $OPTIM_OPTION \
        --l_low_end $L_LOW_END --l_high_end $L_HIGH_END \
        --perturb_data --perturb_p $PERTURB_P \
        --window $WINDOW --step $STEP --time_limit $TIME_LIMIT --stop_search_time $EARLY_STOP_TIME \
        --train_data_dir $TRAIN_DATA_DIR --jss_data_dir $FJSP_DATA_DIR \
        --log_dir $LOG_DIR --test_stats_dir $TEST_STATS_DIR --model_dir $MODEL_DIR \
        --load_best_params --best_params_file $BEST_PARAM_FILE