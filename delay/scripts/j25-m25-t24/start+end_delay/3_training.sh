# training data indices
export DATA_INDEX_START=10
export DATA_INDEX_END=460
# validation data indices
export VAL_INDEX_START=460
export VAL_INDEX_END=480

# scaling factor for the positive labels (fixed operations)
export POS_WEIGHT=0.5
export MODEL_NAME=model_pw0.5

# number of training epochs
export EPOCHS=200

# FJSP setting 
export N_MACHINES=25
export N_JOBS=25
export N_TASKS_PER_JOB=24 
export OPTIM_OPTION='start_and_end_delay'
# end delay related parameters
export L_LOW_END=0
export L_HIGH_END=30

# Oracle's best parameter combination for the FJSP setting
export WINDOW=80
export STEP=25
export TIME_LIMIT=60
export EARLY_STOP_TIME=3

# directory that contains the training data 
export TRAIN_DATA_DIR=train_data_dir/train_data/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-optimtype_start_and_end_delay-tlo0-thi30-w80-s25-t60-st3
# directory that contains the FJSP instances
export FJSP_DATA_DIR=train_data_dir/instance/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-optimtype_start_and_end_delay-tlo0-thi30
# directory to save the training logs
export LOG_DIR=train_test_dir/train_log/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-optimtype_start_and_end_delay-tlo0-thi30-tlo0-thi30-w80-s25-t60-st3
# directory to save the test statistics
export TEST_STATS_DIR=train_test_dir/test_stats/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-optimtype_start_and_end_delay-tlo0-thi30-tlo0-thi30-w80-s25-t60-st3
# directory to save the model
export MODEL_DIR=train_test_dir/model/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-optimtype_start_and_end_delay-tlo0-thi30-tlo0-thi30-w80-s25-t60-st3


python flexible_jss_learn.py \
        --data_index_start $DATA_INDEX_START --data_index_end $DATA_INDEX_END \
        --val_index_start $VAL_INDEX_START --val_index_end $VAL_INDEX_END --num_epochs $EPOCHS --model_name $MODEL_NAME --pos_weight $POS_WEIGHT \
        --train_data_dir $TRAIN_DATA_DIR --jss_data_dir $FJSP_DATA_DIR \
        --log_dir $LOG_DIR --test_stats_dir $TEST_STATS_DIR --model_dir $MODEL_DIR \
        --n_machines $N_MACHINES --n_jobs $N_JOBS \
        --n_tasks_per_job $N_TASKS_PER_JOB --optim_option $OPTIM_OPTION \
        --l_low_end $L_LOW_END --l_high_end $L_HIGH_END \
        --window $WINDOW --step $STEP --time_limit $TIME_LIMIT --stop_search_time $EARLY_STOP_TIME