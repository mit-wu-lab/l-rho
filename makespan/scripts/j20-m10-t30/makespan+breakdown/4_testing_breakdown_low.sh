### the test data indices; note that we use a slurm script to test FJSP instances with IDX from 500 to 600 in parallel
export TEST_INDEX_START=500   # start index of the _test_ data
export TEST_INDEX_END=501   # end index of the _test_ data

export DATA_INDEX_START=0   # start index of the training data
export DATA_INDEX_END=450   # end index of the training data

### relevant directories: note that you can specify a different JSS_DATA_DIR (and the corresponding BREAKDOWN_DATA_DIR below) to test on different instances
export JSS_DATA_DIR=train_data_dir/instance/j20-m10-t30_mix   # directory of the FJSP instances
export TRAIN_DATA_DIR=train_data_dir/machine_breakdown/train_data/j20-m10-t30_mix-w80-s30-t60-st3-low   # directory where training data are saved
export LOG_DIR=train_test_dir/machine_breakdown/train_log/j20-m10-t30_mix-w80-s30-t60-st3-low   # directory to save the training logs
export TEST_STATS_DIR=train_test_dir/machine_breakdown/test_stats/j20-m10-t30_mix-w80-s30-t60-st3-low   # directory to save the test statistics
export MODEL_DIR=train_test_dir/machine_breakdown/model/j20-m10-t30_mix-w80-s30-t60-st3-low   # directory to save the trained models

export MODEL_NAME=model_pw0.5
export EPOCH=200
export POS_WEIGHT=0.5  # weight of the positive class for the weighted cross-entropy loss

export WINDOW=80
export STEP=30
export TIME_LIMIT=60
export STOP_SEARCH_TIME=3

export LOAD_MODEL_EPOCH=120  # epoch of the model to be loaded
export MODEL_TH=0.5   # test time prediction threshold for the model (predicted prob > MODEL_TH: fixed)

# breakdown parameters (you should modify this based on the setting you use to generate the breakdown data)
export BREAKDOWN_DATA_DIR=train_data_dir/machine_breakdown/breakdown_data/j20-m10-t30_mix-low
export NUM_MACHINE_BREAKDOWN_P=0.2
export FIRST_BREAKDOWN_BUFFER_LB=50
export FIRST_BREAKDOWN_BUFFER_UB=150
export MACHINE_BREAKDOWN_DURATION_LB=100
export MACHINE_BREAKDOWN_DURATION_UB=100
export BREAKDOWN_BUFFER_LB=400
export BREAKDOWN_BUFFER_UB=600

python -u flexible_jss_learn.py --train_data_dir $TRAIN_DATA_DIR --jss_data_dir $JSS_DATA_DIR --log_dir $LOG_DIR \
        --test_stats_dir $TEST_STATS_DIR --model_dir $MODEL_DIR --data_index_start $DATA_INDEX_START --data_index_end $DATA_INDEX_END \
        --val_index_start $TEST_INDEX_START --val_index_end $TEST_INDEX_END --num_epochs $EPOCH --model_name $MODEL_NAME \
        --window $WINDOW --step $STEP --time_limit $TIME_LIMIT --stop_search_time $STOP_SEARCH_TIME --pos_weight $POS_WEIGHT \
        --test --load_model_epoch $LOAD_MODEL_EPOCH --model_th $MODEL_TH \
        --machine_breakdown --breakdown_data_dir $BREAKDOWN_DATA_DIR \
        --num_machine_breakdown_p $NUM_MACHINE_BREAKDOWN_P \
        --first_breakdown_buffer_lb $FIRST_BREAKDOWN_BUFFER_LB \
        --first_breakdown_buffer_ub $FIRST_BREAKDOWN_BUFFER_UB \
        --machine_breakdown_duration_lb $MACHINE_BREAKDOWN_DURATION_LB \
        --machine_breakdown_duration_ub $MACHINE_BREAKDOWN_DURATION_UB \
        --breakdown_buffer_lb $BREAKDOWN_BUFFER_LB \
        --breakdown_buffer_ub $BREAKDOWN_BUFFER_UB

