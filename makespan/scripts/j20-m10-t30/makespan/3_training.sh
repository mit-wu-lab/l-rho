export TRAIN_DATA_DIR=train_data_dir/train_data/j20-m10-t30_mix-w80-s30-t60-st3   # directory where training data are saved
export JSS_DATA_DIR=train_data_dir/instance/j20-m10-t30_mix   # directory of the FJSP instances
export LOG_DIR=train_test_dir/train_log/j20-m10-t30_mix-w80-s30-t60-st3   # directory to save the training logs
export TEST_STATS_DIR=train_test_dir/test_stats/j20-m10-t30_mix-w80-s30-t60-st3   # directory to save the test statistics
export MODEL_DIR=train_test_dir/model/j20-m10-t30_mix-w80-s30-t60-st3   # directory to save the trained models

export DATA_INDEX_START=0   # start index of the training data
export DATA_INDEX_END=450   # end index of the training data
export VAL_INDEX_START=450   # start index of the validation data
export VAL_INDEX_END=470   # end index of the validation data

export EPOCH=200   # number of training epochs
export MODEL_NAME=model_pw0.5
export WINDOW=80
export STEP=30
export TIME_LIMIT=60
export STOP_SEARCH_TIME=3
export POS_WEIGHT=0.5   # weight of the positive class for the weighted cross-entropy loss

python -u flexible_jss_learn.py --train_data_dir $TRAIN_DATA_DIR --jss_data_dir $JSS_DATA_DIR --log_dir $LOG_DIR \
        --test_stats_dir $TEST_STATS_DIR --model_dir $MODEL_DIR --data_index_start $DATA_INDEX_START --data_index_end $DATA_INDEX_END \
        --val_index_start $VAL_INDEX_START --val_index_end $VAL_INDEX_END --num_epochs $EPOCH --model_name $MODEL_NAME \
        --window $WINDOW --step $STEP --time_limit $TIME_LIMIT --stop_search_time $STOP_SEARCH_TIME --pos_weight $POS_WEIGHT
