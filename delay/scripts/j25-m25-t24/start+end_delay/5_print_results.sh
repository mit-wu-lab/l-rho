export INDEX_START=500   # start index of the _test_ data
export INDEX_END=600  # end index of the _test_ data
export TEST_STATS_DIR=train_test_dir/test_stats/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-optimtype_start_and_end_delay-tlo0-thi30
export DATA_NAME=stats_e120th0.5   # name of the saved stats file, excluding the data idx suffix

python -u print_results.py --index_start $INDEX_START --index_end $INDEX_END --stats_dir $TEST_STATS_DIR --data_name $DATA_NAME 