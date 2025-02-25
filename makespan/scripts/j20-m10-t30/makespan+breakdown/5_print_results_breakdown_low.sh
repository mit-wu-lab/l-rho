export INDEX_START=500   # start index of the _test_ data
export INDEX_END=600  # end index of the _test_ data
export TEST_STATS_DIR=train_test_dir/machine_breakdown/test_stats/j20-m10-t30_mix-w80-s30-t60-st3-low   # directory of the saved test statistics
export DATA_NAME=stats_e120th0.5   # name of the saved stats file, excluding the data idx suffix

python -u print_results.py --index_start $INDEX_START --index_end $INDEX_END --stats_dir $TEST_STATS_DIR --data_name $DATA_NAME 