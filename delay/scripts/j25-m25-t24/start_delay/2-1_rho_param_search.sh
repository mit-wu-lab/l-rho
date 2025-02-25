# the index of the parameter to evaluate the performance
export PARAM_IDX=0   # [INSERT PARAMETER INDEX HERE] we use a slurm script to search over all parameters in parallel

export N_MACHINES=25  # the number of machines
export N_JOBS=25  # the number of jobs
export N_OPS_PER_JOB=24   # the number of operations per job
export OPTIM_OPTION='start_delay'   # the objective

# the indices of FJSP instances to perform parameter search
export DATA_IDX_START=0
export DATA_IDX_END=10  

# The directory to load the set of parameters to perform grid search on
export PARAM_DIR='train_test_dir/param_search/params/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct08-madapt-task-machine-cf3'
# The directory to save parameter grid search statistics to
export PARAM_STATS_DIR='train_test_dir/param_search/stats/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct08-madapt-task-machine-cf3'
# FJSP instance directory
export FJSP_DATA_DIR='train_data_dir/instance/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3'

python flexible_jss_parameter_search.py --n_machines $N_MACHINES --n_jobs $N_JOBS \
        --n_tasks_per_job $N_OPS_PER_JOB --optim_option $OPTIM_OPTION --param_idx $PARAM_IDX \
        --param_dir  $PARAM_DIR --param_stats_dir $PARAM_STATS_DIR --jss_data_dir $FJSP_DATA_DIR \
        --data_index_start $DATA_IDX_START --data_index_end $DATA_IDX_END

