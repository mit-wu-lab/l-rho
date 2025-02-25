export N_MACHINES=25  # the number of machines
export N_JOBS=25  # the number of jobs
export N_OPS_PER_JOB=24  # the number of operations per job
export OPTIM_OPTION='start_delay'  # the objective
export DATA_IDX_START=0
export DATA_IDX_END=600  # the number of instances to generate

python flexible_jss_instance.py --n_machines $N_MACHINES --n_jobs $N_JOBS \
        --n_tasks_per_job $N_OPS_PER_JOB --optim_option $OPTIM_OPTION \
        --data_index_start $DATA_IDX_START --data_index_end $DATA_IDX_END