export INSTANCE_TYPE=standard
export NUM_JOBS=20   # number of jobs
export NUM_MACHINES=10   # number of machines
export NUM_OPS_PER_JOB=30   # number of operations per job
export DATA_DIR=train_data_dir   #parent directory to save the FJSP instances
export NUM_DATA=600   # number of FJSP instances to genreate

python -u gen_instance.py --instance_type $INSTANCE_TYPE --n_j $NUM_JOBS --n_m $NUM_MACHINES --op_per_job $NUM_OPS_PER_JOB --data_dir $DATA_DIR --n_data $NUM_DATA