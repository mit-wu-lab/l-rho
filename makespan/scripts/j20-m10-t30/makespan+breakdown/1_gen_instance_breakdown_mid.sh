export INSTANCE_TYPE=breakdown
export NUM_JOBS=20   # number of jobs
export NUM_MACHINES=10   # number of machines
export NUM_OPS_PER_JOB=30   # number of operations per job
export DATA_DIR=train_data_dir   #parent directory to save the FJSP instances
export NUM_DATA=600   # number of FJSP instances to genreate

export NUM_MACHINE_BREAKDOWN_P=0.35
export FIRST_BREAKDOWN_BUFFER_LB=50
export FIRST_BREAKDOWN_BUFFER_UB=150
export MACHINE_BREAKDOWN_DURATION_LB=100
export MACHINE_BREAKDOWN_DURATION_UB=100
export BREAKDOWN_BUFFER_LB=175
export BREAKDOWN_BUFFER_UB=300
export BREAKDOWN_SUFFIX=mid

python -u gen_instance.py --instance_type $INSTANCE_TYPE --n_j $NUM_JOBS --n_m $NUM_MACHINES --op_per_job $NUM_OPS_PER_JOB --data_dir $DATA_DIR --n_data $NUM_DATA \
        --num_machine_breakdown_p $NUM_MACHINE_BREAKDOWN_P --first_breakdown_buffer_lb $FIRST_BREAKDOWN_BUFFER_LB --first_breakdown_buffer_ub $FIRST_BREAKDOWN_BUFFER_UB \
        --machine_breakdown_duration_lb $MACHINE_BREAKDOWN_DURATION_LB --machine_breakdown_duration_ub $MACHINE_BREAKDOWN_DURATION_UB \
        --breakdown_buffer_lb $BREAKDOWN_BUFFER_LB --breakdown_buffer_ub $BREAKDOWN_BUFFER_UB --breakdown_suffix $BREAKDOWN_SUFFIX