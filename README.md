# Learning-Guided Rolling Horizon Optimization for Long-Horizon Flexible Job Shop Scheduling


This directory is the official implementation for our ICLR 2025 paper titled `Learning-Guided Rolling Horizon Optimization for Long-Horizon Flexible Job Shop Scheduling`. This README file provides instructions on environment setup, FJSP instance generation, training data collection, model training and testing.


## Relevant Links
You may find this project at: [ArXiv](http://arxiv.org/abs/2502.15791) and [OpenReview](https://openreview.net/forum?id=Aly68Y5Es0).

```
@inproceedings{li2025learning,
title={Learning-Guided Rolling Horizon Optimization for Long-Horizon Flexible Job-Shop Scheduling},
author={Li, Sirui and Ouyang, Wenbin and Ma, Yining and Wu, Cathy},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025}
}
```

## Environment Setup

Our implementation uses python 3.9.16, Pytorch 2.0.0, OR-Tools 9.9.3963. The other package dependencies are listed in `requirement.txt` and can be installed with the following command:

```
conda create --name lrho --file requirements.txt
conda activate lrho
```
Or, from your existing environment, do
```
pip install -r requirements.txt
```

## Usage

| FJSP Objective | Code Directory  | Considered Variants |                              |                                  |
| -------------- | ---------- | ------------------- | ---------------------------- | -------------------------------- |
| [Makespan](#makespan-variants)       | `./makespan` | [Makespan](#makespan)            | [Makespan + Machine Breakdown](#makespan--machine-breakdown) |                                  |
| [Delay](#delay-variants)          | `./delay`    | [Start Delay](#start-delay)         | [Start and End Delay](#start-and-end-delay)          | [Start and End Delay + Obs. Noise](#start-and-end-delay--obs-noise) |

---
### Makespan Variants
Please go to the `./makespan` directory by `cd ./makespan`. The example scripts can be found in `./makespan/scripts/j20-m10-t30`. We provide detailed instructions below:
#### Makespan
<details>
  <summary>1. FJSP Instance Generation</summary>

The FJSP instances will be generated under `{DATA_DIR}/instance/j{NUM_JOBS}-m{NUM_MACHINES}-t{NUM_OPS_PER_JOB}_mix`.

```script
export INSTANCE_TYPE=standard
export NUM_JOBS=20   # number of jobs
export NUM_MACHINES=10   # number of machines
export NUM_OPS_PER_JOB=30   # number of operations per job
export DATA_DIR=train_data_dir   #parent directory to save the FJSP instances
export NUM_DATA=600   # number of FJSP instances to genreate

python -u gen_instance.py --instance_type $INSTANCE_TYPE --n_j $NUM_JOBS --n_m $NUM_MACHINES --op_per_job $NUM_OPS_PER_JOB --data_dir $DATA_DIR --n_data $NUM_DATA
```
</details>

<details>
  <summary>2. Training Data Collection (Lookahead Oracle)</summary>

The following script generate the training data for FJSP instance idx `DATA_IDX`. In our experiments, we run DATA_IDX from 0 to 500 to generate training (and validation) data. You should make sure `JSS_DATA_DIR` contains the previously generated FJSP instances.

```script
export DATA_IDX=0  # [INSERT DATA INDEX HERE] we have a slurm script that runs DATA_IDX from 0 to 500 in parallel
export TRAIN_DATA_DIR=train_data_dir/train_data/j20-m10-t30_mix-w80-s30-t60-st3   # directory to save the training (and validation) data
export JSS_DATA_DIR=train_data_dir/instance/j20-m10-t30_mix   # directory of the FJSP instances
export STATS_DIR=train_data_dir/stats/j20-m10-t30_mix-w80-s30-t60-st3   # directory to save he statistics

export N_CPUS=3
export NUM_ORACLE_TRIALS=3
export WINDOW=80    # RHO subproblem size (number of operations)
export STEP=30   # RHO execution step size (number of operations)
export TIME_LIMIT=60   # time limit for each RHO subproblem in seconds
export STOP_SEARCH_TIME=3   # early stop if performance plateaued for STOP_SEARCH_TIME seconds

python -u flexible_jss_main.py --train_data_dir $TRAIN_DATA_DIR --jss_data_dir $JSS_DATA_DIR --stats_dir $STATS_DIR \
        --data_idx $DATA_IDX --time_limit $TIME_LIMIT --stop_search_time $STOP_SEARCH_TIME \
        --oracle_time_limit $TIME_LIMIT --oracle_stop_search_time $STOP_SEARCH_TIME \
        --n_cpus $N_CPUS --num_oracle_trials $NUM_ORACLE_TRIALS --window $WINDOW --step $STEP 
```
</details>

<details>
  <summary>3. Model Training</summary>

We can use the following script to train the model. You should make sure `TRAIN_DATA_DIR`contains the previously collected training data and `JSS_DATA_DIR` contains the previously generated FJSP instances. The trained models will be saved at `MODEL_DIR`.

```script
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

```
</details>

<details>
  <summary>4. Testing (Rollout)</summary>
  
After the model is trained, we can use the following script to test the model by solving FJSP instances using our proposed L-RHO algorithm. You should make sure the relevant directories below are specified correctly. For example, `JSS_DATA_DIR` should contain the previously generated FJSP instances, and `MODEL_DIR` should contain the previously trained model. We can specify different test FSJP distribution by using a different `JSS_DATA_DIR`, while keeping the other parameters the same.  You should specify the index of the testing FJSP instances by setting `VAL_INDEX_START` and `VAL_INDEX_END` below properly, and add `--test` flag when running the python script. We save the test statistics in the `TEST_STATS_DIR` directory.


```script
### the test data indices; note that we use a slurm script to test FJSP instances with IDX from 500 to 600 in parallel
export TEST_INDEX_START=500   # start index of the _test_ data
export TEST_INDEX_END=501   # end index of the _test_ data

export DATA_INDEX_START=0   # start index of the training data
export DATA_INDEX_END=450   # end index of the training data

### relevant directories: note that you can specify a different JSS_DATA_DIR to test on different instances
export JSS_DATA_DIR=train_data_dir/instance/j20-m10-t30_mix   # directory of the FJSP instances
export TRAIN_DATA_DIR=train_data_dir/train_data/j20-m10-t30_mix-w80-s30-t60-st3   # directory where training data are saved
export LOG_DIR=train_test_dir/train_log/j20-m10-t30_mix-w80-s30-t60-st3   # directory to save the training logs
export TEST_STATS_DIR=train_test_dir/test_stats/j20-m10-t30_mix-w80-s30-t60-st3   # directory to save the test statistics
export MODEL_DIR=train_test_dir/model/j20-m10-t30_mix-w80-s30-t60-st3   # directory where the trained models are saved 

export MODEL_NAME=model_pw0.5   # saved model name
export EPOCH=200   # number of training epochs
export POS_WEIGHT=0.5  # weight of the positive class for the weighted cross-entropy loss

export WINDOW=80
export STEP=30
export TIME_LIMIT=60
export STOP_SEARCH_TIME=3

export LOAD_MODEL_EPOCH=120   # epoch of the model to be loaded
export MODEL_TH=0.5   # test time prediction threshold for the model (predicted prob > MODEL_TH: fixed)

python -u flexible_jss_learn.py --train_data_dir $TRAIN_DATA_DIR --jss_data_dir $JSS_DATA_DIR --log_dir $LOG_DIR \
        --test_stats_dir $TEST_STATS_DIR --model_dir $MODEL_DIR --data_index_start $DATA_INDEX_START --data_index_end $DATA_INDEX_END \
        --val_index_start $TEST_INDEX_START --val_index_end $TEST_INDEX_END --num_epochs $EPOCH --model_name $MODEL_NAME \
        --window $WINDOW --step $STEP --time_limit $TIME_LIMIT --stop_search_time $STOP_SEARCH_TIME --pos_weight $POS_WEIGHT \
        --test --load_model_epoch $LOAD_MODEL_EPOCH --model_th $MODEL_TH 
```
</details>

<details>
  <summary>5. Print Test Stats</summary>

We can use the `print_results.py` script to print the test results, with the test stats directory `TEST_STATS_DIR` and the data name `DATA_NAME` (name of the saved stats file, excluding the data idx suffix) properly specified.
  
```script
export INDEX_START=500   # start index of the _test_ data
export INDEX_END=600  # end index of the _test_ data
export TEST_STATS_DIR=train_test_dir/test_stats/j20-m10-t30_mix-w80-s30-t60-st3/model_pw0.5   # directory of the saved test statistics
export DATA_NAME=stats_e120th0.5   # name of the saved stats file, excluding the data idx suffix

python -u print_results.py --index_start $INDEX_START --index_end $INDEX_END --stats_dir $TEST_STATS_DIR --data_name $DATA_NAME 
```
</details>

<details>
  <summary>6. Troubleshoot</summary>
As discussed in the theory section of our paper, L-RHO may not always be able to Default RHO due to the following reasons: (1) The Lookahead Oracle may not be good enough to accelerate Default RHO (2) The learning model cannot learn the lookahead oracle.

- For (1), you should run the following script to solve FJSP instances using both Default RHO and Oracle RHO, and check if Oracle RHO can improve from Default RHO. You should also consider varying WINDOW, STEP, TIME_LIMIT, and STOP_SEARCH_TIME parameters (e.g. via a grid search) to see what parameter set leads to the best behavior. We provide a sample grid search code for the delay-based variants [below](#delay-variants).
  
- For (2), you should consider modifying the learning pipeline to improve the learning ability.

```script
export DATA_IDX=0  # INSERT DATA INDEX HERE
export JSS_DATA_DIR=train_data_dir/instance/j20-m10-t30_mix   # directory of the FJSP instances
export STATS_DIR=train_data_dir/debug_stats/j20-m10-t30_mix-w80-s30-t60-st3   # directory to save he statistics

export N_CPUS=3
export NUM_ORACLE_TRIALS=3
export WINDOW=80    # RHO subproblem size (number of operations)
export STEP=30   # RHO execution step size (number of operations)
export TIME_LIMIT=60   # time limit for each RHO subproblem in seconds
export STOP_SEARCH_TIME=3   # early stop if performance plateaued for STOP_SEARCH_TIME seconds

export SCRIPT_ACTION=debug

python -u flexible_jss_main.py --jss_data_dir $JSS_DATA_DIR --stats_dir $STATS_DIR \
        --data_idx $DATA_IDX --time_limit $TIME_LIMIT --stop_search_time $STOP_SEARCH_TIME \
        --oracle_time_limit $TIME_LIMIT --oracle_stop_search_time $STOP_SEARCH_TIME \
        --n_cpus $N_CPUS --num_oracle_trials $NUM_ORACLE_TRIALS --window $WINDOW --step $STEP \
        --script_action $SCRIPT_ACTION

### then we can run gen_analysis.py with the appropriate parameters (similar to step 5) to compare the behavior of Default RHO and Oracle RHO.
```
</details>

#### Makespan + Machine Breakdown
<details>
  <summary>1. FJSP Breakdown Events Generation</summary>

The FJSP breakdown events will be generated under `{DATA_DIR}/machine_breakdown/breakdown_data/j{NUM_JOBS}-m{NUM_MACHINES}-t{NUM_OPS_PER_JOB}_mix-{BREAKDOWN_SUFFIX}`. If you have already generated FJSP instances from the makespan variant above, we will reuse those instances instead of generating new ones. Otherwise, we will also generate FJSP instances and similarly save them under `{DATA_DIR}/instance/j{NUM_JOBS}-m{NUM_MACHINES}-t{NUM_OPS_PER_JOB}_mix`; 

```script
export INSTANCE_TYPE=breakdown
export NUM_JOBS=20   # number of jobs
export NUM_MACHINES=10   # number of machines
export NUM_OPS_PER_JOB=30   # number of operations per job
export DATA_DIR=train_data_dir   #parent directory to save the FJSP instances
export NUM_DATA=600   # number of FJSP instances to genreate

export NUM_MACHINE_BREAKDOWN_P=0.2   # we use low: 0.2, mid: 0.35, high: 0.5
export FIRST_BREAKDOWN_BUFFER_LB=50   # we use the same 50 for low, mid, and high
export FIRST_BREAKDOWN_BUFFER_UB=150   # we use the same 50 for low, mid, and high
export MACHINE_BREAKDOWN_DURATION_LB=100   # we use low: 100, mid: 100, high: 50
export MACHINE_BREAKDOWN_DURATION_UB=100   # we use low: 100, mid: 100, high: 50
export BREAKDOWN_BUFFER_LB=400   # we use low: 400, mid: 175, high: 100
export BREAKDOWN_BUFFER_UB=600   # we use low: 600, mid: 300, high: 200
export BREAKDOWN_SUFFIX=low   # we use low, mid, or high

python -u gen_instance.py --instance_type $INSTANCE_TYPE --n_j $NUM_JOBS --n_m $NUM_MACHINES --op_per_job $NUM_OPS_PER_JOB --data_dir $DATA_DIR --n_data $NUM_DATA \
        --num_machine_breakdown_p $NUM_MACHINE_BREAKDOWN_P --first_breakdown_buffer_lb $FIRST_BREAKDOWN_BUFFER_LB --first_breakdown_buffer_ub $FIRST_BREAKDOWN_BUFFER_UB \
        --machine_breakdown_duration_lb $MACHINE_BREAKDOWN_DURATION_LB --machine_breakdown_duration_ub $MACHINE_BREAKDOWN_DURATION_UB \
        --breakdown_buffer_lb $BREAKDOWN_BUFFER_LB --breakdown_buffer_ub $BREAKDOWN_BUFFER_UB --breakdown_suffix $BREAKDOWN_SUFFIX
```
</details>

<details>
  <summary>2. Training Data Collection (Lookahead Oracle)</summary>

The following script generate the training data for FJSP instance idx `DATA_IDX`. In our experiments, we run DATA_IDX from 0 to 500 to generate training (and validation) data.
  
```script
export DATA_IDX=0  # [INSERT DATA INDEX HERE] we have a slurm script that runs DATA_IDX from 0 to 500 in parallel
export TRAIN_DATA_DIR=train_data_dir/machine_breakdown/train_data/j20-m10-t30_mix-w80-s30-t60-st3-mid
export JSS_DATA_DIR=train_data_dir/instance/j20-m10-t30_mix
export STATS_DIR=train_data_dir/machine_breakdown/stats/j20-m10-t30_mix-w80-s30-t60-st3-mid
export PLOTS_DIR=train_data_dir/machine_breakdown/plots/j20-m10-t30_mix-w80-s30-t60-st3-mid
export ORACLE_TIME_LIMIT=60
export ORACLE_STOP_SEARCH_TIME=3
export N_CPUS=3
export NUM_ORACLE_TRIALS=3
export WINDOW=80
export STEP=30
export TIME_LIMIT=60
export STOP_SEARCH_TIME=3

### breakdown parameters (you should modify this based on the setting you use to generate the breakdown data)
export BREAKDOWN_DATA_DIR=train_data_dir/machine_breakdown/breakdown_data/j20-m10-t30_mix-mid
export NUM_MACHINE_BREAKDOWN_P=0.35
export FIRST_BREAKDOWN_BUFFER_LB=50
export FIRST_BREAKDOWN_BUFFER_UB=150
export MACHINE_BREAKDOWN_DURATION_LB=100
export MACHINE_BREAKDOWN_DURATION_UB=100
export BREAKDOWN_BUFFER_LB=175
export BREAKDOWN_BUFFER_UB=300

python -u flexible_jss_main_machine_breakdown.py --train_data_dir $TRAIN_DATA_DIR --jss_data_dir $JSS_DATA_DIR --stats_dir $STATS_DIR \
        --data_idx $DATA_IDX --time_limit $TIME_LIMIT --stop_search_time $STOP_SEARCH_TIME \
        --oracle_time_limit $ORACLE_TIME_LIMIT --oracle_stop_search_time $ORACLE_STOP_SEARCH_TIME --n_cpus $N_CPUS \
        --num_oracle_trials $NUM_ORACLE_TRIALS --window $WINDOW --step $STEP  \
        --breakdown_data_dir $BREAKDOWN_DATA_DIR \
        --num_machine_breakdown_p $NUM_MACHINE_BREAKDOWN_P \
        --first_breakdown_buffer_lb $FIRST_BREAKDOWN_BUFFER_LB \
        --first_breakdown_buffer_ub $FIRST_BREAKDOWN_BUFFER_UB \
        --machine_breakdown_duration_lb $MACHINE_BREAKDOWN_DURATION_LB \
        --machine_breakdown_duration_ub $MACHINE_BREAKDOWN_DURATION_UB \
        --breakdown_buffer_lb $BREAKDOWN_BUFFER_LB \
        --breakdown_buffer_ub $BREAKDOWN_BUFFER_UB
```
</details>

<details>
  <summary>3. Model Training</summary>

We can use the following script to train the model. You should make sure `TRAIN_DATA_DIR` contains the previously collected training data, `JSS_DATA_DIR` contains the previously generated FJSP instances, and `BREAKDOWN_DATA_DIR` contains the previously generated breakdown events data. The trained models will be saved at `MODEL_DIR`.

```script
export TRAIN_DATA_DIR=train_data_dir/machine_breakdown/train_data/j20-m10-t30_mix-w80-s30-t60-st3-mid   # directory where training data are saved
export JSS_DATA_DIR=train_data_dir/instance/j20-m10-t30_mix  # directory of the FJSP instances
export LOG_DIR=train_test_dir/machine_breakdown/train_log/j20-m10-t30_mix-w80-s30-t60-st3-mid   # directory to save the training logs
export TEST_STATS_DIR=train_test_dir/machine_breakdown/test_stats/j20-m10-t30_mix-w80-s30-t60-st3-mid   # directory to save the test statistics
export MODEL_DIR=train_test_dir/machine_breakdown/model/j20-m10-t30_mix-w80-s30-t60-st3-mid   # directory to save the trained models

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

### breakdown parameters (you should modify this based on the setting you use to generate the breakdown data)
export BREAKDOWN_DATA_DIR=train_data_dir/machine_breakdown/breakdown_data/j20-m10-t30_mix-mid
export NUM_MACHINE_BREAKDOWN_P=0.35
export FIRST_BREAKDOWN_BUFFER_LB=50
export FIRST_BREAKDOWN_BUFFER_UB=150
export MACHINE_BREAKDOWN_DURATION_LB=100
export MACHINE_BREAKDOWN_DURATION_UB=100
export BREAKDOWN_BUFFER_LB=175
export BREAKDOWN_BUFFER_UB=300


python -u flexible_jss_learn.py --train_data_dir $TRAIN_DATA_DIR --jss_data_dir $JSS_DATA_DIR --log_dir $LOG_DIR \
        --test_stats_dir $TEST_STATS_DIR --model_dir $MODEL_DIR --data_index_start $DATA_INDEX_START --data_index_end $DATA_INDEX_END \
        --val_index_start $VAL_INDEX_START --val_index_end $VAL_INDEX_END --num_epochs $EPOCH --model_name $MODEL_NAME \
        --window $WINDOW --step $STEP --time_limit $TIME_LIMIT --stop_search_time $STOP_SEARCH_TIME --pos_weight $POS_WEIGHT \
        --machine_breakdown --breakdown_data_dir $BREAKDOWN_DATA_DIR \
        --num_machine_breakdown_p $NUM_MACHINE_BREAKDOWN_P \
        --first_breakdown_buffer_lb $FIRST_BREAKDOWN_BUFFER_LB \
        --first_breakdown_buffer_ub $FIRST_BREAKDOWN_BUFFER_UB \
        --machine_breakdown_duration_lb $MACHINE_BREAKDOWN_DURATION_LB \
        --machine_breakdown_duration_ub $MACHINE_BREAKDOWN_DURATION_UB \
        --breakdown_buffer_lb $BREAKDOWN_BUFFER_LB \
        --breakdown_buffer_ub $BREAKDOWN_BUFFER_UB
```
</details>

<details>
  <summary>4. Testing (Rollout)</summary>

After the model is trained, we can use the following script to test the model by solving FJSP instances using our proposed L-RHO algorithm. You should make sure the relevant directories below are specified correctly. For example, `JSS_DATA_DIR` should contain the previously generated FJSP instances, `BREAKDOWN_DATA_DIR` should contain the corresponding machine breakdown events, and `MODEL_DIR` should contain the previously trained model. We can specify different test FSJP distribution by using a different `JSS_DATA_DIR` (along with the corresponding `BREAKDOWN_DATA_DIR`), while keeping the other parameters the same. We should specify the index of the testing FJSP instances by setting `TEST_INDEX_START` and `TEST_INDEX_END` below properly, and add `--test` flag when running the python script. 


```script
### the test data indices; note that we use a slurm script to test FJSP instances with IDX from 500 to 600 in parallel
export TEST_INDEX_START=500   # start index of the _test_ data
export TEST_INDEX_END=501   # end index of the _test_ data

export DATA_INDEX_START=0   # start index of the training data
export DATA_INDEX_END=450   # end index of the training data

### relevant directories: note that you can specify a different JSS_DATA_DIR (and the corresponding BREAKDOWN_DATA_DIR below) to test on different instances
export JSS_DATA_DIR=train_data_dir/instance/j20-m10-t30_mix   # directory of the FJSP instances
export TRAIN_DATA_DIR=train_data_dir/machine_breakdown/train_data/j20-m10-t30_mix-w80-s30-t60-st3-mid   # directory where training data are saved
export LOG_DIR=train_test_dir/machine_breakdown/train_log/j20-m10-t30_mix-w80-s30-t60-st3-mid   # directory to save the training logs
export TEST_STATS_DIR=train_test_dir/machine_breakdown/test_stats/j20-m10-t30_mix-w80-s30-t60-st3-mid   # directory to save the test statistics
export MODEL_DIR=train_test_dir/machine_breakdown/model/j20-m10-t30_mix-w80-s30-t60-st3-mid   # directory to save the trained models

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
export BREAKDOWN_DATA_DIR=train_data_dir/machine_breakdown/breakdown_data/j20-m10-t30_mix-mid
export NUM_MACHINE_BREAKDOWN_P=0.35
export FIRST_BREAKDOWN_BUFFER_LB=50
export FIRST_BREAKDOWN_BUFFER_UB=150
export MACHINE_BREAKDOWN_DURATION_LB=100
export MACHINE_BREAKDOWN_DURATION_UB=100
export BREAKDOWN_BUFFER_LB=175
export BREAKDOWN_BUFFER_UB=300

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
```
</details>

<details>
  <summary>5. Print Test Stats</summary>
We can use the `print_results.py` script to print the test results, with the test stats directory `TEST_STATS_DIR` and the data name `DATA_NAME` (name of the saved stats file, excluding the data idx suffix) properly specified.
 
```script
export INDEX_START=500   # start index of the _test_ data
export INDEX_END=600  # end index of the _test_ data
export TEST_STATS_DIR=train_test_dir/machine_breakdown/test_stats/j20-m10-t30_mix-w80-s30-t60-st3-mid   # directory of the saved test statistics
export DATA_NAME=stats_e120th0.5   # name of the saved stats file, excluding the data idx suffix

python -u print_results.py --index_start $INDEX_START --index_end $INDEX_END --stats_dir $TEST_STATS_DIR --data_name $DATA_NAME 
```
</details>

<details>
  <summary>6. Troubleshoot</summary>
Please read the troubleshoot section for the Makespan variant [above](#makespan) to see our suggestions of ways to troubleshoot the results. 
</details>

---
### Delay Variants
Please go to the `./delay` directory by `cd ./delay`. The example scripts can be found in `./delay/scripts/j25-m25-t24`. We provide detailed instructions below:

#### Start Delay

<details>
  <summary>1. FJSP Instance Generation</summary>
We can generate FJSP instances with `N_MACHINES` number of machines, `N_JOBS` number of jobs, `N_OPS_PER_JOB` number of operations per job as follows. The FJSP objective is specified by OPTIM_OPION [start_delay | start_and_end_delay]. In our experiments, we generate 600 instances for each FJSP setting, with index 0-9 for RHO parameter search, 10-459 for training, 460-479 for validation, and 500-599 for testing.

```script
export N_MACHINES=25  # the number of machines
export N_JOBS=25  # the number of jobs
export N_OPS_PER_JOB=24  # the number of operations per job
export OPTIM_OPTION='start_delay'  # the objective
export DATA_IDX_START=0
export DATA_IDX_END=600  # the number of instances to generate

python -u gen_instance.py --n_machines $N_MACHINES --n_jobs $N_JOBS \
        --n_tasks_per_job $N_OPS_PER_JOB --optim_option $OPTIM_OPTION \
        --data_index_start $DATA_IDX_START --data_index_end $DATA_IDX_END
```
</details>

<details>
  <summary>2. RHO Parameter Search </summary>
As detailed in Appendix A.5.4, we perform a grid search to find the best parameter (Planning Horizon Size $H$, Execution Step Size $S$, Subproblem Time Limit $T$, Subproblem Early Termination Time $T_{es}$) for each method (Default RHO, Oracle, First and Random baselines). The following script evaluates the performance on a specific combination of the parameters, specified by `PARAM_IDX` (39 parameters in total). The performance statistics (objective and solve time) are saved in the `$PARAM_STATS_DIR` directory.

```script
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

python -u flexible_jss_parameter_search.py --n_machines $N_MACHINES --n_jobs $N_JOBS \
        --n_tasks_per_job $N_OPS_PER_JOB --optim_option $OPTIM_OPTION --param_idx $PARAM_IDX \
        --param_dir  $PARAM_DIR --param_stats_dir $PARAM_STATS_DIR --jss_data_dir $FJSP_DATA_DIR \
        --data_index_start $DATA_IDX_START --data_index_end $DATA_IDX_END
```

After which, we can find the best combination of RHO parameters for each method using the following grid search procedure, where the best combination will be saved to `$PARAM_STATS_DIR/best_params.pkl`.

```script
export N_MACHINES=25  # the number of machines
export N_JOBS=25  # the number of jobs
export N_OPS_PER_JOB=24  # the number of operations per job
export OPTIM_OPTION='start_delay'  # the objective

# the index of the parameter to evaluate the performance
export DATA_IDX_START=0  # the starting index of the FJSP instances to perform parameter search
export DATA_IDX_END=10  # the ending index of the FJSP instances to perform parameter search

# The directory to load the set of parameters to perform grid search on
export PARAM_DIR='train_test_dir/param_search/params/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct08-madapt-task-machine-cf3'
# The directory to load the parameter grid search statistics
export PARAM_STATS_DIR='train_test_dir/param_search/stats/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct08-madapt-task-machine-cf3'
# FJSP instance directory
export FJSP_DATA_DIR='train_data_dir/instance/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3'

python -u find_best_rho_param.py --n_machines $N_MACHINES --n_jobs $N_JOBS \
        --n_tasks_per_job $N_OPS_PER_JOB --optim_option $OPTIM_OPTION \
        --param_dir  $PARAM_DIR --param_stats_dir $PARAM_STATS_DIR --jss_data_dir $FJSP_DATA_DIR \
        --data_index_start $DATA_IDX_START --data_index_end $DATA_IDX_END
```
</details>

<details>
  <summary>3. Training Data Collection (Lookahead Oracle)</summary>
We can use the following script to generate training data from the assignment-based look-ahead oracle described in the main paper Sec. 4. `$DATA_IDX` specifies the index of the FJSP instance that we want to generate the training data,  (`WINDOW`, `STEP`, `TIME_LIMIT`, `EARLY_STOP_TIME`) specifies Oracle's best RHO parameter combination for the FJSP setting, obtained from the previous parameter search procedure. The training data is saved to `$TRAIN_DATA_DIR`. 

```script
# index of the FJSP instance to generate the training data (input and oracle labels)
export DATA_IDX=0   # [INSERT DATA INDEX HERE], we use a slurm script to generate data for DATA_IDX from 0 to 500 in parallel

# FJSP setting 
export N_MACHINES=25
export N_JOBS=25
export N_TASKS_PER_JOB=24
export OPTIM_OPTION='start_delay'

# Oracle's best parameter combination for the FJSP setting [MODIFY THIS BASED ON THE RESULT OF STEP 2]
export WINDOW=80
export STEP=25
export TIME_LIMIT=60
export EARLY_STOP_TIME=3

# directory to save the training data
export TRAIN_DATA_DIR='train_data_dir/train_data/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-w80-s25-t60-st3'
# FJSP instance directory
export FJSP_DATA_DIR='train_data_dir/instance/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3'
# directory to save data collection statistics
export stats_dir='train_data_dir/stats/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-w80-s25-t60-st3'


python -u flexible_jss_main.py --data_idx $DATA_IDX  \
        --n_machines $N_MACHINES --n_jobs $N_JOBS \
        --n_tasks_per_job $N_TASKS_PER_JOB --optim_option $OPTIM_OPTION \
        --window $WINDOW --step $STEP --time_limit $TIME_LIMIT --stop_search_time $EARLY_STOP_TIME \
        --train_data_dir $TRAIN_DATA_DIR  --jss_data_dir $FJSP_DATA_DIR \
        --stats_dir $stats_dir --load_data
```
</details>


<details>
  <summary>4. Model Training</summary>
We can use the following script to train the model.

```script
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
export OPTIM_OPTION='start_delay'

# Oracle's best parameter combination for the FJSP setting
export WINDOW=80
export STEP=25
export TIME_LIMIT=60
export EARLY_STOP_TIME=3

# directory that contains the training data 
export TRAIN_DATA_DIR=train_data_dir/train_data/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-w80-s25-t60-st3
# directory that contains the FJSP instances
export FJSP_DATA_DIR=train_data_dir/instance/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3
# directory to save the training logs
export LOG_DIR=train_test_dir/train_log/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-w80-s25-t60-st3
# directory to save the test statistics
export TEST_STATS_DIR=train_test_dir/test_stats/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3
# directory to save the model
export MODEL_DIR=train_test_dir/model/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-w80-s25-t60-st3


python flexible_jss_learn.py \
        --data_index_start $DATA_INDEX_START --data_index_end $DATA_INDEX_END \
        --val_index_start $VAL_INDEX_START --val_index_end $VAL_INDEX_END --num_epochs $EPOCHS --model_name $MODEL_NAME --pos_weight $POS_WEIGHT \
        --train_data_dir $TRAIN_DATA_DIR --jss_data_dir $FJSP_DATA_DIR \
        --log_dir $LOG_DIR --test_stats_dir $TEST_STATS_DIR --model_dir $MODEL_DIR \
        --n_machines $N_MACHINES --n_jobs $N_JOBS \
        --n_tasks_per_job $N_TASKS_PER_JOB --optim_option $OPTIM_OPTION \
        --window $WINDOW --step $STEP --time_limit $TIME_LIMIT --stop_search_time $EARLY_STOP_TIME
```
</details>


<details>
  <summary>5. Testing (Rollout)</summary>
We can use the following script to evaluate the test performances for Default RHO, First and Random RHO baselines, our learning method L-RHO (following the inference procedure described in the main paper Sec. 4), and solving the full FJSP without decomposition. In particular, we need to specify the test data indices through `$TEST_INDEX_START` and `$TEST_INDEX_END`, the FJSP instance directory `$FJSP_DATA_DIR`, the pretrained model directory `$MODEL_DIR`, and the path to the best RHO parameter file `$BEST_PARAM_FILE`. The test performance statistics (objective and solve time) will be saved to `$TEST_STATS_DIR`.

```
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
export OPTIM_OPTION='start_delay'

# Oracle's best parameter combination for the FJSP setting 
# (only used if BEST_PARAM_FILE below does not exist)
export WINDOW=80
export STEP=25
export TIME_LIMIT=60
export EARLY_STOP_TIME=3

# directory that contains the training data 
export TRAIN_DATA_DIR=train_data_dir/train_data/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-w80-s25-t60-st3
# directory that contains the FJSP instances
export FJSP_DATA_DIR=train_data_dir/instance/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3
export LOG_DIR=train_test_dir/train_log/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-w80-s25-t60-st3

# directory to save the test performance statistics 
export TEST_STATS_DIR=train_test_dir/test_stats/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3
# directory that contains the pretrained model
export MODEL_DIR=train_test_dir/model/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3-w80-s25-t60-st3
# path to the best RHO parameter file (obtained from the above parameter search procedure)
export BEST_PARAM_FILE='train_test_dir/param_search/params/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3/best_params.pkl'

python -u flexible_jss_learn.py --data_index_start $DATA_INDEX_START --data_index_end $DATA_INDEX_END \
        --val_index_start $TEST_INDEX_START --val_index_end $TEST_INDEX_END --model_name $MODEL_NAME \
        --pos_weight $POS_WEIGHT --test --load_model_epoch $LOAD_MODEL_EPOCH  \
        --n_machines $N_MACHINES --n_jobs $N_JOBS \
        --n_tasks_per_job $N_TASKS_PER_JOB --optim_option $OPTIM_OPTION \
        --window $WINDOW --step $STEP --time_limit $TIME_LIMIT --stop_search_time $EARLY_STOP_TIME \
        --train_data_dir $TRAIN_DATA_DIR --jss_data_dir $FJSP_DATA_DIR \
        --log_dir $LOG_DIR --test_stats_dir $TEST_STATS_DIR --model_dir $MODEL_DIR \
        --load_best_params --best_params_file $BEST_PARAM_FILE
```
</details>

<details>
  <summary>6. Print Test Stats</summary>
We can use the `print_results.py` script to print the test results, with the test stats directory `TEST_STATS_DIR` and the data name `DATA_NAME` (name of the saved stats file, excluding the data idx suffix) properly specified.

```script
export INDEX_START=500   # start index of the _test_ data
export INDEX_END=600  # end index of the _test_ data
export TEST_STATS_DIR=train_test_dir/test_stats/m25-j25-t24-p1-lo5-hi20-s0-h15-p-ct0.8-madapt-task-machine-cf3/model_pw0.5   # directory of the saved test statistics
export DATA_NAME=stats_e120th0.5   # name of the saved stats file, excluding the data idx suffix

python -u print_results.py --index_start $INDEX_START --index_end $INDEX_END --stats_dir $TEST_STATS_DIR --data_name $DATA_NAME 
```
</details>

#### Start and End Delay
The scripts for start and end delay should be very similar to that for start delay. The main differences are that (1) we set `optim_option` to `start_and_end_delay` instead of `start_delay`, and (2) we specify a few more parameters related to end delay (`l_low_end`, `l_high_end`). We provide example scripts for this setting in `./delay/scripts/j25-m25-t24/start+end_delay`.


<details>
  <summary>1. FJSP Instance Generation </summary>

Please see an example script at `./delay/scripts/j25-m25-t24/start+end_delay/1_gen_instance.sh`.
</details>

<details>
  <summary>2. Training Data Collection (Lookahead Oracle) </summary>

Please see an example script at `./delay/scripts/j25-m25-t24/start+end_delay/2_collect_data.sh`. Note that for the best performance, we can use similar scripts as in the start delay variant to grid search the best set of RHO parameters, and set the `WINDOW`, `STEP`, `TIME_LIMIT` and `EARLY_STOP_TIME` arguments in the script accordingly.
</details>

<details>
  <summary>3. Model Training </summary>

Please see an example script at `./delay/scripts/j25-m25-t24/start+end_delay/3_training.sh`.
</details>


<details>
  <summary>4. Testing (Rollout) </summary>

Please see an example script at `./delay/scripts/j25-m25-t24/start+end_delay/4_testing.sh`.
</details>


<details>
  <summary>5. Print Test Stats</summary>

Please see an example script at `./delay/scripts/j25-m25-t24/start+end_delay/5_print_results.sh`.
</details>

#### Start and End Delay + Obs. Noise
The scripts for start and end delay + obs. noise should be very similar to that for start and end delay. The main difference is that we specify the `perturb_p` parameter related to obs. noise and add the `--perturb_data` flag. We can reuse the same FJSP instances as in the start and end delay setting, and the observation noise are generated on the fly (online). We provide example scripts for this setting in `./delay/scripts/j25-m25-t24/start+end_delay+obs_noise`.

<details>
  <summary>1. Training Data Collection (Lookahead Oracle) </summary>

Please see an example script at `./delay/scripts/j25-m25-t24/start+end_delay+obs_noise/1_collect_data.sh`. Note that for the best performance, we can use similar scripts as in the start delay variant to grid search the best set of RHO parameters, and set the `WINDOW`, `STEP`, `TIME_LIMIT` and `EARLY_STOP_TIME` arguments in the script accordingly. 
</details>

<details>
  <summary>2. Model Training </summary>

Please see an example script at `./delay/scripts/j25-m25-t24/start+end_delay+obs_noise/2_training.sh`.
</details>


<details>
  <summary>3. Testing (Rollout) </summary>

Please see an example script at `./delay/scripts/j25-m25-t24/start+end_delay+obs_noise/3_testing.sh`.
</details>

<details>
  <summary>4. Print Test Stats</summary>

Please see an example script at `./delay/scripts/j25-m25-t24/start+end_delay+obs_noise/4_print_results.sh`.
</details>