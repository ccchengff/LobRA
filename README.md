# LobRA

Codes for our submission entitled *LobRA: Multi-tenant Fine-tuning over Heterogeneous Data*. We prepared scripts to run the workflow of profiling, deployment planning and training with homogeneous & heterogeneous parallel configurations on top of DL system `Hetu`. **PLEASE NOTE THAT** the appendix of our paper is available in this repository under the filename `Appendix_of_LobRA.pdf`.

### 1. Build & Compile Hetu

We use `cmake >= 3.24` to compile the `Hetu` system. Related third-party packages like `flash-attn`, `onednn`, `cutlass` have been prepared and will be compiled automatically. You may also configure the path to pre-built modules by modifing the configuration file `cmake/config_refactor.cmake`. Note that we use `nccl 2.20.5-1+cuda11.0` by default, you may replace with your own version in `third_party/nccl` and reconfigure `./cmake/external/nccl.cmake`. The compiling commands of our system are as follows

~~~bash
mkdir -p build && cd build
cmake ..
make -j 16
cd ..
source hetu_refactor.exp
cd LobRA
python setup.py build_ext --inplace # compile the python binding
~~~

Now you can import `hetu` in python by `import hetu`.

As for the `SCIP` library leveraged for MINLP and ILP problem solving, you can download the source code from [GitHub](https://github.com/scipopt/scip) and follow the installation instructions [here](https://github.com/scipopt/scip/blob/master/INSTALL.md) to compile it by yourself. We have provided `SCIPOptSuite-8.1.0` deb package in the `./third_party/scip` directory. You can install it by running the following commands

~~~bash
cd third_party/scip
dpkg -i SCIPOptSuite-8.1.0-Linux-ubuntu.deb
~~~

### 2. Dataset Preparation

In the `LobRA` paper, we use 12 datasets to evaluate the performance of our system. Note that our work is orthogonal to the chosen of datasets, and you may pick any dataset to run our scripts. As for datasets used in the `LobRA` paper, you can download them through the example script provided in `LobRA/scripts/`, which will download the `dolly` dataset. For other datasets, please refer to the provided links in the paper.

~~~bash
bash data_utils/create_dolly_dataset.sh
~~~

### 3. Profiling

We provide a set of scripts to profile the memory and time cost of different parallel configurations. The memory and time cost profiling results will be stored in `exp_result/profile/memory/` and `exp_result/profile/cost_model/`, respectively. Take Llama2-7B model, 6 tasks and 16 GPUs as an example, you can run the following commands to profile the memory and time cost of different parallel configurations.

~~~bash
# (under LobRA folder)
# memory profiling
# bash scripts/profile_memory.sh <MODEL_SIZE> <NUM_GPUS_LIMIT> <TRAIN_TASK_NUM> <TRAINER_CONFIG_PATH>
bash scripts/profile_memory.sh 7B 16 6 exp_task6
# time cost profiling
# bash scripts/cost_model_benchmark.sh <MODEL_SIZE> <TP_LIST> (means TP degrees to be profiled)
bash scripts/cost_model_benchmark.sh 7B 1,2,4,8
~~~


### 4. Deployment Planning

We provide a set of scripts to test the deployment planning via different approaches, including solving the MINLP problem with SCIP, and the configuration pruning method. Take Llama2-7B model, 6 tasks and 16 GPUs as an example, you can run the following commands for deployment planning.

~~~bash
# (under LobRA folder)
# get the optimal deployment plan by solving MINLP problem with SCIP library without configuration proposal
export CONFIGURATION_PROPOSAL=FALSE
# bash scripts/MINLP_static_planner.sh <MODEL_SIZE> <NUM_GPUS_LIMIT> <TRAIN_TASK_NUM> <TRAINER_CONFIG_PATH>
bash scripts/MINLP_static_planner.sh 7B 16 6 16 16384 exp_task6
# get the optimal deployment plan by solving MINLP problem with SCIP library with configuration proposal
export CONFIGURATION_PROPOSAL=TRUE
bash scripts/MINLP_static_planner.sh 7B 16 6 16 16384 exp_task6
# get the optimal deployment plan by configuration pruning
# bash scripts/pruning_static_planner.sh <MODEL_SIZE> <NUM_GPUS_LIMIT> <TRAIN_TASK_NUM> <TRAINER_CONFIG_PATH>
bash scripts/pruning_static_planner.sh 7B 16 6 16 16384 exp_task6
~~~

### 5. Fine-tuning with LobRA

Fine-tuning with homogeneous & heterogeneous parallel configurations can be launched via training script `scripts/llama_lora_multi_task.sh`. Take Llama2-7B model, 6 tasks and 16 GPUs as an example, you can run the following commands for fine-tuning.

~~~bash
# (under LobRA folder)
# bash scripts/llama_lora_multi_task.sh <MODEL_SIZE> <TRAIN_TASK_NUM> <CONTEXT_LENGTH_LIST> <DP_LIST> <TP_LIST> <PP_LIST> <BUCKET_NUM> <DP_BUCKET> <MAX_SEQ_LENGTH> <TRAINER_CONFIG_PATH>
# fine-tuning with homoegeneous parallel configurations
export DP_BUCKET=FALSE # whether to use dynamic bucketing
bash scripts/llama_lora_multi_task.sh 7B 6 16384 2 8 1 16 16384 exp_task6
# fine-tuning with heterogeneous parallel configurations (e.g., <1,1> x 6, <2,1> x 1, <8,1> x 1)
export DP_BUCKET=TRUE
bash scripts/llama_lora_multi_task.sh 7B 6 2048,4096,16384 6,1,1 1,2,8 1,1,1 16 16384 exp_task6
~~~

### 6. Experiments in the Paper

You can reproduce the end-to-end, ablation study and scalability experiments in our paper by running the following scripts.

~~~bash
# (under LobRA folder)
bash experiments/expr_7b.sh
bash experiments/expr_32b.sh
bash experiments/expr_70b.sh
~~~

Note that you need to profile the memory and time cost of different parallel configurations before running the experiments. The profiling results will be stored in `exp_result/profile/memory/` and `exp_result/profile/cost_model/`, respectively. We provide our profiling results of Llama2-7B on 16 A100-40GB GPUs in both of the directories for your reference.