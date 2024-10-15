#!/bin/bash

export MINI_APP_DeepDriveMD_DIR="${PWD}/.."
export RADICAL_REPORT=TRUE
export RADICAL_LOG_LVL=DEBUG
exp_dir=/gpfs/alpine2/scratch/dewiy/gen010/soma_mini_app_out/

if [ -d ${exp_dir} ]
then
	echo "Error! Directory ${exp_dir} exists"
	exit -1
fi

mkdir -p ${exp_dir}/model
mkdir -p ${exp_dir}/data

num_phase=1
for((i=0; i<num_phase; i++))
do
	mkdir -p ${exp_dir}/data/phase${i}
done

python ../rct-scripts/scale-soma-ddmd-F-summit.py	\
	--num_phases		${num_phase}		\
	--mat_size 		10000		\
	--data_root_dir		"${exp_dir}/data"	\
	--num_step		60000		\
	--num_epochs_train	150		\
	--model_dir		"${exp_dir}/model"	\
	--num_sample		500		\
	--num_mult_train	4000		\
	--dense_dim_in		12544		\
	--dense_dim_out		128		\
	--preprocess_time_train	30		\
	--preprocess_time_agent	5		\
	--num_epochs_agent	100		\
	--num_mult_agent	1000		\
	--num_mult_outlier	100		\
	--project_id		gen010		\
	--queue			"debug"		\
	--num_sim		12		\
	--num_nodes	        537 		\
	--num_pipelines		512		\
	--io_json_file		"io_size-summit-copy.json"
