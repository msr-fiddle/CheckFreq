#!/bin/bash


if [ "$#" -ne 3 ]; then
	echo "Usage : ./run_img.sh <data-dir> <out-dir> <worker>"
	exit 1
fi

apt-get install jq
DATA_DIR=$1
OUT_DIR=$2
WORKER=$3
SRC="models/image_classification/"
SCRIPTS="scripts/"

mkdir -p $OUT_DIR


gpu=0
num_gpu=8

echo " Data dir is $DATA_DIR"
echo " Out dir is $OUT_DIR"

resnext="resnext101"
densenet="densenet121"

for arch in 'vgg16' ; do
#for arch in 'resnet18' ; do
#for arch in 'resnet50' 'resnet18' 'inception_v3' 'resnext101' 'densenet121' 'vgg16'; do
	for workers in $WORKER; do
		for batch in 256; do

#: <<'END'
			if [ "$arch" = "$resnext" ]; then
				batch=128
			elif [ "$arch" = "$densenet" ]; then
				batch=128
			fi

      # RUN 1 : CheckFreq
			result_dir="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}_dali_fp32_cf"
			echo "result dir is $result_dir" 
			mkdir -p $result_dir
			echo "Now running $arch for $workers workers and $batch batch"
			mpstat -P ALL 1 > cpu_util.out 2>&1 &
			./$SCRIPTS/free.sh &
			#./$SCRIPTS/gpulog.sh &
			dstat -cdnmgyr --output all-utils.csv 2>&1 & 
			python -m torch.distributed.launch --nproc_per_node=$num_gpu $SRC/pytorch-imagenet-cf.py --dali -a $arch -b $batch --workers $workers --epochs 2  --deterministic --noeval --barrier --checkfreq --chk-prefix ./chk/ --cf_iterator --data $DATA_DIR > stdout.out 2>&1
			sync
			echo "RAN $arch for $workers workers, $batch batch with DDP" >> stdout.out
			pkill -f mpstat
			pkill -f dstat
			pkill -f free
			pkill -f gpulog
			pkill -f nvidia-smi
			pkill -f pytorch-imagenet
			sleep 2
			mv *.out  $result_dir/
			mv *.log $result_dir/
			mv *.csv $result_dir/

#exit
#: <<'END'
#END

      # RUN 2 : Epoch boundary
			result_dir="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}_dali_fp32_epoch_chk"
			echo "result dir is $result_dir" 
			mkdir -p $result_dir
			echo "Now running $arch for $workers workers and $batch batch"
			mpstat -P ALL 1 > cpu_util.out 2>&1 &
			./$SCRIPTS/free.sh &
			#./$SCRIPTS/gpulog.sh &
			dstat -cdnmgyr --output all-utils.csv 2>&1 & 
			python -m torch.distributed.launch --nproc_per_node=$num_gpu $SRC/pytorch-imagenet-cf.py --dali -a $arch -b $batch --workers $workers --epochs 1  --deterministic --noeval --barrier --chk-freq 0 --chk_mode_baseline --checkfreq --chk-prefix ./chk/ --cf_iterator --data $DATA_DIR > stdout.out 2>&1

			sync
			echo "RAN $arch for $workers workers, $batch batch with DDP" >> stdout.out
			pkill -f mpstat
			pkill -f dstat
			pkill -f free
			pkill -f gpulog
			pkill -f nvidia-smi
			pkill -f pytorch-imagenet
			sleep 2
			mv *.out  $result_dir/
			mv *.log $result_dir/
			mv *.csv $result_dir/

#exit
#END

      # RUN 3 : Synchronous at chosen frequency
			result_dir="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}_dali_fp32_iter_chk_baseline_persist"
			echo "result dir is $result_dir" 
			mkdir -p $result_dir
			echo "Now running $arch for $workers workers and $batch batch"

			cache_file=".cache_${arch}_${batch}"
			CHK=$(jq '.chk_freq' $cache_file)
			echo "Setting CHK freq = $CHK"
			mpstat -P ALL 1 > cpu_util.out 2>&1 &
			./$SCRIPTS/free.sh &
			#./$SCRIPTS/gpulog.sh &
			dstat -cdnmgyr --output all-utils.csv 2>&1 & 
			python -m torch.distributed.launch --nproc_per_node=$num_gpu $SRC/pytorch-imagenet-cf.py --dali -a $arch -b $batch --workers $workers --epochs 1  --deterministic --noeval --barrier --chk-freq $CHK --chk_mode_baseline --persist --checkfreq --chk-prefix ./chk/ --cf_iterator --data $DATA_DIR > stdout.out 2>&1

			sync
			echo "RAN $arch for $workers workers, $batch batch with DDP" >> stdout.out
			pkill -f mpstat
			pkill -f dstat
			pkill -f free
			pkill -f gpulog
			pkill -f nvidia-smi
			pkill -f pytorch-imagenet
			sleep 2
			mv *.out  $result_dir/
			mv *.log $result_dir/
			mv *.csv $result_dir/





		done
	done
done
