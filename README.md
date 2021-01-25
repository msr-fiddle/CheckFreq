# CheckFreq: Frequent, Fine-Grained DNN Checkpointing

This repository contains the source code implementation of the FAST'21 paper "CheckFreq: Frequent, Fine-Grained DNN Checkpointing". This work was done as part of  Microsoft Research's [Project Fiddle](https://www.microsoft.com/en-us/research/project/fiddle/). This source code is available under the [MIT License](LICENSE.txt).

CheckFreq is an automatic, fine-grained checkpointing framework that 

  1. Algorithmically determines the checkpointing frequency at the granularity of iterations using systematic online profiling
  2. Dynamically tunes checkpointing frequency at runtime to bound the checkpointing overhead using adaptive rate tuning
  3. Maintains the training data invariant of using each item in the dataset exactly once per epoch by checkpointing data loader state using a light-weight resumable iterator
  4. Carefully pipelines checkpointing with computation to reduce the checkpoint cost by introducing two-phase checkpointing. 
  

[[pdf]](https://www.microsoft.com/en-us/research/publication/checkfreq-frequent-fine-grained-dnn-checkpointing/)      [[slides]]()


## Setup

CheckFreq is implemented as a extendible module for PyTorch. To run CheckFreq, you will need a NVIDIA GPU with CUDA 10.0, nvidia-docker2, and Python 3. 
We used the prebuilt NVIDIA docker container [nvcr.io/nvidia/pytorch:19.05-py3](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch/tags) container as the base image, which can be downloaded using,

    docker pull nvcr.io/nvidia/pytorch:19.05-py3
    
CheckFreq's resumable data iterator is built as an extension to the state-of-the-art data loader [CoorDL](https://github.com/msr-fiddle/CoorDL), built on top of NVIDIA [DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html). To build a docker container based off the above base image with CheckFreq's resumable iterator, apply the [patch](dl_patch/esumable_iterator.patch) to the master branch of [CoorDL](https://github.com/msr-fiddle/CoorDL) repo using

    git apply resumable_iterator.patch
    
    
 Then build the docker image with CheckFreq's iterator by following the instructions in its [repo](https://github.com/msr-fiddle/CoorDL).
 
 The final docker image is tagged `nvidia/dali:py36_cu10.run` and can be run using
    
        nvidia-docker run --ipc=host --mount src=/,target=/datadrive/,type=bind -it --rm --network=host --privileged nvidia/dali:py36_cu10.run
 
    

## Using CheckFreq

CheckFreq can be used in the training script with a few changes.

  1. Import CheckFreq manager, and iterator in the training script
  
          from cf_checkpoint import CFCheckpoint
          from cf_manager import CFManager, CFMode
          from cf_iterator import  
          
  2. Initialize a checkpoint wrapper that tracks state to be checkpointed. 
  
          chk = CFCheckpoint(model=model, optimizer=optimizer)
          
   We assume that each of these parameters to be tracked exposes a `state_dict` that is snapshotted during the checkpoint operation.
   Then create a CheckFreq manager by specifying the frequency estimation mode(MANUAL/AUTO), checkpoint wrapper, and the path to store final checkpoints.
   
          cf_manager = CFManager(chk_prefix, chk, mode=CFMode.AUTO)
          
  3. Pass in the epoch and batch ID to resume the dataloader (got from the previous checkpoint if training is resumed or 0 if starting)
  
          self.input = ops.FileReader(..., resume_index=resume_index, resume_epoch=resume_epoch, cf_det=cf_det)
          
  4. Wrap the DALIClassificationIterator by `CFIterator` and optionally pass in arguments for adaptive rate tuning (`dynamic=True`), a checkpointing frequency is MANUAL mode is set (`chk_freq=N`)
  
          train_loader = DALIClassificationIterator(...)
          train_loader = CFIterator(train_loader, ...)
          
  4. On the main process (local rank 0), use the wrapper for optimizer.step 
  
          cf_manager.weight_update()
          

A complete working example with changes to integrate CheckFreq in the training script for image classification is [here](models/image_classification/pytorch-imagenet-cf.py)


## Example

We demonstrate an example of running CheckFreq for image classification using popular models like ResNets, VGGs, and Inception using the ImageNet ILSVC 2012 dataset. 

The source code for the training script with CheckFreq integration is [here](models/image_classification/pytorch-imagenet-cf.py)

To train VGG16 across 8 GPUs on a server for 2 epochs with CheckFreq, use the following command :

      python -m torch.distributed.launch --nproc_per_node=8 models/image_classification/pytorch-imagenet-cf.py --dali -a resnet18 -b 256 --workers 3 --epochs 2  --deterministic --noeval --barrier --checkfreq --chk-prefix ./chk/ --cf_iterator --data <imagenet_data_directory> > stdout.out 2>&1
      
To run the same without CheckFreq, using epoch boundary checkpointing, use:

      python -m torch.distributed.launch --nproc_per_node=8 models/image_classification/pytorch-imagenet-cf.py --dali -a resnet18 -b 256  --workers 3 --epochs 2  --deterministic --noeval --barrier --checkfreq --chk-freq 0 --chk_mode_baseline --chk-prefix ./chk/ --cf_iterator --data $DATA_DIR > stdout.out 2>&1

A complete script to train different models with and without CheckFreq is [here](https://github.com/msr-fiddle/CheckFreq/blob/main/scripts/run_all_256.sh). You can run it using:
    
       ./run_all_256.sh <data-dir> <out-dir> <data_threads_per_GPU>
       

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/).

## License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT license](LICENSE.txt).
