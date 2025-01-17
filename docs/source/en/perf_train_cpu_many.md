<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Distributed CPUs

CPUs are commonly available and can be a cost-effective option for training when GPUs are unavailable. When training large models or if a single CPU is too slow, distributed training with CPUs can help speed up training.

This guide demonstrates how to perform distributed training with multiple CPUs using a [DistributedDataParallel (DDP)](./perf_train_gpu_many#distributeddataparallel) strategy on bare metal with [`Trainer`] and a Kubernetes cluster. All examples shown in this guide depend on the [Intel oneAPI HPC Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit.html).

There are two toolkits you'll need from Intel oneAPI.

1. [oneCCL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneccl.html) is a toolkit that includes efficient implementations of collectives commonly used in deep learning such as all-gather, all-reduce, and reduce-scatter. To install from a prebuilt wheel, make sure you always use the latest release. Refer to the table [here](https://github.com/intel/torch-ccl#install-prebuilt-wheel) to check if a version of oneCCL is supported for a Python and PyTorch version.

```bash
# installs oneCCL for PyTorch 2.4.0
pip install oneccl_bind_pt==2.4.0 -f https://developer.intel.com/ipex-whl-stable-cpu
```

> [!TIP]
> Refer to the oneCCL [installation](https://github.com/intel/torch-ccl#installation) for more details.

1. [MPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html) is a message-passing interface for communications between hardware and networks. The oneCCL toolkit is installed along with MPI, but you need to source the environment as shown below before using it.

```bash
oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
```

Lastly, install the [Intex Extension for PyTorch (IPEX)](https://intel.github.io/intel-extension-for-pytorch/index.html) which enables additional performance optimizations for Intel hardware such as weight sharing and better thread runtime control.

```bash
pip install intel_extension_for_pytorch==<version_name> -f https://developer.intel.com/ipex-whl-stable-cpu
```

> [!TIP]
> Refer to the IPEX [installation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation) for more details.

## Trainer

[`Trainer`] supports distributed training with CPUs with the oneCCL backend. Add the `--ddp_backend ccl` parameter in the command arguments to enable it.

<hfoptions id="distrib-cpu">
<hfoption id="single node">

The example below demonstrates the [run_qa.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) script. It enables training with two processes on one Xeon CPU, with one process running per socket.

> [!TIP]
> Tune the variable `OMP_NUM_THREADS/CCL_WORKER_COUNT` for optimal performance.

```bash
export CCL_WORKER_COUNT=1
export MASTER_ADDR=127.0.0.1
mpirun -n 2 -genv OMP_NUM_THREADS=23 \
python3 run_qa.py \
 --model_name_or_path google-bert/bert-large-uncased \
 --dataset_name squad \
 --do_train \
 --do_eval \
 --per_device_train_batch_size 12  \
 --learning_rate 3e-5  \
 --num_train_epochs 2  \
 --max_seq_length 384 \
 --doc_stride 128  \
 --output_dir /tmp/debug_squad/ \
 --no_cuda \
 --ddp_backend ccl \
 --use_ipex
```

</hfoption>
<hfoption id="multiple nodes">

Scale the training script to four processes on two Xeon CPUs (`node0` and `node1`) by setting `-n 4` and `ppn 2`. The `ppn` parameter specifies the number of processes per node, with one process running per socket.

Assume `node0` is the main process and create a configuration file containing the IP addresses of each node (for example, hostfile) and pass the configuration file path as an argument.

```bash
cat hostfile
xxx.xxx.xxx.xxx #node0 ip
xxx.xxx.xxx.xxx #node1 ip
```

Run the script below on `node0` to enable DDP on `node0` and `node1` and train with bf16 auto mixed precision.

> [!TIP]
> Tune the variable `OMP_NUM_THREADS/CCL_WORKER_COUNT` for optimal performance.

```bash
export CCL_WORKER_COUNT=1
export MASTER_ADDR=xxx.xxx.xxx.xxx #node0 ip
mpirun -f hostfile -n 4 -ppn 2 \
 -genv OMP_NUM_THREADS=23 \
python3 run_qa.py \
 --model_name_or_path google-bert/bert-large-uncased \
 --dataset_name squad \
 --do_train \
 --do_eval \
 --per_device_train_batch_size 12  \
 --learning_rate 3e-5  \
 --num_train_epochs 2  \
 --max_seq_length 384 \
 --doc_stride 128  \
 --output_dir /tmp/debug_squad/ \
 --no_cuda \
 --ddp_backend ccl \
 --use_ipex \
 --bf16
```

</hfoption>
</hfoptions>

## Kubernetes

Distributed training with CPUs can also be deployed to a Kubernetes cluster with [PyTorchJob](https://www.kubeflow.org/docs/components/training/user-guides/pytorch/). Before you get started, you should perform the following setup steps.

1. Ensure you have access to a Kubernetes cluster with [Kubeflow](https://www.kubeflow.org/docs/started/installing-kubeflow/) installed.
1. Install and configure [kubectl](https://kubernetes.io/docs/tasks/tools) to interact with the cluster.
1. Set up a [PersistentVolumeClaim (PVC)](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) to store datasets and model files. There are multiple options to choose from, including a [StorageClass](https://kubernetes.io/docs/concepts/storage/storage-classes/) or a cloud storage bucket.
1. Set up a Docker container for the training script and all required dependencies such as PyTorch, Transformers, IPEX, oneCCL, and OpenSSH to facilitate communicattion between containers.

The example Dockerfile below uses a base image that supports distributed training with CPUs, and extracts Transformers to the `/workspace` directory to include the training scripts in the image. The image needs to be built and copied to the clusters nodes or pushed to a container registry prior to deployment.

```dockerfile
FROM intel/intel-optimized-pytorch:2.4.0-pip-multinode

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
    google-perftools \
    libomp-dev

WORKDIR /workspace

# Download and extract the transformers code
ARG HF_TRANSFORMERS_VER="4.46.0"
RUN pip install --no-cache-dir \
    transformers==${HF_TRANSFORMERS_VER} && \
    mkdir transformers && \
    curl -sSL --retry 5 https://github.com/huggingface/transformers/archive/refs/tags/v${HF_TRANSFORMERS_VER}.tar.gz | tar -C transformers --strip-components=1 -xzf -
```

### PyTorchJob

[PyTorchJob](https://www.kubeflow.org/docs/components/training/user-guides/pytorch/) is an extension of the Kubernetes API for running PyTorch training jobs on Kubernetes. It includes a yaml file that defines the training jobs parameters such as the name of the PyTorchJob, number of workers, types of resources for each worker, and more.

The volume mount parameter is a path to where the PVC is mounted in the container for each worker pod. The PVC is typically used to hold the dataset, checkpoint files, and the model after it has finished training.

The example yaml file below sets up four workers on the [run_qa.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) script. Adapt the yaml file based on your training script and number of nodes in your cluster.

The CPU resource limits and requests are defined in [CPU units](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-cpu). One CPU unit is equivalent to one physical CPU core or virtual core. The CPU units defined in the yaml file should be less than the amount of available CPU and memory capacity of a single machine in order to leave some resources for kubelet and the system. For a `Guaranteed` [quality of service](https://kubernetes.io/docs/tasks/configure-pod-container/quality-service-pod), set the same CPU and memory amounts for both the resource limits and requests.

```yaml
apiVersion: "kubeflow.org/v1"
kind: PyTorchJob
metadata:
  name: transformers-pytorchjob
spec:
  elasticPolicy:
    rdzvBackend: c10d
    minReplicas: 1
    maxReplicas: 4
    maxRestarts: 10
  pytorchReplicaSpecs:
    Worker:
      replicas: 4  # The number of worker pods
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: <image name>:<tag>  # Specify the docker image to use for the worker pods
              imagePullPolicy: IfNotPresent
              command: ["/bin/bash", "-c"]
              args:
                - >-
                  cd /workspace/transformers;
                  pip install -r /workspace/transformers/examples/pytorch/question-answering/requirements.txt;
                  source /usr/local/lib/python3.10/dist-packages/oneccl_bindings_for_pytorch/env/setvars.sh;
                  torchrun /workspace/transformers/examples/pytorch/question-answering/run_qa.py \
                    --model_name_or_path distilbert/distilbert-base-uncased \
                    --dataset_name squad \
                    --do_train \
                    --do_eval \
                    --per_device_train_batch_size 12 \
                    --learning_rate 3e-5 \
                    --num_train_epochs 2 \
                    --max_seq_length 384 \
                    --doc_stride 128 \
                    --output_dir /tmp/pvc-mount/output_$(date +%Y%m%d_%H%M%S) \
                    --no_cuda \
                    --ddp_backend ccl \
                    --bf16 \
                    --use_ipex;
              env:
              - name: LD_PRELOAD
                value: "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4.5.9:/usr/local/lib/libiomp5.so"
              - name: TRANSFORMERS_CACHE
                value: "/tmp/pvc-mount/transformers_cache"
              - name: HF_DATASETS_CACHE
                value: "/tmp/pvc-mount/hf_datasets_cache"
              - name: LOGLEVEL
                value: "INFO"
              - name: CCL_WORKER_COUNT
                value: "1"
              - name: OMP_NUM_THREADS  # Can be tuned for optimal performance
                value: "240"
              resources:
                limits:
                  cpu: 240  # Update the CPU and memory limit values based on your nodes
                  memory: 128Gi
                requests:
                  cpu: 240  # Update the CPU and memory request values based on your nodes
                  memory: 128Gi
              volumeMounts:
              - name: pvc-volume
                mountPath: /tmp/pvc-mount
              - mountPath: /dev/shm
                name: dshm
          restartPolicy: Never
          nodeSelector:  # Optionally use nodeSelector to match a certain node label for the worker pods
            node-type: gnr
          volumes:
          - name: pvc-volume
            persistentVolumeClaim:
              claimName: transformers-pvc
          - name: dshm
            emptyDir:
              medium: Memory
```

### Deploy

After you've setup the PyTorchJob yaml file with the appropriate settings for your cluster and training job, deploy it to the cluster with the command below.

```bash
export NAMESPACE=<specify your namespace>

kubectl create -f pytorchjob.yaml -n ${NAMESPACE}
```

List the pods in the namespace with `kubectl get pods -n ${NAMESPACE}`. At first, the status may be "Pending" but it should change to "Running" once the containers are pulled and created.

```bash
kubectl get pods -n ${NAMESPACE}

NAME                                                     READY   STATUS                  RESTARTS          AGE
...
transformers-pytorchjob-worker-0                         1/1     Running                 0                 7m37s
transformers-pytorchjob-worker-1                         1/1     Running                 0                 7m37s
transformers-pytorchjob-worker-2                         1/1     Running                 0                 7m37s
transformers-pytorchjob-worker-3                         1/1     Running                 0                 7m37s
...
```

Inspect the logs for each worker with the following command. Add `-f` to stream the logs.

```bash
kubectl logs transformers-pytorchjob-worker-0 -n ${NAMESPACE} -f
```

Once training is complete, the trained model can be copied from the PVC or storage location. Delete the PyTorchJob resource from the cluster with the command below.

```bash
kubectl delete -f pytorchjob.yaml -n ${NAMESPACE}
```
