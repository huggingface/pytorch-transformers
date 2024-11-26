<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Efficient Training on Multiple CPUs

When training on a single CPU is too slow, we can use multiple CPUs. This guide focuses on PyTorch-based DDP enabling
distributed CPU training efficiently on [bare metal](#usage-in-trainer) and [Kubernetes](#usage-with-kubernetes).

## Intel® oneCCL Bindings for PyTorch

[Intel® oneCCL](https://github.com/oneapi-src/oneCCL) (collective communications library) is a library for efficient distributed deep learning training implementing such collectives like allreduce, allgather, alltoall. For more information on oneCCL, please refer to the [oneCCL documentation](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html) and [oneCCL specification](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html).

Module `oneccl_bindings_for_pytorch` (`torch_ccl` before version 1.12)  implements PyTorch C10D ProcessGroup API and can be dynamically loaded as external ProcessGroup and only works on Linux platform now

Check more detailed information for [oneccl_bind_pt](https://github.com/intel/torch-ccl).

### Intel® oneCCL Bindings for PyTorch installation

Wheel files are available for the following Python versions:

| Extension Version | Python 3.7 | Python 3.8 | Python 3.9 | Python 3.10 | Python 3.11 |
| :---------------: | :--------: | :--------: | :--------: | :---------: | :---------: |
| 2.5.0             |            | √          | √          | √           | √           |
| 2.4.0             |            | √          | √          | √           | √           |
| 2.3.0             |            | √          | √          | √           | √           |
| 2.2.0             |            | √          | √          | √           | √           |

Please run `pip list | grep torch` to get your `pytorch_version`.
```bash
pip install oneccl_bind_pt=={pytorch_version} -f https://developer.intel.com/ipex-whl-stable-cpu
```
where `{pytorch_version}` should be your PyTorch version, for instance 2.4.0.
Check more approaches for [oneccl_bind_pt installation](https://github.com/intel/torch-ccl).
Versions of oneCCL and PyTorch must match.


## Intel® MPI library
Use this standards-based MPI implementation to deliver flexible, efficient, scalable cluster messaging on Intel® architecture. This component is part of the Intel® oneAPI HPC Toolkit.

oneccl_bindings_for_pytorch is installed along with the MPI tool set. Need to source the environment before using it.

```bash
oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
```

#### Intel® Extension for PyTorch installation

Intel Extension for PyTorch (IPEX) provides performance optimizations for CPU training with both Float32 and BFloat16 (refer to the [single CPU section](./perf_train_cpu) to learn more).


The following "Usage in Trainer" takes mpirun in Intel® MPI library as an example.


## Usage in Trainer
To enable multi CPU distributed training in the Trainer with the ccl backend, users should add **`--ddp_backend ccl`** in the command arguments.

Let's see an example with the [question-answering example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)


The following command enables training with 2 processes on one Xeon node, with one process running per one socket. The variables OMP_NUM_THREADS/CCL_WORKER_COUNT can be tuned for optimal performance.
```shell script
 export CCL_WORKER_COUNT=1
 export MASTER_ADDR=127.0.0.1
 mpirun -n 2 -genv OMP_NUM_THREADS=23 \
 python3 examples/pytorch/question-answering/run_qa.py \
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
The following command enables training with a total of four processes on two Xeons (node0 and node1, taking node0 as the main process), ppn (processes per node) is set to 2, with one process running per one socket. The variables OMP_NUM_THREADS/CCL_WORKER_COUNT can be tuned for optimal performance.

In node0, you need to create a configuration file which contains the IP addresses of each node (for example hostfile) and pass that configuration file path as an argument.
```shell script
 cat hostfile
 xxx.xxx.xxx.xxx #node0 ip
 xxx.xxx.xxx.xxx #node1 ip
```
Now, run the following command in node0 and **4DDP** will be enabled in node0 and node1 with BF16 auto mixed precision:
```shell script
 export CCL_WORKER_COUNT=1
 export MASTER_ADDR=xxx.xxx.xxx.xxx #node0 ip
 mpirun -f hostfile -n 4 -ppn 2 \
 -genv OMP_NUM_THREADS=23 \
 python3 examples/pytorch/question-answering/run_qa.py \
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

## Usage with Kubernetes

The same distributed training job from the previous section can be deployed to a Kubernetes cluster using the
[Kubeflow PyTorchJob training operator](https://www.kubeflow.org/docs/components/training/user-guides/pytorch).

### Setup

This example assumes that you have:
* Access to a Kubernetes cluster with [Kubeflow installed](https://www.kubeflow.org/docs/started/installing-kubeflow)
* [`kubectl`](https://kubernetes.io/docs/tasks/tools) installed and configured to access the Kubernetes cluster
* A [Persistent Volume Claim (PVC)](https://kubernetes.io/docs/concepts/storage/persistent-volumes) that can be used
  to store datasets and model files. There are multiple options for setting up the PVC including using an NFS
  [storage class](https://kubernetes.io/docs/concepts/storage/storage-classes) or a cloud storage bucket.
* A Docker container that includes your model training script and all the dependencies needed to run the script. For
  distributed CPU training jobs, this typically includes PyTorch, Transformers, Intel Extension for PyTorch, Intel
  oneCCL Bindings for PyTorch, and OpenSSH to communicate between the containers.

The snippet below is an example of a Dockerfile that uses a base image that supports distributed CPU training and then
extracts a Transformers release to the `/workspace` directory, so that the example scripts are included in the image:
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
The image needs to be built and copied to the cluster's nodes or pushed to a container registry prior to deploying the
PyTorchJob to the cluster.

### PyTorchJob Specification File

The [Kubeflow PyTorchJob](https://www.kubeflow.org/docs/components/training/user-guides/pytorch) is used to run the distributed
training job on the cluster. The yaml file for the PyTorchJob defines parameters such as:
 * The name of the PyTorchJob
 * The number of replicas (workers)
 * The python script and it's parameters that will be used to run the training job
 * The types of resources (node selector, memory, and CPU) needed for each worker
 * The image/tag for the Docker container to use
 * Environment variables
 * A volume mount for the PVC

The volume mount defines a path where the PVC will be mounted in the container for each worker pod. This location can be
used for the dataset, checkpoint files, and the saved model after training completes.

The snippet below is an example of a yaml file for a PyTorchJob with 4 workers running the
[question-answering example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering).
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
To run this example, update the yaml based on your training script and the nodes in your cluster.

<Tip>

The CPU resource limits/requests in the yaml are defined in 
[cpu units](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-cpu)
where 1 CPU unit is equivalent to 1 physical CPU core or 1 virtual core (depending on whether the node is a physical
host or a VM). The amount of CPU and memory limits/requests defined in the yaml should be less than the amount of
available CPU/memory capacity on a single machine. It is usually a good idea to not use the entire machine's capacity in
order to leave some resources for the kubelet and OS. In order to get ["guaranteed"](https://kubernetes.io/docs/concepts/workloads/pods/pod-qos/#guaranteed)
[quality of service](https://kubernetes.io/docs/tasks/configure-pod-container/quality-service-pod) for the worker pods,
set the same CPU and memory amounts for both the resource limits and requests.

</Tip>

### Deploy

After the PyTorchJob spec has been updated with values appropriate for your cluster and training job, it can be deployed
to the cluster using:
```bash
export NAMESPACE=<specify your namespace>

kubectl create -f pytorchjob.yaml -n ${NAMESPACE}
```

The `kubectl get pods -n ${NAMESPACE}` command can then be used to list the pods in your namespace. You should see
the worker pods for the PyTorchJob that was just deployed. At first, they will probably have a status of "Pending" as
the containers get pulled and created, then the status should change to "Running".
```
NAME                                                     READY   STATUS                  RESTARTS          AGE
...
transformers-pytorchjob-worker-0                         1/1     Running                 0                 7m37s
transformers-pytorchjob-worker-1                         1/1     Running                 0                 7m37s
transformers-pytorchjob-worker-2                         1/1     Running                 0                 7m37s
transformers-pytorchjob-worker-3                         1/1     Running                 0                 7m37s
...
```

The logs for worker can be viewed using `kubectl logs <pod name> -n ${NAMESPACE}`. Add `-f` to stream the logs, for example:
```bash
kubectl logs transformers-pytorchjob-worker-0 -n ${NAMESPACE} -f
```

After the training job completes, the trained model can be copied from the PVC or storage location. When you are done
with the job, the PyTorchJob resource can be deleted from the cluster using `kubectl delete -f pytorchjob.yaml -n ${NAMESPACE}`.

## Summary

This guide covered running distributed PyTorch training jobs using multiple CPUs on bare metal and on a Kubernetes
cluster. Both cases utilize Intel Extension for PyTorch and Intel oneCCL Bindings for PyTorch for optimal training
performance, and can be used as a template to run your own workload on multiple nodes.
