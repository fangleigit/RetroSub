# View and try the demos
We provide two options to view and try our demo.
## Option 1: docker (recommended).
Note: This container was tested on Ubuntu [CPU_only, P100, V100] and Windows 11 with [Docker Desktop](https://www.docker.com/products/docker-desktop/) [CPU_only, P100, V100, GeForce930M (laptop GPU)]. 
We do not support A100 as we use [MolecularTransformer](https://github.com/pschwllr/MolecularTransformer.git) as git submodule, and its cuda version is not supported.

    # the below command will start a jupyter container, click the 
    # link 'http://127.0.0.1:8888/?token=xxxxx' and open the 
    # 'demo.ipynb' notebook. You can try our demo with your own cases.    

    # cpu only
    docker run -p 8888:8888 leifa/retrosub:1    

    # gpu
    docker run --gpus 1  -p 8888:8888 leifa/retrosub:1
    # NOTE: Please change the batch size in `demo_data/subseq2seq.sh` with jupyter if necessary.


    # kill the container
    docker ps # this will show CONTAINER_ID in the first column
    docker kill CONTAINER_ID


## Option 2: setup on your own machine (Windows is not supported)
Note: the following setup was tested on Ubuntu [CPU_only (default), P100, V100]. 

Please follow the [setup](README.md#setup) and unzip our [shared file](https://bdmstorage.blob.core.windows.net/shared/release_data.tar.gz) in the root folder.


### step 2-1: change code of submodule (reaction retrieval) to run on CPU in the code folder.
    bash scripts/step0_fix_demo.sh 
    

### step 2-5: run [demo.ipynb](demo.ipynb), and try your own cases.
    conda activate retrosub && jupyter notebook
    # NOTE: Please change the batch size in `demo_data/subseq2seq.sh` with jupyter if necessary.