# Training U-Net

# Training run
To run the training of the UNET model using the GPU, we must create a docker container in which tensorflow can run.  To run the training, simply execute :
```./run.sh```
Each time the python script : "training.py" is modified, it is necessary to rebuild the docker image, using:
```docker build -t unet_image -f Dockerfile .```

# test unet
## Build custom tf jupyter image
```docker build -t custom-tf-gpu-jupyter -f Dockerfile.tf_jup .```

## Run Jupyter
To run a jupyter notebook instance on docker : tensorflow:2.14.0-gpu-jupyter:
doc
```docker run --gpus all -it --rm -p 8888:8888  -v /:/tf/notebooks custom-tf-gpu-jupyter```


- ```--gpus all```: Enables GPU support.
- ```-it```: Interactive mode with a terminal.
- ```--rm```: Automatically remove the container when it exits.
- ```-p 8888:8888```: Maps port 8888 on your host to port 8888 in the container (where Jupyter runs).
- ```-v /:/tf/notebooks```: Mounts a directory from your host into the container for persistent storage of your notebooks.
Careful, I think the /tf/notebooks is important, if you don't give this address, jupyter can't find anything.