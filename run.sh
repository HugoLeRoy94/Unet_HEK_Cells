#!/bin/bash

docker run --gpus all -it --rm\
    -v /home/hcleroy/PostDoc/Colab_David/ExperimentalData/Processed_Cell_Crops/a-Synuclein_Channel/:/app/data_snca \
    -v /home/hcleroy/PostDoc/Colab_David/ExperimentalData/Processed_Cell_Crops/Synphillin_Channel/:/app/data_sncaip \
    -v /home/hcleroy/PostDoc/Colab_David/ExperimentalData/Processed_Cell_Crops/mask/:/app/data_mask \
    -v $PWD:/app \
    -u `stat -c "%u:%g" $PWD`\
    unet_image