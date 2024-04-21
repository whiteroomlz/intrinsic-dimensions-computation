#!/bin/bash

module load gnu8
module load Python

if conda env list | grep "intrinsic-dims-env" >/dev/null 2>&1; then
    source activate intrinsic-dims-env
else
    echo 'y' | conda create -n intrinsic-dims-env python=3.10
    source activate intrinsic-dims-env
    pip install requests python-dateutil pytz matplotlib openpyxl pandas tqdm git+https://github.com/whiteroomlz/mst-clustering.git
fi

cd src
sbatch --dependency=singleton --job-name=$1 produce-batches.sh
sbatch --dependency=singleton --job-name=$1 build-msts.sh
sbatch --dependency=singleton --job-name=$1 compute-dimensions.sh
