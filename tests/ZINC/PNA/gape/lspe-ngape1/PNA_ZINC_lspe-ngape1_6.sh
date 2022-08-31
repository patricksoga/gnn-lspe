#!/bin/bash
#$ -N PNA_ZINC_lspe-ngape1_6
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/6_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

python3 main_ZINC_graph_regression.py --config tests/test-configs/PNA_ZINC_lspe-ngape1.json --job_num 6 --pos_enc_dim 6 --log_file $fname
