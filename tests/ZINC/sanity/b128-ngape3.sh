#!/bin/bash
#$ -N LSPE-gape-b128
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-1:1

pos_enc_dim=(0 20)
fname=$(pwd)/b128-bnorm-alt-edge_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/gnn-lspe

python main_ZINC_graph_regression.py --config ../configs/GatedGCN_ZINC_LSPE_withLapEigLoss.json --n_gape 3 --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]}
