#!/usr/bin/env bash
#SBATCH --job-name=phyFea_OCRNet_training
#SBATCH --output=/cluster/work/cvl/shbasu/phyfeaOCRNet/results/phyFea_ocr_training_output.log
#SBATCH --error=/cluster/work/cvl/shbasu/phyfeaOCRNet/results/phyFea_ocr_training_error.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shamik.basu@studio.unibo.it
#SBATCH --gpus=rtx_3090:4
#SBATCH --mem-per-cpu=40G
#SBATCH --ntasks=4
#SBATCH --time=1:00:00


GPUS=4
PORT=$(shuf -i 15661-55661 -n 1)

source /cluster/apps/local/env2lmod.sh
module load  eth_proxy


python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT\
    /cluster/work/cvl/shbasu/phyfeaOCRNet/PhyFea-OCRNet/tools/train.py\
   --cfg /cluster/work/cvl/shbasu/phyfeaOCRNet/PhyFea-OCRNet/experiments/cityscapes/seg_hrnet_ocr_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml\
   --seed=138
   --local_rank=4

