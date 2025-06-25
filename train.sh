###
 # @description: 
 # @param : 
 # @return: 
 # @Author: xinyebei@xinye.com
 # @Date: 2025-05-29 16:09:58
 # @LastEditors: xinyebei@xinye.com
### 
TASK_TARGET=$1
PORT=$2
CONFIG=$3
export OMP_NUM_THREADS=8 && export CUDA_VISIBLE_DEVICES=0,1 && \
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT train.py --detector_path  $CONFIG \
--no-save_ckpt --task_target $TASK_TARGET --no-save_feat --ddp