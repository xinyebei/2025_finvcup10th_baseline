TASK_TARGET=$1
PORT=$2
export OMP_NUM_THREADS=4 && export CUDA_VISIBLE_DEVICES=0,1 && python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT training/fintune.py --no-save_ckpt --task_target $TASK_TARGET --no-save_feat --ddp