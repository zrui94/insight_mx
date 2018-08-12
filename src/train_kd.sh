export MXNET_CPU_WORKER_NTHREADS=48
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

DATA_DIR=/root/upload/mxnet_0611

NETWORK=y1
JOB=arcface
MODELDIR="../model-$NETWORK-$JOB-`date '+%m-%d-%H-%M'`"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log-`date '+%m-%d-%H-%M'`"
# CUDA_VISIBLE_DEVICES='3' nohup python -u train_kd.py --data-dir $DATA_DIR --emb-size 512 --network "$NETWORK" --loss-type 4 --pretrained /root/upload/fanyin/model_qiyims_0611_ft_85ft,138 --prefix "$PREFIX" --per-batch-size 128 --target 'lfw' > "$LOGFILE" 2>&1 &

# CUDA_VISIBLE_DEVICES='3' nohup python -u train_kd.py --data-dir $DATA_DIR --lr 0.005 --mom 0.0 --emb-size 512 --network "$NETWORK" \
# --loss-type 4 --pretrained /root/upload/fanyin/model_qiyims_0611_ft_85ft,138 --prefix "$PREFIX" --per-batch-size 128 --target 'lfw' > "$LOGFILE" 2>&1 &

# CUDA_VISIBLE_DEVICES='3' python train_softmax.py --network r100 --loss-type 4 --lr 0.005 --mom 0.0 --per-batch-size 128 --data-dir /root/upload/mxnet_0611 --pretrained /root/upload/fanyin/model_qiyims_0611_ft_85ft,138 --prefix ../model-r100-arcface

# CUDA_VISIBLE_DEVICES='3' nohup python -u train_kd.py --data-dir $DATA_DIR --lr 0.005 --mom 0.0 --emb-size 128 --network "$NETWORK" \
# --loss-type 4 --pretrained ./pretrained/model,0 --prefix "$PREFIX" --per-batch-size 256 --target 'lfw' > "$LOGFILE" 2>&1 & 

# CUDA_VISIBLE_DEVICES='3' nohup python -u train_kd.py --data-dir $DATA_DIR --lr 0.01 --fc7-wd-mult 10 --wd=4e-5 --emb-size 128 --network "$NETWORK" \
# --loss-type 4 --pretrained ./pretrained/model,0 --prefix "$PREFIX" --per-batch-size 256 --target 'lfw' > "$LOGFILE" 2>&1 & 

# CUDA_VISIBLE_DEVICES='2,3' nohup python -u train_kd1.py --data-dir $DATA_DIR --lr 0.01 --fc7-wd-mult 10 --wd=4e-5 --emb-size 512 --network "$NETWORK" \
# --loss-type 4 --pretrained /root/upload/fanyin/model_qiyims_0611_ft_85ft,138 --pretrained_s ../model-y1-arcface/model,1 --prefix "$PREFIX" \
# --per-batch-size 64 --target 'lfw' > "$LOGFILE" 2>&1 & 

# CUDA_VISIBLE_DEVICES='2,3' nohup python -u train_softmax.py --data-dir $DATA_DIR --lr 0.005 --mom 0.0 --emb-size 128 --network "$NETWORK" \
# --loss-type 4 --pretrained ./pretrained/model,0 --prefix "$PREFIX" --per-batch-size 256 --target 'lfw' > "$LOGFILE" 2>&1 & 

# CUDA_VISIBLE_DEVICES='0,1,2' nohup python -u train_kd1.py --data-dir $DATA_DIR --lr 0.01 --fc7-wd-mult 10 --wd=4e-5 --emb-size 512 --network "$NETWORK" \
# --loss-type 4 --margin-s 32.0 --margin-m 0.3 --pretrained /root/upload/fanyin/model_qiyims_0611_ft_85ft,138 --pretrained_s ./pretrained/model-y1-arcface,144 --prefix "$PREFIX" \
# --per-batch-size 128 --target 'lfw' > "$LOGFILE" 2>&1 & 

# CUDA_VISIBLE_DEVICES='0,1,2' python train_kd1.py --data-dir $DATA_DIR --lr 0.01 --mom 0.0 --fc7-wd-mult 10 --wd=4e-5 --emb-size 512 --network "$NETWORK" \
# --loss-type 4 --margin-s 32.0 --margin-m 0.3 --pretrained /root/upload/fanyin/model_qiyims_0611_ft_85ft,138 --pretrained_s ./pretrained/model-y1-arcface,144 --prefix "$PREFIX" \
# --per-batch-size 100 --verbose 40 --target 'lfw'

# CUDA_VISIBLE_DEVICES='0' python -u train_kd1.py --data-dir $DATA_DIR --lr 0.01 --mom 0.0 --fc7-wd-mult 10 --wd=4e-5 --emb-size 512 --network "$NETWORK" \
# --loss-type 4 --margin-s 32.0 --margin-m 0.3 --pretrained /root/upload/fanyin/model_qiyims_0611_ft_85ft,138 --pretrained_s ./pretrained/model-y1-arcface,144 --prefix "$PREFIX" \
# --per-batch-size 64 --verbose 60 --target 'lfw' > "$LOGFILE" 2>&1 & 

CUDA_VISIBLE_DEVICES='0,1' python -u train_kd.py --data-dir $DATA_DIR --lr 0.01 --mom 0.0 --fc7-wd-mult 10 --wd=4e-5 --emb-size 512 --network "$NETWORK" \
--loss-type 4 --margin-s 32.0 --margin-m 0.3 --pretrained /root/upload/fanyin/model_qiyims_0611_ft_85ft,138 --pretrained_s /root/tensorflow/code/insight-mx/model-y1-arcface-08-10-20-36/model,33 --prefix "$PREFIX" \
--per-batch-size 256 --verbose 200 --lr-steps 14000,54000,94000,114000 --target 'lfw' > "$LOGFILE" 2>&1 & 

# CUDA_VISIBLE_DEVICES='2,3' python -u train_kd.py --data-dir $DATA_DIR --lr 0.01 --mom 0.0 --fc7-wd-mult 10 --wd=4e-5 --emb-size 512 --network "$NETWORK" \
# --loss-type 4 --margin-s 32.0 --margin-m 0.3 --pretrained /root/upload/fanyin/model_qiyims_0611_ft_85ft,138 --pretrained_s ./pretrained/model-y1-arcface,144 --prefix "$PREFIX" \
# --per-batch-size 256 --verbose 200 --target 'lfw' > "$LOGFILE" 2>&1 & 
