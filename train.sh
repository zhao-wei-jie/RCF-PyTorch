CUDA_VISIBLE_DEVICES=1 python train.py \
        --batch-size=4\
        --opt=adamw \
        --max-epoch=1000\
        --gpu=0\
        --print-freq=500\
        --lr=1e-4 \
        --stepsize=50 \
        --gamma=0.9 \
        --weight-decay=1e-2 \
        --iter-size=1\
        --amp=O1 \
        --model=unet \
        --fuse_num=5 \
        --short_cat=2 \
        --dataflag=color \
        --LRLP=True\
        --norm=True \
        --msg=归一化到0-1,不减均值,不除方差\
        --augs=flip\
        --norm_mode=2 \
        # --is_photo_distor=True\
        # --pretrain=results/UNET20220315_0858-bs-4-lr-0.0002-dataflag-color-aug-False/checkpoint_epoch60.pth
        # --scale=True\
        
        # --resume=results/UNET20220308_2121-bs-4-lr-0.0002-dataflag-grayscale-aug-False/checkpoint_epoch70.pth
        # --aug=True
        
        
        
       