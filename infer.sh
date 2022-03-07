CUDA_VISIBLE_DEVICES=3 python test.py --checkpoint results/UNET20220307_2038-bs-4-lr-0.0002-dataflag-grayscale-aug-False/checkpoint_epoch1.pth \
--model=unet \
--dataflag=grayscale \
--dataset=/home/zhaowj/python/powerLine/test_set_200/cam0/*/*
# --dataset=/home/zhaowj/python/powerLine/test_pl_set/cam0/*/*

