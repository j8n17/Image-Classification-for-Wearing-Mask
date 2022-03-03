# python train.py --epochs 30 --model GoogLeNet --name google_halfsize_epoch30_cutmix0.7_reg --beta 1 --cutmix_prob 0.7
# python train.py --epochs 180 --model ShuffleNet --name shuffle_fullsize_epoch180_cutmix0.7_reg --beta 1 --cutmix_prob 0.7
# python train.py --epochs 150 --model GoogLeNet --name google_halfsize_epoch150_cutmix0.5_reg_aug_step30 --beta 1 --cutmix_prob 0.5
python train.py --epochs 150 --model GoogLeNet --name google_halfsize_epoch150_cutmix0.5_reg_aug_cosine --beta 1 --cutmix_prob 0.5 --sch cosine
