
######## CIFAR-10

# lr=.001
python GAN_losses_iter.py --loss_D 1 --image_size 32 --seed 1 --lr_D .001 --lr_G .001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000
bash fid_script.sh 10 "Unstable - GAN          lr=.001 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 2 --image_size 32 --seed 1 --lr_D .001 --lr_G .001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000
bash fid_script.sh 10 "Unstable - LSGAN        lr=.001 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 3 --image_size 32 --seed 1 --lr_D .001 --lr_G .001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000
bash fid_script.sh 10 "Unstable - WGAN-GP      lr=.001 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 4 --image_size 32 --seed 1 --lr_D .001 --lr_G .001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000
bash fid_script.sh 10 "Unstable - HingeGAN     lr=.001 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 5 --image_size 32 --seed 1 --lr_D .001 --lr_G .001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000
bash fid_script.sh 10 "Unstable - RGAN         lr=.001 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 6 --image_size 32 --seed 1 --lr_D .001 --lr_G .001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000
bash fid_script.sh 10 "Unstable - RaGAN        lr=.001 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 7 --image_size 32 --seed 1 --lr_D .001 --lr_G .001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000
bash fid_script.sh 10 "Unstable - RaLSGAN       lr=.001 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 8 --image_size 32 --seed 1 --lr_D .001 --lr_G .001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000
bash fid_script.sh 10 "Unstable - RaHingeGAN   lr=.001 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"

# betas=(.9, .9)
python GAN_losses_iter.py --loss_D 1 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .9 --beta2 .9 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000
bash fid_script.sh 10 "Unstable - GAN          lr=.0001 D_iters=1 batch=32 betas=(.9, .9) seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 2 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .9 --beta2 .9 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000
bash fid_script.sh 10 "Unstable - LSGAN        lr=.0001 D_iters=1 batch=32 betas=(.9, .9) seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 3 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .9 --beta2 .9 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000
bash fid_script.sh 10 "Unstable - WGAN-GP      lr=.0001 D_iters=1 batch=32 betas=(.9, .9) seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 4 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .9 --beta2 .9 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000
bash fid_script.sh 10 "Unstable - HingeGAN     lr=.0001 D_iters=1 batch=32 betas=(.9, .9) seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 5 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .9 --beta2 .9 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000
bash fid_script.sh 10 "Unstable - RGAN      lr=.0001 D_iters=1 batch=32 betas=(.9, .9) seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 6 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .9 --beta2 .9 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000
bash fid_script.sh 10 "Unstable - RaGAN lr=.0001 D_iters=1 batch=32 betas=(.9, .9) seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 7 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .9 --beta2 .9 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000
bash fid_script.sh 10 "Unstable - RaLSGAN    lr=.0001 D_iters=1 batch=32 betas=(.9, .9) seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 8 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .9 --beta2 .9 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000
bash fid_script.sh 10 "Unstable - RaHingeGAN lr=.0001 D_iters=1 batch=32 betas=(.9, .9) seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"

# no_BN
python GAN_losses_iter.py --loss_D 1 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000 --no_batch_norm_G True --no_batch_norm_D True
bash fid_script.sh 10 "Unstable - GAN          lr=.0001 D_iters=1 batch=32 betas=(.50, .999) no_BN seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 2 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000 --no_batch_norm_G True --no_batch_norm_D True
bash fid_script.sh 10 "Unstable - LSGAN        lr=.0001 D_iters=1 batch=32 betas=(.50, .999) no_BN seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 3 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000 --no_batch_norm_G True --no_batch_norm_D True
bash fid_script.sh 10 "Unstable - WGAN-GP      lr=.0001 D_iters=1 batch=32 betas=(.50, .999) no_BN seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 4 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000 --no_batch_norm_G True --no_batch_norm_D True
bash fid_script.sh 10 "Unstable - HingeGAN     lr=.0001 D_iters=1 batch=32 betas=(.50, .999) no_BN seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 5 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000 --no_batch_norm_G True --no_batch_norm_D True
bash fid_script.sh 10 "Unstable - RGAN         lr=.0001 D_iters=1 batch=32 betas=(.50, .999) no_BN seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 6 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000 --no_batch_norm_G True --no_batch_norm_D True
bash fid_script.sh 10 "Unstable - RaGAN        lr=.0001 D_iters=1 batch=32 betas=(.50, .999) no_BN seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 7 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000 --no_batch_norm_G True --no_batch_norm_D True
bash fid_script.sh 10 "Unstable - RaLSGAN      lr=.0001 D_iters=1 batch=32 betas=(.50, .999) no_BN seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 8 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000 --no_batch_norm_G True --no_batch_norm_D True
bash fid_script.sh 10 "Unstable - RaHingeGAN   lr=.0001 D_iters=1 batch=32 betas=(.50, .999) no_BN seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"

# Tanh
python GAN_losses_iter.py --loss_D 1 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000 --Tanh_GD True
bash fid_script.sh 10 "Unstable - GAN          lr=.0001 D_iters=1 batch=32 betas=(.50, .999) Tanh seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 2 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000 --Tanh_GD True
bash fid_script.sh 10 "Unstable - LSGAN        lr=.0001 D_iters=1 batch=32 betas=(.50, .999) Tanh seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 3 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000 --Tanh_GD True
bash fid_script.sh 10 "Unstable - WGAN-GP      lr=.0001 D_iters=1 batch=32 betas=(.50, .999) Tanh seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 4 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000 --Tanh_GD True
bash fid_script.sh 10 "Unstable - HingeGAN     lr=.0001 D_iters=1 batch=32 betas=(.50, .999) Tanh seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 5 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000 --Tanh_GD True
bash fid_script.sh 10 "Unstable - RGAN         lr=.0001 D_iters=1 batch=32 betas=(.50, .999) Tanh seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 6 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000 --Tanh_GD True
bash fid_script.sh 10 "Unstable - RaGAN        lr=.0001 D_iters=1 batch=32 betas=(.50, .999) Tanh seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 7 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000 --Tanh_GD True
bash fid_script.sh 10 "Unstable - RaLSGAN      lr=.0001 D_iters=1 batch=32 betas=(.50, .999) Tanh seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 8 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 10000 --print_every 10000 --Tanh_GD True
bash fid_script.sh 10 "Unstable - RaHingeGAN   lr=.0001 D_iters=1 batch=32 betas=(.50, .999) Tanh seed 1" 10000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"


######### Meow tests 64x64 stable (about 3 hours for 100k iterations)
python GAN_losses_iter.py --loss_D 1 --image_size 64 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 64 --Diters 1 --arch 0 --beta1 .50 --beta2 .999 --n_iter 100000 --gen_every 10000 --print_every 10000 --G_h_size 64 --D_h_size 64 --gen_extra_images 10000
bash fid_script.sh 10 "Meow64     - GAN        lr=.0002 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/Meow_64x64/cats_bigger_than_64x64"

python GAN_losses_iter.py --loss_D 5 --image_size 64 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 0 --beta1 .50 --beta2 .999 --n_iter 100000 --gen_every 10000 --print_every 10000 --G_h_size 64 --D_h_size 64 --gen_extra_images 10000
bash fid_script.sh 10 "Meow64     - RGAN       lr=.0002 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/Meow_64x64/cats_bigger_than_64x64"
python GAN_losses_iter.py --loss_D 6 --image_size 64 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 0 --beta1 .50 --beta2 .999 --n_iter 100000 --gen_every 10000 --print_every 10000 --G_h_size 64 --D_h_size 64 --gen_extra_images 10000
bash fid_script.sh 10 "Meow64     - RaGAN      lr=.0002 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/Meow_64x64/cats_bigger_than_64x64"

python GAN_losses_iter.py --loss_D 2 --image_size 64 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 64 --Diters 1 --arch 0 --beta1 .50 --beta2 .999 --n_iter 100000 --gen_every 10000 --print_every 10000 --G_h_size 64 --D_h_size 64 --gen_extra_images 10000
bash fid_script.sh 10 "Meow64     - LSGAN      lr=.0002 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/Meow_64x64/cats_bigger_than_64x64"
python GAN_losses_iter.py --loss_D 7 --image_size 64 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 0 --beta1 .50 --beta2 .999 --n_iter 100000 --gen_every 10000 --print_every 1000 --G_h_size 64 --D_h_size 64 --gen_extra_images 10000
bash fid_script.sh 10 "Meow64     - RaLSGAN    lr=.0002 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/Meow_64x64/cats_bigger_than_64x64"

python GAN_losses_iter.py --loss_D 4 --image_size 64 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 0 --beta1 .50 --beta2 .999 --n_iter 100000 --gen_every 10000 --print_every 10000 --G_h_size 64 --D_h_size 64 --gen_extra_images 10000
bash fid_script.sh 10 "Meow64     - HingeGAN   lr=.0002 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/Meow_64x64/cats_bigger_than_64x64"
python GAN_losses_iter.py --loss_D 8 --image_size 64 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 0 --beta1 .50 --beta2 .999 --n_iter 100000 --gen_every 10000 --print_every 10000 --G_h_size 64 --D_h_size 64 --gen_extra_images 10000
bash fid_script.sh 10 "Meow64     - RaHingeGAN lr=.0002 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/Meow_64x64/cats_bigger_than_64x64"

python GAN_losses_iter.py --loss_D 5 --image_size 64 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 0 --beta1 .50 --beta2 .999 --n_iter 100000 --gen_every 10000 --print_every 10000 --G_h_size 64 --D_h_size 64 --gen_extra_images 10000 --grad_penalty True
bash fid_script.sh 10 "Meow64     - RGAN       lr=.0002 D_iters=1 batch=32 betas=(.50, .999) seed 1 grad_penalty" 10000 "/home/alexia/Datasets/Meow_64x64/cats_bigger_than_64x64"
python GAN_losses_iter.py --loss_D 6 --image_size 64 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 0 --beta1 .50 --beta2 .999 --n_iter 100000 --gen_every 10000 --print_every 10000 --G_h_size 64 --D_h_size 64 --gen_extra_images 10000 --grad_penalty True
bash fid_script.sh 10 "Meow64     - RaGAN      lr=.0002 D_iters=1 batch=32 betas=(.50, .999) seed 1 grad_penalty" 10000 "/home/alexia/Datasets/Meow_64x64/cats_bigger_than_64x64"

######### Meow tests 128x128 stable (about 22.7 hours for 100k iterations)

# Fails
python GAN_losses_iter.py --loss_D 1 --image_size 128 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 0 --beta1 .50 --beta2 .999 --n_iter 20000 --gen_every 10000 --print_every 10000 --G_h_size 64 --D_h_size 64 --gen_extra_images 10000 --input_folder '/home/alexia/Datasets/Meow_128x128'
bash fid_script.sh 2 "Meow     - GAN           lr=.0001 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/Meow_128x128/cats_bigger_than_128x128"

python GAN_losses_iter.py --loss_D 5 --image_size 128 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 0 --beta1 .50 --beta2 .999 --n_iter 100000 --gen_every 10000 --print_every 10000 --G_h_size 64 --D_h_size 64 --gen_extra_images 10000 --input_folder '/home/alexia/Datasets/Meow_128x128'
bash fid_script.sh 10 "Meow128     - RGAN      lr=.0002 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/Meow_128x128/cats_bigger_than_128x128"
python GAN_losses_iter.py --loss_D 6 --image_size 128 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 0 --beta1 .50 --beta2 .999 --n_iter 100000 --gen_every 10000 --print_every 10000 --G_h_size 64 --D_h_size 64 --gen_extra_images 10000 --input_folder '/home/alexia/Datasets/Meow_128x128'
bash fid_script.sh 10 "Meow128     - RaGAN     lr=.0002 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/Meow_128x128/cats_bigger_than_128x128"

python GAN_losses_iter.py --loss_D 2 --image_size 128 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 0 --beta1 .50 --beta2 .999 --n_iter 100000 --gen_every 10000 --print_every 10000 --G_h_size 64 --D_h_size 64 --gen_extra_images 10000 --input_folder '/home/alexia/Datasets/Meow_128x128'
bash fid_script.sh 10 "Meow128     - LSGAN     lr=.0002 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/Meow_128x128/cats_bigger_than_128x128"
python GAN_losses_iter.py --loss_D 7 --image_size 128 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 0 --beta1 .50 --beta2 .999 --n_iter 100000 --gen_every 10000 --print_every 10000 --G_h_size 64 --D_h_size 64 --gen_extra_images 10000 --input_folder '/home/alexia/Datasets/Meow_128x128'
bash fid_script.sh 10 "Meow128     - RaLSGAN   lr=.0002 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/Meow_128x128/cats_bigger_than_128x128"

######### Meow tests 256x256 stable

#Fails
python GAN_losses_iter_PAC.py --loss_D 1 --image_size 256 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 0 --beta1 .50 --beta2 .999 --n_iter 100000 --gen_every 10000 --print_every 1000 --G_h_size 32 --D_h_size 32 --gen_extra_images 2000 --input_folder '/home/alexia/Datasets/Meow_256x256'
bash fid_script.sh 10 "Meow256     - GAN       lr=.0001 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/Meow_256x256/cats_bigger_than_256x256"

python GAN_losses_iter_PAC.py --loss_D 5 --image_size 256 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 0 --beta1 .50 --beta2 .999 --n_iter 100000 --gen_every 10000 --print_every 1000 --G_h_size 32 --D_h_size 32 --gen_extra_images 2000 --input_folder '/home/alexia/Datasets/Meow_256x256'
bash fid_script.sh 10 "Meow256     - RGAN      lr=.0002 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/Meow_256x256/cats_bigger_than_256x256"
python GAN_losses_iter_PAC.py --loss_D 6 --image_size 256 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 0 --beta1 .50 --beta2 .999 --n_iter 100000 --gen_every 10000 --print_every 1000 --G_h_size 32 --D_h_size 32 --gen_extra_images 2000 --input_folder '/home/alexia/Datasets/Meow_256x256'
bash fid_script.sh 10 "Meow256     - RaGAN     lr=.0002 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/Meow_256x256/cats_bigger_than_256x256"

#Fails
python GAN_losses_iter_PAC.py --loss_D 2 --image_size 256 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 0 --beta1 .50 --beta2 .999 --n_iter 100000 --gen_every 10000 --print_every 1000 --G_h_size 32 --D_h_size 32 --gen_extra_images 2000 --input_folder '/home/alexia/Datasets/Meow_256x256'
bash fid_script.sh 10 "Meow256     - LSGAN     lr=.0002 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/Meow_256x256/cats_bigger_than_256x256"
python GAN_losses_iter_PAC.py --loss_D 7 --image_size 256 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 0 --beta1 .50 --beta2 .999 --n_iter 100000 --gen_every 10000 --print_every 1000 --G_h_size 32 --D_h_size 32 --gen_extra_images 2000 --input_folder '/home/alexia/Datasets/Meow_256x256'
bash fid_script.sh 10 "Meow256     - RaLSGAN   lr=.0002 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/Meow_256x256/cats_bigger_than_256x256"

# Extremely long (50+h) couldn't run it in one shot, had to save and load multiple times
python GAN_losses_iter_PAC.py --loss_D 3 --image_size 256 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 0 --beta1 .50 --beta2 .999 --n_iter 100000 --gen_every 10000 --print_every 1000 --G_h_size 32 --D_h_size 32 --gen_extra_images 2000 --input_folder '/home/alexia/Datasets/Meow_256x256'
bash fid_script.sh 10 "Meow256     - WGAN-GP   lr=.0002 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/Meow_256x256/cats_bigger_than_256x256"

python GAN_losses_iter_PAC.py --loss_D 1 --image_size 256 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 0 --beta1 .50 --beta2 .999 --n_iter 100000 --gen_every 10000 --print_every 1000 --G_h_size 32 --D_h_size 32 --gen_extra_images 2000  --spectral True --input_folder '/home/alexia/Datasets/Meow_256x256'
bash fid_script.sh 10 "Meow256     - SpectralSGAN lr=.0001 D_iters=1 batch=32 betas=(.50, .999) seed 1" 10000 "/home/alexia/Datasets/Meow_256x256/cats_bigger_than_256x256"
