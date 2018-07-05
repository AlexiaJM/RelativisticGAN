
#### CIFAR-10

## DCGAN setup
python GAN_losses_iter.py --loss_D 14 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 50000 --print_every 5000 --spectral True
bash fid_script.sh 2 "Stable 1 - RelativisticAverageHingeGAN lr=.0002 D_iters=1 batch=32 betas=(.50, .999) spectral seed 1" 50000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 10 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 50000 --print_every 10000 --spectral True
bash fid_script.sh 2 "Stable 1 - RelativisticGAN      lr=.0002 D_iters=1 batch=32 betas=(.50, .999) spectral seed 1" 50000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 12 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 50000 --print_every 10000 --spectral True
bash fid_script.sh 2 "Stable 1 - RelativisticAverageGAN lr=.0002 D_iters=1 batch=32 betas=(.50, .999) spectral seed 1" 50000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 13 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 50000 --print_every 10000 --spectral True
bash fid_script.sh 2 "Stable 1 - HingeGAN     lr=.0002 D_iters=1 batch=32 betas=(.50, .999) spectral seed 1" 50000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 3 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 50000 --print_every 10000 --spectral True
bash fid_script.sh 2 "Stable 1 - WGAN-GP      lr=.0002 D_iters=1 batch=32 betas=(.50, .999) spectral seed 1" 50000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 1 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 50000 --print_every 10000 --spectral True
bash fid_script.sh 2 "Stable 1 - GAN          lr=.0002 D_iters=1 batch=32 betas=(.50, .999) spectral seed 1" 50000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 2 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 50000 --print_every 10000 --spectral True
bash fid_script.sh 2 "Stable 1 - LSGAN        lr=.0002 D_iters=1 batch=32 betas=(.50, .999) spectral seed 1" 50000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 15 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 50000 --print_every 10000 --spectral True
bash fid_script.sh 2 "Stable 1 - LSGAN_mean   lr=.0002 D_iters=1 batch=32 betas=(.50, .999) spectral seed 1" 50000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 10 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 50000 --print_every 10000 --spectral True --grad_penalty True
bash fid_script.sh 2 "Stable 1 - RelativisticGAN      lr=.0002 D_iters=1 batch=32 betas=(.50, .999) spectral grad_penalty seed 1" 50000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 12 --image_size 32 --seed 1 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 --CIFAR10 True --n_iter 100000 --gen_every 50000 --print_every 10000 --spectral True --grad_penalty True
bash fid_script.sh 2 "Stable 1 - RelativisticAverageGAN lr=.0002 D_iters=1 batch=32 betas=(.50, .999) spectral grad_penalty seed 1" 50000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"

## WGAN-GP setup
python GAN_losses_iter.py --loss_D 14 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 5 --arch 1 --beta1 .50 --beta2 .9 --CIFAR10 True --n_iter 100000 --gen_every 50000 --print_every 10000 --spectral True
bash fid_script.sh 2 "Stable 1 - RelativisticAverageHingeGAN lr=.0001 D_iters=5 batch=32 betas=(.50, .9) spectral seed 1" 50000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 10 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 5 --arch 1 --beta1 .50 --beta2 .9 --CIFAR10 True --n_iter 100000 --gen_every 50000 --print_every 10000 --spectral True
bash fid_script.sh 2 "Stable 1 - RelativisticGAN      lr=.0001 D_iters=5 batch=32 betas=(.50, .9) spectral seed 1" 50000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 12 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 5 --arch 1 --beta1 .50 --beta2 .9 --CIFAR10 True --n_iter 100000 --gen_every 50000 --print_every 10000 --spectral True
bash fid_script.sh 2 "Stable 1 - RelativisticAverageGAN lr=.0001 D_iters=5 batch=32 betas=(.50, .9) spectral seed 1" 50000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 13 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 5 --arch 1 --beta1 .50 --beta2 .9 --CIFAR10 True --n_iter 100000 --gen_every 50000 --print_every 10000 --spectral True
bash fid_script.sh 2 "Stable 1 - HingeGAN     lr=.0001 D_iters=5 batch=32 betas=(.50, .9) spectral seed 1" 50000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 3 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 5 --arch 1 --beta1 .50 --beta2 .9 --CIFAR10 True --n_iter 100000 --gen_every 50000 --print_every 10000 --spectral True
bash fid_script.sh 2 "Stable 1 - WGAN-GP      lr=.0001 D_iters=5 batch=32 betas=(.50, .9) spectral seed 1" 50000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 1 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 5 --arch 1 --beta1 .50 --beta2 .9 --CIFAR10 True --n_iter 100000 --gen_every 50000 --print_every 10000 --spectral True
bash fid_script.sh 2 "Stable 1 - GAN          lr=.0001 D_iters=5 batch=32 betas=(.50, .9) spectral seed 1" 50000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 2 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 5 --arch 1 --beta1 .50 --beta2 .9 --CIFAR10 True --n_iter 100000 --gen_every 50000 --print_every 10000 --spectral True
bash fid_script.sh 2 "Stable 1 - LSGAN        lr=.0001 D_iters=5 batch=32 betas=(.50, .9) spectral seed 1" 50000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 15 --image_size 32 --seed 1 --lr_D 0001 --lr_G .0001 --batch_size 32 --Diters 5 --arch 1 --beta1 .50 --beta2 .9 --CIFAR10 True --n_iter 100000 --gen_every 50000 --print_every 10000 --spectral True
bash fid_script.sh 2 "Stable 1 - LSGAN_mean   lr=.0001 D_iters=5 batch=32 betas=(.50, .9) spectral seed 1" 50000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
python GAN_losses_iter.py --loss_D 10 --image_size 32 --seed 1 --lr_D .0001 --lr_G .0001 --batch_size 32 --Diters 5 --arch 1 --beta1 .50 --beta2 .9 --CIFAR10 True --n_iter 100000 --gen_every 50000 --print_every 10000 --spectral True --grad_penalty True
bash fid_script.sh 10 "Stable 1 - RelativisticGAN      lr=.0001 D_iters=5 batch=32 betas=(.50, .9) spectral grad_penalty seed 1" 50000 "/home/alexia/Datasets/fid_stats_cifar10_train.npz"
