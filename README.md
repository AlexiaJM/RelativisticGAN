# Relativistic GAN

Code to replicate all analyses from the paper [The relativistic discriminator: a key element missing from standard GAN](https://arxiv.org/abs/1807.00734)
**Discussion at https://ajolicoeur.wordpress.com/RelativisticGAN.**

**Needed**

* Python 3.6
* Pytorch (Latest from source)
* Tensorflow (Latest from source, needed to get FID)
* Cat Dataset (http://academictorrents.com/details/c501571c29d16d7f41d159d699d0e7fb37092cbd)

**To do beforehand**

* Change all folders locations in fid_script.sh, stable.sh, unstable.sh, GAN_losses_iter.py, GAN_losses_iter_PAC.py
* Make sure that there are existing folders at the locations you used
* If you want to use the CAT dataset
  * Run setting_up_script.sh (I recommend you open it and run lines manually)
  * Move cats folder to your favorite place
  * mv cats_bigger_than_64x64 "your_input_folder_64x64"
  * mv cats_bigger_than_128x128 "your_input_folder_128x128"
  * mv cats_bigger_than_256x256 "your_input_folder_256x256"

**To run**
* Open stable.sh and run the lines you want
* Open unstable.sh and run the lines you want

**Notes**

Installing both Tensorflow and Pytorch in the same computer is a bit annoying. If you just want to generate pictures and you do not care about the Fréchet Inception Distance (FID), you do not need to download Tensorflow.

Although I always used the same seed (seed = 1), keep in mind that your results may be sightly different. Neural networks are notoriously difficult to perfectly replicate. You never know if something that I changed while working on the paper could have affected the randomness in some of my earlier experiments (I did stable experiments first, then unstable experiments for CIFAR-10, and most recently the unstable experiments for CAT). CUDNN is also said to introduce some randomness and I use it. I forgot to put a seed for the tensorflow FID scripts so FIDs may vary by a 2-3 points, but hopefully they shouldn't change much since I used a very large amount of images for the calculations.

# Results

**64x64 cats with RaLSGAN (FID = 11.97)**

![](/images/best_64x64_crop.png)

**128x128 cats with RaLSGAN (FID = 15.85)**

![](/images/best_128x128_crop.png)

**256x256 cats with SGAN (5k iterations)**

![](/images/GAN.jpeg)

**256x256 cats with LSGAN (5k iterations)**

![](/images/LSGAN.jpeg)

**256x256 cats with RaSGAN (FID = 32.11)**

![](/images/RaSGAN.jpeg)

**256x256 cats with RaLSGAN (FID = 35.21)**

![](/images/RaLSGAN.jpeg)

**256x256 cats with SpectralSGAN (FID = 54.73)**

![](/images/SpectralSGAN.jpeg)

**256x256 cats with WGAN-GP (FID > 100)**

![](/images/WGAN-GP.jpeg)
