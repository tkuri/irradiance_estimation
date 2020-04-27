import os, shutil, glob, re
import matplotlib.pyplot as plt
import glob
import subprocess


def eval_data(data, target_list):
    # data = 'o2l2_mv_scl_black'
    # target_list = [
    #                 '{}/cbox_in_to_irrgt_unet256'.format(data), 
    #               ]
    
    
    docopy = True
    os.chdir("/content/drive/My Drive/Colabdata/PerceptualSimilarity/")
    
    for target in target_list:
      tgtdir = "/content/drive/My Drive/Colabdata/pytorch-CycleGAN-and-pix2pix/results/{}/test_latest/images/".format(target)
      print('Target Directory: {}'.format(tgtdir))
      realname = "_real_B.png"
      fakename = "_fake_B.png"
      outdir = "/content/drive/My Drive/Colabdata/pytorch-CycleGAN-and-pix2pix/results/{}/test_latest/".format(target)
      outname = "dist_ipips.txt"
      if docopy:
        for full in glob.glob(tgtdir+'*'+realname):
            p = os.path.basename(full)
            name = p.replace('_real_B', '')
            name = re.sub('.*_0', '0', name)
            shutil.copy(full, 'test/real/'+name)
            print('test/real/'+name)
        for full in glob.glob(tgtdir+'*'+fakename):
            p = os.path.basename(full)
            name = p.replace('_fake_B', '')
            name = re.sub('.*_0', '0', name)
            shutil.copy(full, 'test/fake/'+name)
            print('test/fake/'+name)
    
      out = outdir+outname
      print('cur dir:', os.getcwd())
    #   !python compute_dists_dirs_rev.py -d0 "./test/real/" -d1 "./test/fake/" --use_gpu -o "$out"
      subprocess.run(['python','compute_dists_dirs_rev.py','-d0', r'./test/real/','-d1', r'./test/fake/','--use_gpu','-o','{}'.format(out)])
    
      from skimage.measure import compare_ssim, compare_psnr
      from skimage.color import rgb2gray
      from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    
      realdir = './test/real/'
      fakedir = './test/fake/'
      outdir = "/content/drive/My Drive/Colabdata/pytorch-CycleGAN-and-pix2pix/results/{}/test_latest/".format(target)
      outname = "dist_psnr_ssim.txt"
      f = open(outdir+outname,'w')
      f.writelines('file, psnr(rgb), ssim(rgb), psnr(gray), ssim(gray)\n')
      print('file, psnr(rgb), ssim(rgb), psnr(gray), ssim(gray)')
      #files = os.listdir(realdir)
      files = sorted(glob.glob(realdir+'*.png'))
      means = {'psnr_rgb': [], 'ssim_rgb': [], 'psnr_gray': [], 'ssim_gray': []}
      for file in files:
        fi = os.path.basename(file)
        real = plt.imread(realdir + fi)
        fake = plt.imread(fakedir + fi)
        real_gray = rgb2gray(real)
        fake_gray = rgb2gray(fake)
        psnr_rgb = peak_signal_noise_ratio(real, fake)
        psnr_gray = peak_signal_noise_ratio(real_gray, fake_gray)
        ssim_gray = structural_similarity(real_gray, fake_gray)
        ssim_rgb = structural_similarity(real, fake, multichannel=True)
    
        print('{}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(fi, psnr_rgb, ssim_rgb, psnr_gray, ssim_gray))
        f.writelines('{}, {:.6f}, {:.6f}, {:.6f}, {:.6f}\n'.format(fi, psnr_rgb, ssim_rgb, psnr_gray, ssim_gray))
        means['psnr_rgb'].append(psnr_rgb)
        means['ssim_rgb'].append(ssim_rgb)
        means['psnr_gray'].append(psnr_gray)
        means['ssim_gray'].append(ssim_gray)
    
      mean_psnr_rgb = sum(means['psnr_rgb'])/len(means['psnr_rgb'])
      mean_ssim_rgb = sum(means['ssim_rgb'])/len(means['ssim_rgb'])
      mean_psnr_gray = sum(means['psnr_gray'])/len(means['psnr_gray'])
      mean_ssim_gray = sum(means['ssim_gray'])/len(means['ssim_gray'])
      print('means, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(mean_psnr_rgb, mean_ssim_rgb, mean_psnr_gray, mean_ssim_gray))
      f.writelines('means, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(mean_psnr_rgb, mean_ssim_rgb, mean_psnr_gray, mean_ssim_gray))
      f.close()