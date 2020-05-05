import os, shutil, glob, re
import matplotlib.pyplot as plt
import glob
import subprocess


def rem_checkmodel(target):
  remdir = '/content/drive/My Drive/Colabdata/irradiance_estimation/checkpoints/{}/'.format(target)
  for i in range(5, 195, 5):
    remfile = remdir+'{}_net_G.pth'.format(i)
    try:
      os.remove(remfile)
      print('Rem: ', remfile)
    except OSError as e:
      print('No exist: ', remfile)
      pass  

    remfile = remdir+'{}_net_G2.pth'.format(i)
    try:
      os.remove(remfile)
      print('Rem: ', remfile)
    except OSError as e:
      print('No exist: ', remfile)
      pass  


def eval_data(target, docopy=True):
  tgtdir = "/content/drive/My Drive/Colabdata/irradiance_estimation/results/{}/test_latest/images/".format(target)
  print('Target Directory: {}'.format(tgtdir))
#   realname = "_real_B.png"
#   fakename = "_fake_B.png"
  realname = "_real_B*.png"
  fakename = "_fake_B*.png"
  realdir = '/content/drive/My Drive/Colabdata/PerceptualSimilarity/test/real/'
  fakedir = '/content/drive/My Drive/Colabdata/PerceptualSimilarity/test/fake/'
  outdir = "/content/drive/My Drive/Colabdata/irradiance_estimation/results/{}/test_latest/".format(target)
  ipips_py = '/content/drive/My Drive/Colabdata/PerceptualSimilarity/compute_dists_dirs_rev.py'
  outname = "dist_ipips.txt"
  if docopy:
    shutil.rmtree(realdir)
    print('Removed:', realdir)
    shutil.rmtree(fakedir)
    print('Removed:', fakedir)
    os.makedirs(realdir, exist_ok=True)
    print('Maked:', realdir)
    os.makedirs(fakedir, exist_ok=True)
    print('Maked:', fakedir)
    for full in glob.glob(tgtdir+'*'+realname):
        p = os.path.basename(full)
        name = p.replace('_real_B', '')
        name = re.sub('.*\D_0', '0', name, 1)
        shutil.copy(full, realdir+name)
        print(realdir+name)
    for full in glob.glob(tgtdir+'*'+fakename):
        p = os.path.basename(full)
        name = p.replace('_fake_B', '')
        name = re.sub('.*\D_0', '0', name, 1)
        shutil.copy(full, fakedir+name)
        print(fakedir+name)

  out = outdir+outname
  print('cur dir:', os.getcwd())
  subprocess.run(['python', ipips_py, '-d0', realdir, '-d1', fakedir,'--use_gpu','-o','{}'.format(out)])

  from skimage.measure import compare_ssim, compare_psnr
  from skimage.color import rgb2gray
  from skimage.metrics import peak_signal_noise_ratio, structural_similarity

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