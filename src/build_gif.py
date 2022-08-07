import imageio
import os
import time

def build_gif(pattern='@2x', sample=0, gif_name='pretrain_apn_cub200', cache_path='build/img'):
    print(f'pattern {pattern} sample {sample}')
    files = [x for x in os.listdir(cache_path) if f"_sample_{sample}{pattern}" in x]
    files.sort(key=lambda x: int(x.split('@')[0].split('_')[1]))
    gif_images = [imageio.imread(f'{cache_path}/{img_file}') for img_file in files]
    imageio.mimsave(f"build/{gif_name}{pattern}_{str(sample)}-{int(time.time())}.gif", gif_images, fps=8)

for i in range(4):
    build_gif(pattern='@2x', sample=i, gif_name='pretrain_apn_cub200')
    build_gif(pattern='@4x', sample=i, gif_name='pretrain_apn_cub200')
