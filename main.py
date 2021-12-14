import argparse
from pathlib import Path
from pansharpen import rescale_img, simple_mean, esri, brovey, pca, nmf, wavelet, gs
from utils import *


def main(method, out_dir, zip_name='sub.zip', visualize=True):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img_data = load_data()

    if method == 'rescale':
        generated_data = rescale_img(img_data)

    elif method == 'simple_mean':
        generated_data = simple_mean(img_data)

    elif method == 'esri':
        generated_data = esri(img_data)

    elif method == 'brovey':
        generated_data = brovey(img_data, W=0.25)
    
    elif method == 'gs':
        generated_data = gs(img_data)

    elif method == 'wavelet':
        generated_data = wavelet(img_data)

    elif method == 'pca':
        generated_data = pca(img_data)

    elif method == 'lin-pca':
        generated_data = pca(img_data, 'linear')
    
    elif method == 'rbf-pca':
        generated_data = pca(img_data, 'rbf')

    elif method == 'frob-nmf':
        generated_data = nmf(img_data)

    elif method == 'kl-nmf':
        generated_data = nmf(img_data, 'kullback-leibler')

    elif method == 'is-nmf':
        generated_data = nmf(img_data, 'itakura-saito')

    if visualize:
        out_path = str(Path(out_dir) / f'{method}.png')
        visualize_rgb(generated_data, out_path)
    
    create_submission(out_dir, generated_data, zip_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-method', '-m', help="method name", type=str, required=True)
    parser.add_argument('-out_dir', '-o', help="output directory of pan-sharpened images and the zip file", type=str, required=True)
    parser.add_argument('-zip_name', '-z', help="zip file name", type=str, default="sub.zip")
    parser.add_argument('--vis', help='visualization of the RGB pan-sharpened image' , action='store_true')
    args = parser.parse_args()

    main(args.method, args.out_dir, args.zip_name, args.vis)