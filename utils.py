import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import subprocess
import matplotlib.pyplot as plt

UINT16_MAX = 65535
def load_data():
    pan = cv2.imread('data/pan.tif', -1)
    red = cv2.imread('data/lr_red.tif', -1)
    blue = cv2.imread('data/lr_blue.tif', -1)
    green = cv2.imread('data/lr_green.tif', -1)
    nir = cv2.imread('data/lr_nir.tif', -1)

    data = { 
        'pan' : pan.astype(np.float64) / UINT16_MAX,
        'red' : red.astype(np.float64) / UINT16_MAX,
        'blue' : blue.astype(np.float64) / UINT16_MAX,
        'green' : green.astype(np.float64) / UINT16_MAX,
        'nir' : nir.astype(np.float64) / UINT16_MAX
    }
    return data


def save_image(path, image):
    im = np.array(Image.fromarray(image * UINT16_MAX), dtype=np.uint16)
    cv2.imwrite(path,np.uint16(cv2.resize(im, (1200, 1200), interpolation=cv2.INTER_CUBIC)))

def create_submission(out_dir, images, zip_name="sub.zip"):
    # save pansharpened images
    save_image(str(Path(out_dir) / "red.tif"), images['red'])
    save_image(str(Path(out_dir) / "green.tif"), images['green'])
    save_image(str(Path(out_dir) / "blue.tif"), images['blue'])
    save_image(str(Path(out_dir) / "nir.tif"), images['nir'])
    
    # generate zip file
    bashCommand = f"zip -r {zip_name} red.tif green.tif blue.tif nir.tif"
    process = subprocess.Popen(bashCommand.split(), cwd=out_dir, stdout=subprocess.PIPE)
    _, _ = process.communicate()

def to_rgb(images):
    h, w = images['red'].shape
    image = np.zeros((3, h, w))
    image[0] = images['red']
    image[1] = images['green']
    image[2] = images['blue']
    return np.transpose(image, [1, 2, 0])

def to_rgbn(images):
    h, w = images['red'].shape
    image = np.zeros((4, h, w))
    image[0] = images['red']
    image[1] = images['green']
    image[2] = images['blue']
    image[3] = images['nir']
    return np.transpose(image, [1, 2, 0])



def visualize(images, out_path):
    rgb = to_rgb(images)
    norm_image = cv2.normalize(rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    plt.imshow(norm_image)
    plt.savefig(out_path, dpi=300)

def visualize_rgb(rgb, out_path):
    rgb = rgb[: , :,  :3]
    norm_image = cv2.normalize(rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    plt.imshow(norm_image)
    plt.savefig(out_path, dpi=300)