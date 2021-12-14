from matplotlib import image
import cv2
from skimage.transform import rescale
import numpy as np
from sklearn.decomposition import PCA, KernelPCA, NMF
from utils import to_rgbn, visualize_rgb
import pywt

BANDS = ['red', 'green', 'blue', 'nir']

def rescale_img(images):
    rescaled_ms = { 
        name: rescale(images[name].astype(np.float64), (4, 4)) for name in BANDS }
    return rescaled_ms

def simple_mean(images):
    rescaled_ms = rescale_img(images)
    generated_image = { name : 0.5 * (rescaled_ms[name] + images['pan']) for name in BANDS }
    return generated_image

def esri(images):
    rescaled_ms = rescale_img(images)
    all_in = rescaled_ms['red'] + rescaled_ms['green'] + rescaled_ms['blue'] + rescaled_ms['nir']
    ADJ = images['pan'] - all_in / 4
    generated_image = { name : rescaled_ms[name] + ADJ for name in BANDS }
    return generated_image

def brovey(images, W=0.1):
    rescaled_ms = rescale_img(images)
    DNF = (images['pan']  - W * rescaled_ms['nir']) / (W * (rescaled_ms['red'] + rescaled_ms['green'] + rescaled_ms['blue']))
    generated_image = { name : rescaled_ms[name] * DNF for name in BANDS }
    return generated_image

def pca(images, kernel=None):
    rescaled_ms = rescale_img(images)
    rgbn = to_rgbn(rescaled_ms).reshape((-1, 4))
    if kernel is None:
        model = PCA(n_components=4)
    else:
        model = KernelPCA(n_components=4, kernel=kernel)

    z = model.fit_transform(rgbn)
    pan_vec = images['pan'].flatten()

    z[:, 0] = ((pan_vec - np.mean(pan_vec)) * np.std(z[:, 0])) / np.std(pan_vec) + np.mean(z[:, 0])
    X = model.inverse_transform(z)

    generated_image = {
        'red' : X[:, 0].reshape(1200, 1200), 
        'green' : X[:, 1].reshape(1200, 1200),
        'blue' : X[:, 2].reshape(1200, 1200),
        'nir' : X[:, 3].reshape(1200, 1200)
    }
    return generated_image

def nmf(images, loss='frobenius'):
    rescaled_ms = rescale_img(images)
    rgbn = to_rgbn(rescaled_ms).reshape((-1, 4))
    model = NMF(n_components=4, beta_loss=loss, solver='mu')

    z = model.fit_transform(rgbn)
    pan_vec = images['pan'].flatten()
    idx = np.argmax(np.corrcoef(z.T, pan_vec)[-1, :-1])

    z[:, idx] = ((pan_vec - np.mean(pan_vec)) * np.std(z[:, idx])) / np.std(pan_vec) + np.mean(z[:, idx])

    X = model.inverse_transform(z)

    generated_image = {
        'red' : X[:, 0].reshape(1200, 1200), 
        'green' : X[:, 1].reshape(1200, 1200),
        'blue' : X[:, 2].reshape(1200, 1200),
        'nir' : X[:, 3].reshape(1200, 1200)
    }
    return generated_image

def wavelet(images):
    #upsample
    rescaled_ms = rescale_img(images)
    
    pc = pywt.wavedec2(images['pan'], 'haar', level=2)
    generated_image = {}
    for band in rescaled_ms.keys():
        temp_dec = pywt.wavedec2(rescaled_ms[band] , 'haar', level=2)
        pc[0] = temp_dec[0]
        
        temp_rec = pywt.waverec2(pc, 'haar')
        temp_rec[temp_rec < 0] = 0
        temp_rec[temp_rec > 1] = 1
        generated_image[band] = temp_rec.reshape(1200, 1200)
    
    return generated_image

def gs(images):
    rescaled_ms = rescale_img(images)
    #remove means from u_hs
    rgbn = to_rgbn(rescaled_ms)
    band_mean = np.mean(rgbn, axis=(0, 1))
    normalized_rgbn = rgbn - band_mean
    
    #sintetic intensity
    I = np.mean(rgbn, axis=2, keepdims=True)
    I0 = I - np.mean(I)
    
    image_hr = (images['pan']-np.mean(images['pan']))*(np.std(I0, ddof=1)/np.std(images['pan'], ddof=1))+np.mean(I0)
    
    #computing coefficients
    g = []
    g.append(1)
    C = normalized_rgbn.shape[2]

    for i in range(C):
        normalized_band = normalized_rgbn[:, :, i]
        c = np.cov(np.reshape(I0, (-1,)), np.reshape(normalized_band, (-1,)), ddof=1)
        g.append(c[0,1]/np.var(I0))
    # (1, <pan_sim, MS_1>/<pan_sim, pan_sim>, <pan_sim, MS_2>/<pan_sim, pan_sim>, ... )
    g = np.array(g)
    
    delta = image_hr.reshape((1200, 1200, 1)) - I0 # pan - pan_sim
    deltam = np.tile(delta, (1, 1, C+1)) # (1200, 1200, 1) -> (1200, 1200, 5)
    
    # fusion
    # (pan_sim, MS_1, MS_2, ... )
    V = np.concatenate((I0, normalized_rgbn), axis=-1)
    
    g = np.expand_dims(g, 0) # (5) -> (1, 5)
    g = np.expand_dims(g, 0) # (1, 5) -> (1, 1, 5)
    g = np.tile(g, (1200, 1200, 1)) # (1, 1, 5) -> (1200, 1200, 5)
    
    V_hat = V + g * deltam
    
    X = V_hat[:, :, 1:]
    
    X = X - np.mean(X, axis=(0, 1)) + band_mean
    
    # adjustment
    X[X<0]=0
    X[X>1]=1
    
    generated_image = {
        'red' : X[:, :, 0].reshape(1200, 1200), 
        'green' : X[:, :, 1].reshape(1200, 1200),
        'blue' : X[:, :, 2].reshape(1200, 1200),
        'nir' : X[:, :, 3].reshape(1200, 1200)
    }
    
    return generated_image
