import imageio
import os
import torch
import models.ae as ae
import models.vae as vae
import sys
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from skimage.transform import resize

def openImage(name):
    """Open an image file and return a handle to it."""
    # Path: RunSweeping.py
    try:
        return imageio.imread(name)
    except IOError:
        print("Cannot open", name)
        sys.exit(1)
        
def loadModel(name="ae"):
    """
        Loads the model with its precomputed parameters.
    """

    weights_dir = 'weights'

    if name.lower() == "ae":
        weights_path = os.path.join(weights_dir, 'ae_best.pth.tar')
        model = ae.AE(32)
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))['state_dict'])
    
    elif name.lower() == "vae":
        weights_path = os.path.join(weights_dir, 'vae_best.pth.tar')
        model = vae.BetaVAE(32)
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))['state_dict'])

    elif name.lower() == "tbh":
        weights_path = os.path.join(weights_dir, '64bit/model')
        model = tf.keras.models.load_model(weights_path)
        model.training = False
        pretrained_net = tf.keras.Sequential([
            hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5", input_shape=(224,224,3))
        ])
        model = (model, pretrained_net)
    
    else:
        raise NotImplementedError(("Model '{}' is not a valid model. " +
            "Argument 'name' must be in ['ae', 'vae', 'tbh'].").format(name))

    return model

def getEncoding(descr, patch):
    """
        Computes the encoding of a given patch with a given descriptor.
    """
    
    if isinstance(descr, ae.AE) or isinstance(descr, vae.BetaVAE):
        patch = np.array(resize(patch, (65, 65)))
        patch = patch / 255.0
        patch = np.expand_dims(np.expand_dims(patch, axis=0), axis=0)
        patch = torch.from_numpy(patch).float()

        if isinstance(descr, vae.BetaVAE):
            patch_encoding, _, _ = descr.encode(patch)

        else:
            patch_encoding = descr.encode(patch)

        patch_encoding = patch_encoding.detach().numpy()
        patch_encoding = patch_encoding.reshape(patch_encoding.shape[0], np.product(patch_encoding.shape[1:]))

    elif isinstance(descr, tuple) and len(descr) == 2:
        sample = np.reshape(resize(patch, (224, 224)),(1,224,224,1))
        sample = np.repeat(sample,3,3)
        sample = np.expand_dims(descr[1].predict(sample, verbose=False).flatten(),0)
        
        fc1 = descr[0].encoder.fc_1(sample)
        fc2 = descr[0].encoder.fc_2_1(fc1)
        
        mean, logvar = tf.split(fc2, num_or_size_splits=2, axis=1)
        patch_encoding = ((tf.sign(mean)+1)/2).numpy()[0], None

    else:
        raise NotImplementedError(("Argument 'descr' is not a valid descriptor" +
                "Argument 'descr' must be of type 'AE', 'BetaVAE' or 'TBH', but type {} was given.").format(descr.__class__))

    return patch_encoding[0]

if __name__ == "__main__":

    model = loadModel("ae")

    images = os.listdir("images")

    encodings = []
    queries = []
    n_queries = 20
    dim = 100
    stride = 50
    for index, image in enumerate(images):
        print(index)
        img = openImage("images/" + image)
        for i in range(0, img.shape[0]-dim+1, stride):
            for j in range(0, img.shape[1]-dim+1, stride):
                patch = resize(img[i:i+dim, j:j+dim], (64, 64))
                for _ in range(4):
                    patch = np.rot90(patch)
                    encoding = [index, i, j] + list(getEncoding(model, patch))
                    
                    if index <= n_queries and np.sum(openImage("labels/" + image)[i:i+dim, j:j+dim] == 1)/dim**2 > 0.5:
                        queries += [encoding]
                    elif index > n_queries:
                        encodings += [encoding]
                

    # store encodings in csv file
    np.savetxt("encodings_ae.csv", np.array(encodings), delimiter=";")
    np.savetxt("queries_ae.csv", np.array(queries), delimiter=";")