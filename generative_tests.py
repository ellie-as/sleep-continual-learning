import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from utils import display, noise, prepare_data
from generative_model import label_classifier
from math import sqrt
from tensorflow import keras
from tensorflow.keras.utils import to_categorical


def check_generative_recall(vae, test_data, noise_level=0.15):
    test_data = noise(test_data, noise_factor=noise_level)
    latents = vae.encoder.predict(test_data)
    predictions = vae.decoder.predict(latents[0])
    fig = display(test_data, predictions, title='Inputs and outputs for VAE')
    return fig


def plot_history(history, decoding_history, titles=False):
    recon_loss_values = history.history['reconstruction_loss']
    decoding_acc_values = decoding_history.decoding_history
    epochs = range(1, len(recon_loss_values)+1)

    fig, ax = plt.subplots(figsize=(3,3))
    ax2 = ax.twinx()
    ax2.set_ylim(0, 1)
    ax.set_ylabel("Reconstruction error")
    ax2.set_ylabel("Decoding accuracy")

    ax.plot(epochs, recon_loss_values, label='Reconstruction Error', color='red')
    ax2.plot(epochs, decoding_acc_values, label='Decoding Accuracy', color='blue')

    if titles is True:
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
        plt.title('Reconstruction error and decoding accuracy over time')

    ax.set_xlabel('Epoch')
    plt.show()
    return fig


def interpolate_ims(latents, vae, first, second):
    encoded_imgs = latents[0]
    enc1 = encoded_imgs[first:first+1]
    enc2 = encoded_imgs[second:second+1]

    linfit = interp1d([1,10], np.vstack([enc1, enc2]), axis=0)

    fig = plt.figure(figsize=(20, 5))

    for j in range(10):
        ax = plt.subplot(1, 10, j+1)
        decoded_imgs = vae.decoder.predict(np.array(linfit(j+1)).reshape(1,encoded_imgs.shape[1]))
        ax.imshow(decoded_imgs[0])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.suptitle('Interpolation between items')
    plt.show()
    return fig


def vector_arithmetic(latents, vae, first, second, third):
    encoded_imgs = latents[0]
    enc1 = encoded_imgs[first:first+1]
    enc2 = encoded_imgs[second:second+1]
    enc3 = encoded_imgs[third:third+1]

    fig, axs = plt.subplots(1,4)
    axs[0].imshow(vae.decoder.predict(enc1.reshape(1,encoded_imgs.shape[1]))[0])
    axs[0].axis('off')
    axs[1].imshow(vae.decoder.predict(enc2.reshape(1,encoded_imgs.shape[1]))[0])
    axs[1].axis('off')
    axs[2].imshow(vae.decoder.predict(enc3.reshape(1,encoded_imgs.shape[1]))[0])
    axs[2].axis('off')
    # enc1-enc2=enc3-enc4 -> enc4=enc3+enc2-enc1
    res = - enc1 + enc2 + enc3
    axs[3].imshow(vae.decoder.predict(res.reshape(1,encoded_imgs.shape[1]))[0])
    axs[3].axis('off')
    fig.suptitle('Vector arithmetic')
    plt.show()
    return fig


def add_vae_self_sampling(vae, num_to_sample, test_data, scale=1.0):
    latents = vae.encoder.predict(test_data)
    pca = PCA(n_components=2)
    pca.fit(latents[0])
    
    # display a n*n 2D manifold of digits
    n = int(sqrt(num_to_sample))
    digit_size = 28
    scale = scale
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    samples = []
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(pca.inverse_transform(z_sample))
            digit = x_decoded[0].reshape(digit_size, digit_size)
            samples.append(digit)
    return samples

    
def plot_latent_space(vae, noisy_test_data, n=20, figsize=15):
    
    latents = vae.encoder.predict(noisy_test_data)
    pca = PCA(n_components=2)
    pca.fit(latents[0])
    
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(pca.inverse_transform(z_sample))
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    fig = plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.title('Latent space of the VAE, projected into 2D with PCA')
    plt.show()
    
    return fig


def plot_latent_space_with_labels(latents, labels, titles=False):
    np.random.seed(1)
    fig = plt.figure(figsize=(4, 4))

    embedded = TSNE(n_components=2, init='pca').fit_transform(latents[0][0:800])
    x = [x[0] for x in embedded]
    y = [x[1] for x in embedded]

    plt.scatter(x, y, c=labels[0:800], alpha=0.5, cmap=plt.cm.plasma)
    if titles is True:
        plt.title('Latent space in 2D, colour-coded by label')
    plt.show()
    return fig
