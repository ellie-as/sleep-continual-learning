import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import tensorflow as tf
from generative_model import VAE
from generative_tests import check_generative_recall
from tensorflow import keras
import numpy as np
import matplotlib.backends.backend_pdf
from generative_model import models_dict
import matplotlib
import random


def train_with_schedule(total_eps=100, num_cycles=20, start_fraction_rem=0.2, end_fraction_rem=0.8,
                        use_initial_weights=True, latent_dim=5, seed=0, inverted=True, lr=0.001, num=10,
                        continue_training=True):


    np.random.seed(seed)
    
    eps_per_cycle = int(total_eps / num_cycles)

    if inverted is True:
        mnist_train, mnist_test, fmnist_train, fmnist_test = prepare_datasets(split_by_digits=False, 
                                                                              split_by_inversion=True)
    else:
        mnist_train, mnist_test, fmnist_train, fmnist_test = prepare_datasets(split_by_digits=True, 
                                                                              split_by_inversion=False)

    if use_initial_weights is False:
        vae = train_mnist_vae(mnist_train, 'mnist', generative_epochs=25, learning_rate=0.001, latent_dim=latent_dim)
    else:
        print("Starting with saved weights:")

    encoder, decoder = models_dict['mnist'](latent_dim=latent_dim)
    vae = VAE(encoder, decoder, kl_weighting=1)
    if inverted is True:
        vae.encoder.load_weights("encoder_inv.h5")
        vae.decoder.load_weights("decoder_inv.h5")
    if inverted is False:
        vae.encoder.load_weights("encoder_non_inv.h5")
        vae.decoder.load_weights("decoder_non_inv.h5")
    opt = keras.optimizers.Adam(learning_rate=lr, jit_compile=False)
    vae.compile(optimizer=opt)
    
    m_err, f_err = plot_error_dists(vae, mnist_test, fmnist_test)
    check_generative_recall(vae, mnist_test, noise_level=0.0)
    
    sampled_digits = [sample_item(vae, latent_dim=latent_dim) for i in range(100)]
    sampled_digits = np.array(sampled_digits)
    show_samples(sampled_digits)
    
    mnist_errors = []
    fmnist_errors = []
    
    mnist_errors.append(np.mean(m_err))
    fmnist_errors.append(np.mean(f_err))
    
    # save copy of vae
    encoder_copy, decoder_copy = models_dict['mnist'](latent_dim=latent_dim)
    if inverted is True:
        encoder_copy.load_weights("encoder_inv.h5")
        decoder_copy.load_weights("decoder_inv.h5")
    if inverted is False:
        encoder_copy.load_weights("encoder_non_inv.h5")
        decoder_copy.load_weights("decoder_non_inv.h5")        
    old_vae = VAE(encoder_copy, decoder_copy, kl_weighting=1)

    
    for cycle in range(num_cycles):
        
        current_fraction_rem = start_fraction_rem + (end_fraction_rem - start_fraction_rem) * cycle / (num_cycles - 1)
        # Update the nrem_eps and rem_eps values for this cycle
        rem_eps = int(current_fraction_rem * eps_per_cycle)
        nrem_eps = eps_per_cycle - rem_eps
        
        # train for nrem_eps on real fmnist
        print("NREM phase")
        random_indices = np.random.choice(fmnist_train.shape[0], num, replace=False)
        fmnist_subset = fmnist_train[random_indices]
        vae.fit(fmnist_subset, epochs=nrem_eps, verbose=0, batch_size=1, shuffle=True)
        
        # train for rem_eps on sampled mnist
        print("REM phase")
        if continue_training is False:
            sampled_digits = [sample_item(old_vae, latent_dim=latent_dim) for i in range(num)]
        if continue_training is True:
            sampled_digits = [sample_item(vae, latent_dim=latent_dim) for i in range(num)]
            
        sampled_digits = np.array(sampled_digits)
        print("Show REM samples:")
        show_samples(sampled_digits)
        vae.fit(sampled_digits, epochs=rem_eps, verbose=0, batch_size=1, shuffle=True)
        
        # test reconstruction error of mnist_test and fmnist_test
        m_err, f_err = plot_error_dists(vae, mnist_test, fmnist_test)
        mnist_errors.append(np.mean(m_err))
        fmnist_errors.append(np.mean(f_err))

        check_generative_recall(vae, mnist_test, noise_level=0.0)
        check_generative_recall(vae, fmnist_test, noise_level=0.0)
        
    # Plot the reconstruction errors over time
    plt.figure()
    plt.plot(range(0, num_cycles + 1), mnist_errors, label='Dataset 1')
    plt.plot(range(0, num_cycles + 1), fmnist_errors, label='Dataset 2')
    plt.xlabel('Cycle')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    plt.show()
    
    return mnist_errors, fmnist_errors


def train_with_schedule_multiple_seeds(seeds, total_eps=100, num_cycles=20, start_fraction_rem=0.2, end_fraction_rem=0.8,
                                       use_initial_weights=True, latent_dim=5, inverted=True, lr=0.001, num=10,
                                       continue_training=True):
    
    pdf_path = "./outputs/total_eps={}_num_cycles={}_start_rem={}_end_rem={}_inverted={}_lr={}_num={}.pdf".format(str(total_eps),
                                                                                                                  str(num_cycles),
                                                                                                                  str(start_fraction_rem),
                                                                                                                  str(end_fraction_rem),
                                                                                                                  str(inverted),
                                                                                                                  str(lr),
                                                                                                                  str(num))

    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)
    
    eps_per_cycle = int(total_eps / num_cycles)
    fig = plot_schedule(num_cycles, total_eps, eps_per_cycle, start_fraction_rem, end_fraction_rem)
    pdf.savefig(fig, bbox_inches = "tight")
    
    results = [train_with_schedule(total_eps=total_eps,
                                   num_cycles=num_cycles,
                                   start_fraction_rem=start_fraction_rem,
                                   end_fraction_rem=end_fraction_rem,
                                   use_initial_weights=use_initial_weights,
                                   latent_dim = latent_dim,
                                   lr = lr,
                                   inverted=inverted,
                                   seed=s,
                                   num=num,
                                   continue_training=continue_training) for s in seeds]


    fig = plot_mean_recons_over_time(results, num_cycles)
    pdf.savefig(fig, bbox_inches = "tight")
        
    pdf.close()
    

def prepare_datasets(split_by_digits=False, split_by_inversion=True):
    # Load the MNIST dataset
    (x_train_orig, y_train), (x_test_orig, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the images and labels
    x_train_orig = np.expand_dims(x_train_orig, -1).astype("float32") / 255
    x_test_orig = np.expand_dims(x_test_orig, -1).astype("float32") / 255

    if split_by_digits:
        # Create a boolean mask to filter digits 0-4
        train_mask = np.isin(y_train, [0, 1, 2, 3, 4])
        test_mask = np.isin(y_test, [0, 1, 2, 3, 4])

        # Filter the images and labels using the masks
        x_train_filtered_0_4 = x_train_orig[train_mask]
        x_test_filtered_0_4 = x_test_orig[test_mask]

        x_train_filtered_5_9 = x_train_orig[~train_mask]
        x_test_filtered_5_9 = x_test_orig[~test_mask]

        return x_train_filtered_0_4, x_test_filtered_0_4, x_train_filtered_5_9, x_test_filtered_5_9
    
    if split_by_inversion:
        # Invert the color of each image
        x_train_inverted = np.subtract(1, x_train_orig)
        x_test_inverted = np.subtract(1, x_test_orig)

        return x_train_orig, x_test_orig, x_train_inverted, x_test_inverted


def plot_error_dists(vae, mnist_digits, fmnist_digits):
    encs = vae.encoder.predict(mnist_digits)
    decs = vae.decoder.predict(encs[0])
    mnist_recons = tf.reduce_sum(keras.losses.mean_absolute_error(mnist_digits, decs), axis=(1, 2)).numpy().tolist()

    f_encs = vae.encoder.predict(fmnist_digits)
    f_decs = vae.decoder.predict(f_encs[0])
    fmnist_recons = tf.reduce_sum(keras.losses.mean_absolute_error(fmnist_digits, f_decs), axis=(1, 2)).numpy().tolist()

    matplotlib.rcParams.update({'font.size': 14})
    plt.rcParams.update({"figure.figsize": (5, 3)})

    fig = plt.figure()

    n, bins, patches = plt.hist(mnist_recons, 25, density=True, facecolor='black', alpha=0.5, label='Dataset 1')
    n, bins, patches = plt.hist(fmnist_recons, 25, density=True, facecolor='blue', alpha=0.5, label='Dataset 2')
    plt.xlabel('Reconstruction error')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()
    
    return mnist_recons, fmnist_recons


def plot_mean_recons_over_time(results, NUM_CYCLES):
    # Calculate the mean and SEM for each cycle across the runs
    mnist_errors = np.mean([res[0] for res in results], axis=0)
    fmnist_errors = np.mean([res[1] for res in results], axis=0)
    mnist_sem = np.std([res[0] for res in results], axis=0) 
    fmnist_sem = np.std([res[1] for res in results], axis=0) 
    
    # Plot the mean reconstruction errors with error bars
    fig = plt.figure()
    plt.errorbar(range(0, NUM_CYCLES + 1), mnist_errors, yerr=mnist_sem, label='Dataset 1', capsize=4)
    plt.errorbar(range(0, NUM_CYCLES + 1), fmnist_errors, yerr=fmnist_sem, label='Dataset 2', capsize=4)
    plt.xlabel('Cycle')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    plt.show()
    return fig
    

def plot_schedule(NUM_CYCLES, TOTAL_EPS, eps_per_cycle, starting_fraction_rem, ending_fraction_rem):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 2)

    nrem_starts = list(range(0, TOTAL_EPS, int(eps_per_cycle)))
    
    for cycle in range(NUM_CYCLES):
        # Calculate the current fraction of REM sleep for this cycle
        current_fraction_rem = starting_fraction_rem + (ending_fraction_rem - starting_fraction_rem) * cycle / (NUM_CYCLES - 1)

        # Update the nrem_eps and rem_eps values for this cycle
        rem_eps = int(current_fraction_rem * eps_per_cycle)
        nrem_eps = eps_per_cycle - rem_eps

        rem_start = nrem_starts[cycle] + nrem_eps
        
        ax.broken_barh([(rem_start, rem_eps)],
                       (10, 9),
                       facecolors='tab:blue')
        ax.broken_barh([(nrem_starts[cycle], nrem_eps)],
                       (20, 9),
                       facecolors='tab:red')

    ax.set_xlabel('Epochs')
    ax.set_yticks([15, 25], labels=['REM', 'NREM'])
    ax.grid(True)

    plt.show()
    return fig


def train_mnist_vae(train_data, dataset, generative_epochs=50, latent_dim=20, kl_weighting=1, learning_rate=0.001):
    encoder, decoder = models_dict[dataset](latent_dim=latent_dim)
    vae = VAE(encoder, decoder, kl_weighting)
    opt = keras.optimizers.Adam(learning_rate=learning_rate, jit_compile=False)
    vae.compile(optimizer=opt)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = vae.fit(train_data, epochs=generative_epochs, verbose=1,
                      batch_size=32, shuffle=True, callbacks=[early_stopping])
    vae.encoder.save('encoder.h5')
    vae.decoder.save('decoder.h5')
    return vae


def sample_item(vae, latent_dim=20):
    z_sample = np.array([[random.gauss(0, 1) for i in range(latent_dim)]])
    x_decoded = vae.decoder.predict(z_sample, verbose=False)
    digit = x_decoded[0]

    return digit


def show_samples(sampled_digits):
    num_to_show = min(10, sampled_digits.shape[0])
    plt.figure()
    f, axarr = plt.subplots(1, num_to_show)
    f.set_size_inches(10, 2)

    for i in range(num_to_show):
        axarr[i].imshow(sampled_digits[i].reshape(28, 28))
        axarr[i].axis('off')

    plt.show()

