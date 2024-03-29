### Sleep simulations

The code at https://github.com/ellie-as/generative-consolidation models consolidation as teacher-student learning, where the teacher is the hippocampus and the student is the neocortical generative (predictive) model of the world. In our paper, we show that catastrophic forgetting is alleviated by generative replay (in this case training the generative model on its own samples).

Papers such as Singh et al. (2022) suggest remote memories are replayed by neocortex in REM and recent memories are replayed by HPC in NREM, with this combination preventing catastrophic interference of recent memories with remote memories. This follows on from papers like Norman et al. (2005), which suggests REM involves 'autonomous memory rehearsal'. This code simulates how different sleep schedules affect this, modelling NREM as replay of recent data, and REM simply as training on data generated by the existing generative model (even though this is obviously a huge simplification). In other words, this extends the ideas in the papers above to generative models.

The simulations start with a generative model (variational autoencoder) trained on dataset 1, such that it can reconstruct images from dataset 1 ('memories') from partial inputs. The code then trains a model according to the schedule specified. For each NREM stage shown, the model is trained on dataset 2 for a certain duration. For each REM stage, samples are taken from the generative model, which is then trained on these samples.

Experiments can be run as shown in the 'Sleep scheduler.ipynb'notebook, which save outputs to the 'outputs' folder.

#### Variables:

The arguments of train_with_scheduler_multiple_seeds are:
* seeds: A list of integer seeds to use for random number generation during training. Each seed corresponds to a separate run of the experiment with the same set of parameters.
* total_eps: Total number of epochs to train the VAE across all cycles. An epoch is a complete iteration through the dataset during training.
* num_cycles: The number of cycles to train the VAE. Each cycle consists of a non-REM phase and a REM phase.
* start_fraction_rem: The fraction of epochs per cycle allocated to the REM phase at the beginning of the training.
* end_fraction_rem: The fraction of epochs per cycle allocated to the REM phase at the end of the training.
* use_initial_weights: If True, the VAE will start with previously saved weights; otherwise, it will be trained on the MNIST dataset before proceeding with the experiment.
* latent_dim: The dimensionality of the latent space in the VAE.
* seed: A single integer seed for random number generation during training. This seed is used for initializing the first run of the experiment.
* inverted: If True, the datasets will be split by inversion (MNIST and inverted MNIST); otherwise, they will be split by digits (MNIST 0-4 and 5-9).
* lr: The learning rate for the optimizer during training.
* num: The number of samples to use for training during each phase of a cycle.
* continue_training: If True, the VAE will use its own generated samples for the REM phase; otherwise, it will use samples generated by the previous VAE version (the old_vae model).