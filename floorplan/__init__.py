"""Generative floorplan synthesis on CubiCasa5K -- final-review code base.

Every model in this package is written from scratch in NumPy (forward pass,
hand-derived backward pass and the Adam optimiser). Modules:

    config    paths, seeds and the shared hyper-parameters
    data      official split files, PNG -> 32x32 wall map, cache, corruption
    layers    im2col convolution, ReLU, upsampling, linear layer, Adam
    losses    weighted binary cross-entropy and the evaluation metrics
    models    DAE (denoising autoencoder), VAE (variational autoencoder),
              GNN (GraphSAGE on a patch graph)
    trainer   mini-batch training loop with best-checkpoint saving
    checkpoint  save / load model weights (.npz) with their metadata
    analysis  evaluation, error analysis and generation statistics
    plots     every figure used in the report and the slides
"""
