parameters = {
    'learning_rate': 0.02,
    'display_step': 10,
    'input_noise': 0.02,
    'input_salt_and_pepper_noise': 0.02,
    'amplitude_plusminus_factor': 0.2,
    'noise': 0.01,
    'dilations': sum(([1, 2, 4, 8, 16, 32, 64, 128, 256, 512] for i in range(1)), []),
    # The receptive field size for a dilation set is the number of 1-512 sequences times 1023 + 1.
    # For one second receptive field approximately 44 sequences are needed.
    # Convolution sizes.
    'dense_channels': 8,
    'dilation_channels': 8,
    'quantization_channels': 256,
    'filter_width': 2,
    'sample_length': 1023 * 1 + 1,
    'training_length': (1023 * 1 + 1) * 512,
    'clip_gradients': 20.0
}
