parameters = {
    'learning_rate': 0.01,
    'display_step': 5,
    'dilations': sum(([1, 2, 4, 8, 16, 32, 64, 128, 256, 512] for i in range(3)), []),
    # The receptive field size for a dilation set is the number of 1-512 sequences times 1023 + 1.
    # For one second receptive field approximately 44 sequences are needed.
    # Convolution sizes.
    'dense_channels': 32,
    'dilation_channels': 32,
    'quantization_channels': 256,
    'filter_width': 2,
    'sample_length': (1023 * 3 + 1) * 2
}
