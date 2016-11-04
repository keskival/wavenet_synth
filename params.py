parameters = {
    'learning_rate': 0.04,
    'display_step': 10,
    'input_noise': 0.005,
    'input_salt_and_pepper_noise': 0.005,
    'amplitude_plusminus_factor': 0.005,
    'noise': 0.00, #0.01,
    'dilations': sum(([1, 2, 4, 8, 16, 32, 64, 128, 256, 512] for i in range(4)), []), # 1 is too small, 5 used by ibab.
    # The receptive field size for a dilation set is the number of 1-512 sequences times 1023 + 1.
    # Convolution sizes.
    'dense_channels': 32, # 64 is ok, 128 is too large, 32 used by ibab.
    'dilation_channels': 32, # 64 is ok, 128 is too large
    'skip_channels': 32,
    'preoutput_channels': 32,
    'quantization_channels': 256,
    'filter_width': 2,
    'sample_length': 1023 * 4 + 1,
    'training_length': 1024 * 512,
    'clip_gradients': 1000.0,
    'temperature': 0.7
}
