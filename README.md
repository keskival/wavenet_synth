An implementation and demonstration of Google WaveNet.

This application learns the given sound file and
the resultant model can be used to synthesize sounds.

It is a work in progress, and will be used in Cybercom #HackingGreat
event in November 2016.

At the moment of writing CuDNN does not have applicable GPU acceleration functions to accelerate
strided convolutions, so we have to use a trick to swap time dimensions to batch dimension and back for the normal 1-stride convolution to use GPU acceleration.

The application takes `corpus.wav` as the input file and `test.wav` as the test set file and trains the WaveNet with that corpus using
different amplitude and noise distortions.

Running process.py trains a model, and saves the best model found so far into the file `sound-model-best` and `sound-model-best.meta`.

After having a trained model, you can use it to generate random sounds, by running `generate.py`. It saves the generated sound to Octave file `sound.mat`,
and the probability distribution used for generation in `image.mat`. For generating really long samples,
you might want to disable the image.mat generation by commenting out the relevant parts in `generate.py`.

You can draw the probability distribution and play intermediate training results in Octave using the Octave script file: `plot_training.m`

You can show the generated results and play the generated sound in Octave using the Octave script file: `plot_generation.m`
