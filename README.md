An implementation and demonstration of Google WaveNet.

This application learns the given sound file and
the resultant model can be used to synthesize sounds.

It is a work in progress, and will be used in Cybercom #HackingGreat
event in November 2016.

At the moment of writing CuDNN does not have applicable GPU acceleration functions to accelerate
strided convolutions, so we have to use a trick to swap time dimensions to batch dimension and back for the normal 1-stride convolution to use GPU acceleration.

The application takes red.wav as input file and trains the WaveNet for samples taken from that file.

Running process.py trains a model, and saves the best model found so far into the file `sound-model-best` and `sound-model-best.meta`.

After having a trained model, you can use it to generate random sounds, by running `generate.py`. It saves the generated sound to Octave file `sound.mat`,
and the probability distribution used for generation in `image.mat`.

You can draw the probability distribution in Octave using: `load("image.mat");imagesc(i');`

You can play the generated sound in Octave using: `load("sound.mat");soundsc(s, 48000);`
