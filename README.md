# NeuroSynth
Simple synthesizer based on a neural network. Network uses adversarial training to learn attack/sustain/decay transformation which can be applied to an input waveform. 

## Overview
Our generator is trained adversarially using the wgan-gp algorithm. Our discriminator(or critic) network takes as input the original waveform, the transformed waveform, as well as the timestamp coordinates (a list of integers 0-length of sequence). Both networks are fully convolutional, with a couple of ResNet blocks in the generator network.

The inclusion of the original waveform means that the discriminator will learn the transformation between the two waveforms (instead of just learning whether a sample is real or not). This discourages the generator from just learning off a couple of realistic outputs and disregarding it's input. 

The inclusion of the timestamp helps the network learn temporal features (attack and decay). This idea was inspired by [this blog post](https://eng.uber.com/coordconv/).

The discriminator has two output layers, one for whether it believes the source of the input was real or generated, and one for what class it believes the input belongs to. The class output layer allows us to learn to generate conditional examples. The loss function for the discriminator is the source loss on real examples, the class loss on real examples, the source loss on fake examples, and the gradient penalty loss.

Our Generator takes the original waveform, timestamp coordinates, and desired class outputs as input and produces transformed waveforms as output. The loss function for the generator is the source loss and class of the discriminator.

## Results

The networks were trained using three waveforms (sin, sawtooth, square) with three values for attack (None, 20% of waveform, 40% of waveform), three for release (None, 20% of waveform, 40% of waveform), and three for sustain (amplitude of 1, 0.7, or 0.4). Here are the some of the results optained:

Sine wave with attack: 40%, sustain: 0.4, release: 40%
![sine wave 222](https://github.com/eoin-og/NeuroSynth/blob/master/results/sine222.png)

Squre wave with attack: 20%, sustain: 0.7, release: 0%
![square wave 110](https://github.com/eoin-og/NeuroSynth/blob/master/results/square110.png)

Sawtooth wave with attack: 0%, sustain: 1.0, release: 0% (no transformation)
![saw wave 000](https://github.com/eoin-og/NeuroSynth/blob/master/results/saw000.png)

To test how well the network generalises I tired an input sample that was much noisier than anything it was trained on:
Noisey sine wave with attack: 40%, sustain: 0.4, release: 40%
![noisey sin 222](https://github.com/eoin-og/NeuroSynth/blob/master/results/noisey_sine.png)

Finally I tried giving it a waveform it had never seen before, an inverted sawtooth. The results here are not perfect but I think pretty good condisering it was only trained on three different wave forms to begin with. We can see the rise and fall of the attack/sustain/release is pretty accurate and in most cases the general character of the wave is still preserved (sharp rise followed by a slow fall)
Inverted sawtooth wave with attack: 40%, sustain: 0.4, release: 40%
![inv saw wave 222](https://github.com/eoin-og/NeuroSynth/blob/master/results/invsaw222.png)

## Files
* `main.py` - function that parses arguments, calls the trainer function.
* `gan.py` - Trainer function for both networks.
* `models.py` - Model class for generator and discriminator.
* `utils.py` - A few helper functions.
