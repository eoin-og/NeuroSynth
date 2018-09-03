# NeuroSynth
Simple synthesizer based on a neural network. Network uses adversarial training to learn attack/sustain/decay transformation which can be applied to an input waveform. 

## Overview


## Results

The networks were trained using three waveforms (sin, sawtooth, square) with three values for attack (None, 20% of waveform, 40% of waveform), three for release (None, 20% of waveform, 40% of waveform), and three for sustain (amplitude of 1, 0.7, or 0.4). Here are the some of the results optained:

Sine wave with attack: 40%, sustain: 0.4, release: 40%
![sine wave 222](https://github.com/eoin-og/NeuroSynth/blob/master/results/sine222.png)

Squre wave with attack: 20%, sustain: 0.7, release: 0%
![square wave 110](https://github.com/eoin-og/NeuroSynth/blob/master/results/square110.png)

Sawtooth wave with attack: 0%, sustain: 1.0, release: 0% (no transformation)
![saw wave 000](https://github.com/eoin-og/NeuroSynth/blob/master/results/saw000.png)

To test how well the network generalises I tired an input sample that was much noisier than anything it was trained on:
Noisey saw wave with attack: 40%, sustain: 0.4, release: 40%
![noisey sin 222](https://github.com/eoin-og/NeuroSynth/blob/master/results/noisey_sine.png)

Finally I tried giving it a waveform it had never seen before, an inverted sawtooth. The results here are not perfect but I think pretty good condisering it was only trained on three different wave forms to begin with. We can see the rise and fall of the attack/sustain/release is pretty accurate and in most cases the general character of the wave is still preserved (sharp rise followed by a slow fall)
Inverted sawtooth wave with attack: 40%, sustain: 0.4, release: 40%
![inv saw wave 222](https://github.com/eoin-og/NeuroSynth/blob/master/results/invsaw222.png)

## Files
* `main.py` - function that parses arguments, calls the trainer function.
* `gan.py` - Trainer function for both networks.
* `models.py` - Model class for generator and discriminator.
* `utils.py` - A few helper functions.
