# Calibration of Bob

As we saw before, two parameters are needed for the parameters estimation step: the efficiency of the detector {math}`\eta` and the electronic noise of the detector {math}`V_{el}`. However, this last quantity is normalised to the shot noise, so we actually need to calibrate three quantities.

Also note that the electronic noise samples and the electronic and shot noise samples have to follow the same treatment at the quantum data, in particular the DSP, which means that those quantity needs to be estimated before the DSP step.

## Eta

A photodiode is a device that convert photons into electrons. The perfect behaviour would be that for each incoming photon, one electron is emitted. In practice, this is not the case, and we have a finite efficiency {math}`\eta` such that 

```{math}
n_{e} = \eta \cdot n_{ph}
```

where {math}`n_e` is the number of emitted electrons, {math}`n_{ph}` the number of incoming photons and {math}`0 \leq \eta \leq 1`. Using the fact that the current through the photodiode is given by

```{math}
I = n_{e}\cdot e
```

where {math}`e` is the elementary charge and the input optical power is given by

```{math}
P = n_{ph}\cdot \frac{hc}{\lambda}
```

where {math}`h` is Planck's constant, {math}`c` the speed of light and {math}`\lambda` the wavelength, we get the relation

```{math}
\eta = \frac{hc}{\lambda e}\frac{I}{P} = \frac{hc}{\lambda e}\mathcal{R}
```

where {math}`\mathcal{R} = \frac{I}{P}` is called the responsivity of the photodiode and has SI units of A/W.

At {math}`\lambda=1550`nm, we have the following relation

```{math}
\eta \simeq \frac{\mathcal{R}}{1.25 A/W}
```

This means, that the efficiency of the photodiode can be deduced from the ratio of the current to the optical power.

However, we are not working with a single photodiode but with an interferometric detector. The idea is then to say that the responsitivty is the ratio between the sum of all photocurrents, to the optical power of the signal before the interferometer.

For instance, we the example of an homodyne detector, with 1 beam splitter and 1 balanced detector with photocurrents {math}`{I_+}` and {math}`I_-`, if the optical power of the signal at the entrance of Bob is {math}`P_s`, then the responsivity is 

```{math}
\mathcal{R} = \frac{I_+ + I_-}{P_s}
```

This means that to get the efficiency of the whole detector, we need to measure two quantities: the sum of all the photocurrents and the input optical power on the signal side.

This can be done using the following setup:


```{figure} ../_static/calibration_eta.png
---
align: center
---
Proposition of Calibration setup
```

Note that in the proposed scheme, the local oscillator is not plugged.

The input optical power is measured using a beam splitter and an optical power meter. The VOA allows to change the input optical power to gain in precision. The photocurrents are acquired using amperemeters.

A script in `qosst-bob` is provided to do this characterisation (assuming the output of the monitoring outputs are actually voltage outputs). The script can be called using the `qosst-bob-tools eta-voltage` command. The configuration will be done interactively in the script. More information on this script can be found {doc}`here <../cli/documentation>`. The script `qosst-bob-tools eta-current` can also be used if the photocurrents are directly measured from amperemeters.

## Electronic noise

Electronic noise samples must be acquired in order to estimate the electronic noise value. This must be done beforehand, with the signal input switched off and the local oscillator off.

When everything is off, it's possible to use the GUI to acquire those samples. Once the configuration has been loaded and the hardware initialized, the "acquire electronic noise samples" button can be clicked to acquire the samples. Once acquired, the samples can be saved using the "save electronic noise" button. This will save the electronic noise container ({py:class}`qosst_bob.data.ElectronicNoise`) to the location set in the configuration file, in the parameter `bob.electronic_noise.path` (default is `electronic_noise.qosst`).

The samples can then be loaded when using the GUI, or scripts. The samples will be loaded also from the location set in `bob.electronic_noise.path`.

The electronic noise will usually not changed too much, unless the detector is changed. The temperature will have an influence in the electronic noise but the effect should not be too big in a temperature-controlled room.

## Electronic and shot noise

The shot noise also needs to be calibrated as it is needed for normalisation. In practice we acquire electronic and shot noise samples and the variance of the electronic noise is subtracted to get the variance of the sole shot noise.

However, in contrast with the electronic noise, the shot noise can have quicker variations, especially since it is proportional to the power of the local oscillator, which might vary. For this reason, three methods of calibration are proposed.

In any case the calibration of the shot noise must be done with the local oscillator on and the signal input switched off.

### One time calibration

The one time calibration of the electronic and shot noise is very similar to the one for the electronic noise. Once in the GUI, with the local oscillator on, and the signal input switched off, the samples can be acquire dby clicking on the "acquire electronic and shot noise samples", can be saved with the button "save electronic and shot noise" at the location set in `bob.electronic_shot_noise.path` and can then be loaded in the GUI or scripts, at the same path.

```{warning}
This method is strongly discouraged as it will not give good excess noise results.
```

### Slow automatic calibration

The second option is to calibrate automatically the electronic and shot noise before each frame, with a different acquisition. This means that before each frame, the client will automatically switch off the input signal, make an acquisition and switch on the input signal again.

This can be set by choosing `bob.automatic_shot_noise_calibration = true` in the configuration file.

```{note}

This method is better than the first one but will still suffer from a few seconds between the two acquisitions, and it's possible to do better, as shown in the next section.
```

### Fast automatic calibration

The final method for calibrating the shot noise is also automatic, and the shot noise is also calibrated for each frame, but this time the shot noise is calibrated in the same acquisition as the quantum data.

This is done in the following way:

1. Switch off the signal input
2. Start the acquisition
3. Wait some amount of time {math}`t_{switch}`
4. Switch on the signal input
5. Send the trigger to Alice

With these method, the first part of the acquisition, until {math}`t_{switch}` can be considered shot noise because the switch had not happened yet. Using this method the time between the end of calibration of shot noise and the beginning of the CV-QKD frame is roughly the classical communication time (in the order of tens of milliseconds).

This calibration method can be enabled by setting a non-zero value in `bob.switch.switching_time`, which will represent {math}`t_{switch}` in seconds.

```{warning}
When using this method, the acquisition time has to be seen accordingly to take into account the additional time for the shot noise calibration.
```