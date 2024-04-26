# GUI



Here is a picture of the GUI:

```{figure} ../_static/gui_numbered.png
---
align: center
---
Image of the GUI, with numbered sections
```

1. Configuration loading and hardware initialization
2. Actions
3. Noises and exports
4. Figures
5. Parameters estimation results
6. Logs
7. Configuration parameters

## Configuration loading and hardware initialization

The configuration file can be chosen either by directly typing the path in the text field, or by using the Browse button, that will open an explorer. When the file has been selected, the configuration can be loaded by clicking on the "Read configuration" button.

The text field will become gray and cannot be modified without exiting. The "Init hardware" button will become available.

The "Read configuration" button does not become disabled. Indeed it is possible to read again the configuration file to update the parameters without restarting the GUI.

```{warning}
If you reload the configuration after clicking on the `Init hardware` button, please note that any modification on the hardware will have no effect.
```

Clicking on the "Init hardware" button will open the hardware (ADC, and switch), and perform the basic configuration.

```{warning}
The GUI is never enabling the laser by itself. When using the GUI, the laser should be set manually.
```

## Actions

There are 8 possible actions that are described below.

### Connect

This action will initialize the {py:class}`~qosst_core.control_protocol.sockets.QOSSTClient` socket on connect to Alice's socket.

```{note}
Make sure that Alice is connected before this step.
```

### Identification

This button will perform the identification procedure, which in practice, initialize the authentication.

### Initialization

This button will perform the initialization for a new frame. In particular, it will generate a UUID for the frame and send to Alice.

```{note}
While in theory, this step should be performed each time before starting a new frame. In practice however, the software won't complain starting a new quantum information exchange without having re-initialized the frame.
```

### QIE

This button will start the Quantum Information Exchange (QIE) procedure. This step will involve sending a message to Alice to get ready, and when she is ready, start the acquisition and trigger Alice.

At the end of this step, the figures are plotted if the autoplot is enabled.

### DSP

This button will apply the DSP on the quantum data, on the electronic noise data and on the electronic and shot noise data. It will also ask Alice to send a portion of its symbols.

At the end of this step, the recovered symbols, the electronic noise symbols and the electronic and shot noise symbols are available.

At the end of this step, the figures are plotted if the autoplot is enabled.

### Parameters estimation

This button will perform the parameters estimation step after getting the number of photon at Alice's output by a request to Alice.

At the end of this step, the section 5 of the GUI, with the parameters estimation is updated with the values.

### Error correction

This step is not yet implemented.

### Privacy amplification

This step is not yet implemented.

## Noises and exports

### Noises

This tab has button to acquire, load and save the electronic noise and the electronic and shot noise. You can refer to the {doc}`documentation on Bob calibration <./calibration>` for more information.

### Exports

This section has button to export the electronic noise, the electronic and shot noise and the signal.

```{note}
One might ask what is the different between the export and save features for the noises. When saving, the save is operated through QOSST data containers (see {doc}`here <../api/data>` and {external+qosst-core:doc}`understanding/data` for more information) and at a location set in the configuration file, meaning that saving again will in general, result in overwriting the date. When exporting, the numpy array is saved using the numpy `save` function in a directory set in the configuration (`config.bob.export_directory`) with a unique timestamp. The save method is more useful for use inside the QOSST ecosystem, and the export for outside QOSST.
```

## Figures

The figure area allows for direct feedback from the experiment. 6 figures are available:

* Temporal: temporal representation of the acquired data. It also displays the results of frame synchronisation;
* Frequential: Power Spectral Density of the acquire data, the electronic noise and the electronic and shot noise;
* FFT: Fourier Transform of the acquired data;
* Tone: 2D heatmap of the recovered tone (one of them);
* Uncorrected: 2D heatmap of the data before relative and global phase recovery;
* Recovered: 2D heatmap of the data after the end of the DSP.

The first 3 are available after the QIE step and the last 3 are available after the DSP step.

After the QIE step and the DSP step, the GUI performs the autoplot for the selected figures. By default autoplot is enabled for temporal, frequential and tone.

It is also possible to manually plot one figure by going to the tab and click on "Plot ...". Finally it is also possible to export the figure with "Export ..."

It is also possible to choose a plotting style from the one listed below:

```{program-output} python3 -c "from qosst_bob.gui.plot_utils import get_styles; print('\n'.join(['* default'] + ['* ' + x for x in get_styles()]))"

```

## Parameters estimation results

After the parameters estimation step, the results are printed in this section, including 

* Efficiency of the detector (actually not a result);
* The number of photons estimated by Alice;
* The total transmittance, *i.e.* {math}`\eta T`;
* The transmittance, *i.e.* {math}`T`;
* The equivalent distance in km of the attenuation with an attenuation coefficient of 0.2 dB/km;
* The shot noise variance (in arbitrary units);
* The electronic noise variance {math}`V_{el}` in Shot Noise Units;
* The excess noise at Bob side {math}`\xi_B` in Shot Noise Units;
* The excess noise at Alice side {math}`\xi` in Shot Noise Units;
* The secret key rate in bits/s.

## Logs

The logs section will display the same logs as in the console. The verbosity of the logs will be the one passed through the command line with the following relation:

* No `-v`: Only print errors and critical errors;
* `-v`: Same as above with warnings added;
* `-vv`: Same as above with info added;
* `-vvv`: all logs.

```{warning}
Due to the limitation of blocking calls, this window is only updated at the end of an action. However the logs in the console are displaced in real time, so it's always better to monitor the logs directly from the console.
```

## Configuration parameters

When the configuration is loaded (or reloaded), this section of tabs is updated to display the following information:

* QI (Quantum frame)
  * Number of symbols;
  * Frequency shift of the quantum data;
  * Symbol rate;
  * Roll-Off;
  * Modulation;
  * Modulation size;
* ZC (Zadoff-Chu)
  * Root of the Zadoff-Chu sequence;
  * Length of the Zadoff-Chu sequence;
  * Rate;
* Pilots
  * Frequencies;
* DSP
  * Tone cutoff;
  * Subframes size;
  * Abort clock recovery;
  * Alice DAC rate;
  * Exclusion zone;
  * Phase filtering.
