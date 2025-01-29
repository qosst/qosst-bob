# Using the Command Line Interface (CLI)

Here we describe the general idea of the command line interfaces that are shipped with `qosst-bob`. The full documentation of the CLI is available {doc}`here <./documentation>`.

The package is shipped with 5 command line interfaces:

* `qosst-bob-gui`
* `qosst-bob-excess-noise`
* `qosst-bob-transmittance`
* `qosst-bob-optimize`
* `qosst-bob-tools`

We give a brief description and usage example in the following.

## Level of verbosity

Usually, the scripts are not logging a lot in the console: they are only logging errors and critical errors. To get more logs, it is possible to pass `-v` to the command line with the following relation:

* No `-v`: Only print errors and critical errors;
* `-v`: Same as above with warnings added;
* `-vv`: Same as above with info added;
* `-vvv`: all logs.

```{note}
The verbosity of logs saved in a file are not handled in the command line but in the configuration file, in the `logs` section.
```

## qosst-bob-gui


This command launches the GUI, and don't take parameters apart from the verbosity level discussed above.

```{prompt} bash
qosst-bob-gui -vv
```

## qosst-bob-excess-noise

This script will initialize Bob client, and proceed to a number of frame exchanges that is passed as a parameter. Other important parameters is the verbosity level discussed above and the location of the config file.

```{prompt} bash
qosst-bob-excess-noise -f config.toml -vv 200
```

will do 200 frames exchange. For each frame the different parameters are saved and at the end of the script the results of the experiment are saved in a {py:class}`~qosst_bob.data.ExcessNoiseResults` container.

## qosst-bob-transmittance

This script will initialize Bob client, and will apply an attenuation to a VOA to emulate a channel. For each value of attenuation, the script will exchange a number of fames that is given in parameter. Other important parameters include the verbosity and the location of the configuration file.

```{prompt} bash
qosst-bob-transmittance -f config.toml -vv -n 5 0 5 0.01
```

mean that the value to apply to the VOA ranges from 0 to 5 with a step of 0.01 and for each value of attenuation, 5 frames are exchanged.

For each frame, the different parameters that were estimated are saved, and at the end of the script, the data is saved in a {py:class}`~qosst_bob.data.TransmittanceResults` container.

## qosst-bob-optimize

This script will initialize Bob client and an updater. The updater will automatically change one configuration parameter on Bob side and on Alice side (or only one of the two when the other change is not necessary). For each value of the parameter, the script will exchange a number of frames passed as a parameter.

There are 10 updaters available and the list can be found below:

```{program-output} python3 -c "from qosst_bob.optimization import updaters; print('\n'.join(['* ' + str(mod).split('.')[0].replace('_', ' ').capitalize() for mod in updaters.__loader__.get_resource_reader().contents() if str(mod)[:2] != '__']))"

```

and more information {doc}`here <../api/optimization>`. The exact parameters will differ from one updater to the other, so it's important to either look at the {doc}`cli documentation <./documentation>` or to use the `-h` to get help directly from the command line. But some parameters are always there, such as the verbosity, the location of the configuration file and the number of repetitions per parameter value.

For instance, this is an example to optimize the value of the roll-off parameter on a range from 0 to 1 with step of 0.01 and 5 repetitions per roll-off value:

```{prompt} bash
qosst-bob-optimize -f config.toml -vv -n 5 roll-off 0 1 0.01
```

For each frame, the values of the estimated parameters are saved and at the end of the script saved in a {py:class}`~qosst_bob.data.OptimizationResults` container.

## qosst-bob-offline-dsp

```{warning}
Offline DSP is an experimental feature.
```

This script allows to run the DSP of Bob without having to run Alice and Bob. The script requires, at least, the data of the acquisition, the electronic noise data, the electronic and shot noise data, the symbols of Alice and the average number of photons per symbol at Alice's side. These are the position arguments.

Optional arguments include the configuration file (for Bob), the verbosity and if the data should be saved.

An example of usage is

```{prompt} bash
python3 offline_dsp.py -f config.toml -vv signal.npy elec.npy elec_shot.npy symbols.npy 5
```

QOSST is made such that the exported data from the GUI or Bob's client can be fed directly to this script.

## qosst-bob-tools

The last command line interface is for tools for Bob. Currently two tools are available: `eta-voltage` and `eta_current` that can be used for the calibration of eta (see {doc}`here <../understanding/calibration>` for more information).

### eta voltage

The script can be started with the following command:

```{prompt} bash
qosst-bob-tools eta-voltage 10
```

The important parameter is the gain (in the above example, it's 10) which is the gain between the monitoring output in V and the photocurrent in A (the gain is in V/A). Other parameters include the verbosity level and the `--no-save` options. All the other configuration will be done interactively in the script.

More information can be found {doc}`here <../understanding/calibration>`.

### eta current

The script can be started with the following command:

```{prompt} bash
qosst-bob-tools eta-current
```

The only parameters are the verbosity level and the `--no-save` option. All the other configuration will be done interactively in the script.

More information can be found {doc}`here <../understanding/calibration>`.
