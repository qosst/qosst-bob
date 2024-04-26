# Using qosst-bob

This section will explain how to start using `qosst-bob`. This procedure is also explained in the {external+qosst:doc}`QOSST tutorial <technical/tutorial>`

## Creating the configuration file and filling it

`qosst-bob` is not shipped with a default configuration file but the default configuration file of QOSST can be generated using the following command

```{prompt} bash

qosst configuration create
```

This will create the configuration file at the `config.toml` default location. The documentation of this command can be found at the page {external+qosst-core:doc}`cli/index` of the `qosst-core` documentation.

Once the default configuration file is created, the whole `[alice]` section can be removed. The `[bob]` and `[frame]` sections must then be completed to reach the expected behaviour, and to connect to the good hardware. Here are some link that can be useful for filling these sections:

* {external+qosst:doc}`QOSST tutorial <technical/tutorial>`;
* {external+qosst-core:doc}`configuration explanation in qosst-core documentation <understanding/configuration>`;
* {external+qosst-core:doc}`filters and Zadoff-Chu explanation in qosst-core documentation <understanding/comm>`;
* {doc}`explanation on Bob's DSP <../understanding/dsp>`


## Using the GUI

The GUI can then be launched with the `qosst-bob-gui` command. The good configuration file can then be loaded using the top left panel. After clicking on the "load" configuration button, the configuration parameters should now be displayed on the bottom right tabs. The hardware can be loaded by clicking on the button "Init hardware" and the communication with Alice can start, if Alice is running, by clicking on "Connect". Then clicking on "Identification" and "Initialization" will perform the initialization steps and the actual frame exchange can be done by clicking on "QIE".

To run the DSP and the parameters estimation step however, it is required to have the electronic noise and electronic and shot noise. The electronic noise must be acquired before, and the electronic and shot can be either be acquired before or have an automatic calibration. This is explained in more details {doc}`here <../understanding/calibration>`.

## Using a script

We take here the example of the `qosst-bob-excess-noise` but the procedure is similar with the other.

The script can then be started using the simple command

```{prompt} bash
qosst-bob-excess-noise -f config.toml 200
```

200 means that the script will sequentially perform 200 exchanges of frame.

It can also be useful to get the more logs by adding one or several `-v`:

```{prompt} bash
qosst-bob-excess-noise -f config.toml -vv 200
```

with the following relation:

* No `-v`: Only print errors and critical errors;
* `-v`: Same as above with warnings added;
* `-vv`: Same as above with info added;
* `-vvv`: all logs.

More information on the command line can be found {doc}`here <../cli/documentation>`.

```{note}
Alice server must be running before starting the scripts. It is also necessary to perform the calibration steps as explained {doc}`here <../understanding/calibration>`.
```