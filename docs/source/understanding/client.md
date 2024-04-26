# Client

Bob client is represented by a python class {py:class}`qosst_bob.bob.Bob` with methods and attributes.

Usually the *public* methods represent the different steps of the protocol, that also corresponds to the different actions in the GUI. The attributes correspond to the variables that Bob needs to pass to the different methods such as the recovered symbols, Alice symbols, etc... but also the hardware.

Here we give a short introduction on how this client works and does. We also give the python instructions with a fully working example at the end.

In general, more details about the control protocol can be found on the {external+qosst-core:doc}`corresponding page in qosst-core documentation <understanding/control_protocol>`.

## Initializing the client and the hardware

The client is to be initialized like any other python instance. The required parameter is the location of the configuration path:

```{code-block} python
from qosst_bob.bob import Bob

bob = Bob("config.toml")
```

This call does almost nothing, apart from reading the configuration and setting some attributes to their default values. Once this is done, the hardware can be initialized with the following command

```{code-block} python

bob.open_hardware()
```

## Load the calibration data

We now need to load the calibration data, in particular the electronic noise data, this can be done with the following command:

```{code-block} 
bob.load_electronic_noise_data()
```

The location that will be used to load the electronic shot is the one set in the configuration path in `bob.electronic_noise.path`.

````{note}
It is possible to acquire the electronic noise data with Bob client using the following code

```{code-block} python
bob.get_electronic_noise_data()
```

However this code will only make the acquisition so you need to check that the local oscillator is off, and that the input signal is switched off. Also it is possible to make the acquisition from the GUI.

It is also possible to save the electronic noise data with 

```{code-block} python
bob.save_electronic_noise_data()
```

The location will be the one set in the configuration file in `bob.electronic_noise.path`.
````

You might also want, if you don't use one of the automatic calibration (see {doc}`here <./calibration>`) to load the electronic and shot noise data

```{code-block} python
bob.load_electronic_shot_noise_data()
```

## Connect, identify and initialize the frame

Once the hardware is set up, and the calibration is loaded, it's time to connect to Alice. Make sure Alice server has started and execute the following code

```{code-block}
bob.connect()
bob.identification()
bob.initialization()
```

The first command will establish the connection with Alice.

In the second command, Bob will send its serial number and the version of QOSST, so that Alice can verify that everything is in order. It's also during this step that the authentication is initialized and that Alice and Bob agree on a configuration.

In the last command, Bob generates a UUID and send it to Alice. This starts a new CV-QKD frame.

## Quantum Information Exchange

The next step is to proceed to the Quantum Information Exchange. This step is triggered with the following code

```{code-block} python
bob.quantum_information_exchange()
```

Here we describe the steps that are involved when this method is called:

1. Check if slow automatic shot noise calibration has to be performed. If yes, perform the calibration by switching off the signal input, making an acquisition, and switching on the signal input. If no, do nothing;
2. Send the `QIE_REQUEST` message to Alice.;
3. Check if fast automatic shot noise calibration has to be performed. If yes, switch off the signal input. If no, do nothing.
4. When Alice answers with `QIE_READY`, start the acquisition;
5. Check if fast automatic shot noise calibration has to be performed. If yes, wait for the amount of time specified and switch back on the signal input. If no, don't wait (and don't switch bask as the switch off never happened in the first place).
6. Send the message `QIE_TRIGGER` to Alice;
7. Upon reception of the message `QIE_EMISSION_STARTED` from Alice, start a timer;
8. When the timer has ended, send the message `QIE_ACQUISITION_ENDED` to Alice;
9. Upon reception of the message `QIE_ENDED` from Alice, finish this action.

## Digital Signal Processing

The digital signal processing step can be started with the code

```{code-block} python
bob.dsp()
```

This will actually do a few things. First it will do the actual DSP, as explained {doc}`here <./dsp>` and then, it will also ask the symbols to Alice, to be able to correct the global angle (and then later to estimate the parameters), and it will also apply the DSP on the electronic noise data and electronic and shot noise data.

## Parameters estimation

Performing the parameters estimation can be done with the following code

```{code-block} python
bob.parameters_estimation()
```

that will execute a code as described {doc}`here <./parameters_estimation>`. In particular, after this step, the {py:attr}`~qosst_bob.bob.Bob.skr` attribute is updated and can be displayed


```{code-block} python
print(bob.skr)
```

During this step, the average number of photons per symbol {math}`\langle n \rangle` is requested to Alice . The results of parameters estimation are also sent to Alice.

## Error correction and privacy amplification

Those steps are not implemented yet and calling one of {py:meth}`~qosst_bob.bob.Bob.error_correction` or {py:meth}`~qosst_bob.bob.Bob.privacy_amplification` will result in raising the `NotImplemented` exception.

## Closing Bob

Whenever it is possible, it is better to properly close Bob with the following code

```{code-block} python

bob.close()
```

This will properly close the socket and hardware connections.


## Full example

```{code}
from qosst_bob.bob import Bob

bob = Bob("config.toml")

# Initialize the hardware
bob.open_hardware()

# Load the electronic noise data
bob.load_electronic_noise_data()

# Connect to Alice
bob.connect()

# Identification
bob.identification()

# Initialization
bob.initialization()

# QIE
bob.quantum_information_exchange()

# DSP
bob.dsp()

# Parameters estimation
bob.parameters_estimation()
print(bob.skr)

# Close Bob
bob.close()
```