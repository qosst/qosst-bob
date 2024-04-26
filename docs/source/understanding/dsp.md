# Digital Signal Processing

The goal of this section is to give a general understanding of the Digital Signal Processing algorithm, while also giving links to where to look to for more information.

## Structure of the DSP

The python structure is the following: the main code of the DSP is in the {py:mod}`qosst_bob.dsp.dsp` module but also call functions from modules including {py:mod}`qosst_bob.dsp.pilots`, {py:mod}`qosst_bob.dsp.resample` and {py:mod}`qosst_bob.dsp.zc`.

```{note}
The code contained in the {py:mod}`qosst_bob.dsp.equalizers` is not currently in use, but the code for an equalizer based on the Constant Modulus Algorithm (CMA) is available there.
```

The entrypoint for python is the {py:func}`~qosst_bob.dsp.dsp.dsp_bob` function that takes as input the data and the configuration object and returns the recovered symbols, the parameters for the special DSP and some debug information for the DSP.

This function actually calls the {py:func}`~qosst_bob.dsp.dsp.dsp_bob_params` which takes as parameters the data and all the DSP parameters (20+ parameters).

This last function actually calls one of four functions, depending on actual setup (clock reference shared or not, local oscillator transmitter or not).

If the clock reference is not shared, and the local oscillator is not transmitted, the {py:func}`~qosst_bob.dsp.dsp._dsp_bob_general` function is called.

```{warning}
Here, we only talk about the general DSP, that should be used whenever possible. While some simplifications are possible in special cases, the other DSP algorithms should be considered unsafe.
```

The internal steps of the {py:func}`~qosst_bob.dsp.dsp._dsp_bob_general` function are listed below:

* Find an approximative start of the Zadoff-Chu sequence
* Find the pilots
* Correct clock difference
* Find the pilots again with the good clock
* Estimate the beat frequency
* Find the Zadoff-Chu sequence
* Estimate the beat frequency (per subframe)
* Find one pilot (per subframe)
* Unshift the quantum signal (per subframe)
* Apply matched RRC filter (per subframe)
* Downsample (per subframe)
* Correct relative phase noise (per subframe)

We give a more involved description of the steps below.


## Clock recovery

The first 3 steps above are actually done for one thing: correct the clock mismatch between Alice and Bob.

As Alice and Bob are not actively sharing a clock, their "definition of a Hertz" might vary. Even a slight variation will have big impacts at the considered rates. This is the reason why two pilots are needed in the general case: the difference of them gives us a reference for the Hertz.

The correction algorithm goes like this: given that the 2 pilots are emitted with frequencies {math}`f_{pilot,1}` and {math}`f_{pilot,2}` at Alice side, and found at {math}`\tilde{f}^B_{pilot,1}` and {math}`\tilde{f}^B_{pilot,2}`, then we can estimate the clock mismatch with 

```{math}
\Delta f = \frac{\tilde{f}^B_{pilot,2} - \tilde{f}^B_{pilot,1}}{f_{pilot,2} - f_{pilot,1}}
```

This gives the "deviation of Bob's Hertz compared to Alice's Hertz".

Hence, to be able to correct the clock, we need to get the frequencies of two pilot. This can be done using the Fourier Transform or the Power Spectral Density but is more efficient if done on a small chunk of data where we know we have the pilots. This is true for two reasons: the first is the fact that on the whole data, we don't have the pilots all the time and the second is that the frequency of the beat between the two laser is moving a bit, meaning that the two pilots are also moving slightly in frequency. To be able to get a clean estimation, the time scale need to be low.

This is done by finding an estimate of the Zadoff-Chu sequence (it cannot be found precisely before clock recovery and carrier frequency estimation) with a uniform 1D filter. Once the beginning of the sequence, a small chunk of data is taken from what we know has the pilots.

## Carrier frequency estimation

Once the clock is recovered, we want to estimate the beat frequency between the two lasers. Indeed, the way balanced detection works is that if the signal and the local oscillator have a frequency difference (a wavelength difference), then the acquired signal will be displaced in frequency by an amount {math}`f_{beat}` corresponding to the frequency difference between the two lasers.

Even a wavelength difference of tens of pico-meters will induce shift in the order of tens of MHz, at telecom wavelength.

Once the clock has been recovered, we can find again the frequency of one of the tones, for instance the first one, {math}`f^B_{pilot,1}` and compare it to the emitted tone to get the beat frequency

```{math}
f_{beat} = f^B_{pilot,1} - f_{pilot,1}
```

This beat frequency is needed to find the Zadoff-Chu sequence.

## Frame synchronisation

To perform frame synchronisation, the whole data is first unshifted by the amount {math}`f_{beat}`, *i.e* multiplied by a complex exponential. This has the effect of bringing the Zadoff-Chu sequence in baseband, the same was emitted.

The beginning of the Zadoff-Chu sequence, and hence the beginning of the quantum data, is found by cross-correlating the data with a locally generated Zadoff-Chu sequence, with adjusted rate.

## Subframe processing

The rest of the DSP is done as subframes. This means that will recover the symbols in a small chunk of data and repeat the analysis. This is done for an already exposed reason: the beat frequency is changing overtime, and not taking this change into account gives bad result.

The size of the frame can be configured in the DSP and is given as the number of symbols that should be recovered in each frame.

### Carrier frequency estimation

The first step is to get a proper estimation of {math}`f_{beat}` from the frame. This is done the same way as before: the pilot frequency {math}`f^B_{pilot,1}` is estimated in the frame and the beat frequency is obtained as

```{math}
f_{beat} = f^B_{pilot,1} - f_{pilot,1}
```

### Pilot recovery

Then we filter the pilot with a FIR window. This pilot data will will be used later for the relative phase recovery. The cutoff frequency of the filter can be configured in the configuration.

### Unshift signal

The whole signal is then unshifted by the amount {math}`f_{beat}+f_{shift}` by a multiplication by a complex exponential of the form

```{math}
\exp(-2\pi j(f_{beat}+f_{shift}) t)
```

After this operation the center frequency of the quantum data should be zero.

### Matched filter

We then proceed to apply a matched RRC filter on the data. The same parameters are used, in particular the roll-off factor and the symbol rate.

More information on the RRC filter can be found on the {external+qosst-alice:doc}`DSP page of qosst-alice <understanding/dsp>`.

The resulting data after this operation would correspond to the output of a raised cosine filter, which is known to minimise inter symbols interference.

### Optimal downsampling

Now we have a mixture of all symbols. The way to get the good symbols, we need to find the good sampling point. Again you can refer to this {external+qosst-alice:doc}`this page of the qosst-alice documentation <understanding/dsp>` to understand why.

If we sample at the good point, the variance of the symbols will be maximum. This is because, if we don't the best sampling point, all the symbols will be sampled with a lower amplitude.

Also the number of possible sampling point is finite because it has to be lower than the symbol period. For instance if the symbol rate is 100 MBaud, and the ADC rate is 2.5 GSamples/s, then the number of samples per symbol (SPS) at Bob side is 25, meaning that we have 25 possible starting point. The general idea is to compute

```{code-block} python
sps = adc_rate/symbol_rate
np.argmax([np.var(symbols[i::sps]) for i range(sps)])
```

In fact this is not that simple since, after clock correction the SPS is usually not an integer. Therefore, we have a function to downsample with a float SPS: {py:func}`qosst_bob.dsp.resample._downsample_float`.

Once the best downsampling point is found, we downsample and we get the symbols with phase error.

### Correct relative phase

The correction of the relative phase is done in the following way. The phase error is computed as the phase difference between the filtered pilot and a perfect complex exponential at frequency {math}`f^B_{pilot,1}`.

This phase error is then used to cancel the error on the symbols.

The actual DSP function stops here, and return a list of arrays of symbols (one array for each subframe).

```{warning}

As the global phase for each subframe is not the same, it is not yet possible to combine the different arrays.
```

### Correct global phase

The correction of the global phase requires Alice symbols. Hence, one the main DSP function is done, Bob client will go again through the subframes and will request symbols from Alice. The ratio of symbols that will be used for global phase recovery and parameters estimation vs the symbols that will be used for key generation can be set in the configuration.

Once Bob gets the symbol, the global phase correction is found by computing the covariance between the rotated version of Bob symbols (by a global phase) and Alice symbols. The maximal covariance corresponds to the best angle for global phase correction.

After this Bob client will merge all the subframes data in one array.

## Special DSP

We quickly discuss here the special DSP. The DSP transformation has to be applied on the electronic noise and the electronic and shot noise, in order to have a correct normalisation.

The special DSP only applies part of the DSP (phase is not relevant, and not need to find again the pilots and the Zadoff-Chu sequence). The operations are

* Unshift;
* RRC filter;
* Downsample.

The parameters for this DSP are outputted by the general DSP in a {py:class}`qosst_bob.dsp.dsp.DSPDebug` and can directly be given to the {py:func}`qosst_bob.dsp.dsp.special_dsp` function.