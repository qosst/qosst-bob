# Parameters estimation

The role of parameters estimation is to estimate the different values that will be used for the computation of the secret key rate. The parameters you need depends on the exact security proof you are using, and the idea of the software was that the output of the parameters estimation procedure could be plugged into the secret key rate calculator. However, this is not currently the case. This issue is also discussed in {external+qosst-skr:doc}`this page of the qosst-skr documentation <introduction/skr>`.

This is why in the following we restrict in the case of the Gaussian modulation where the channel can be assumed Gaussian since the optimal attack is Gaussian. In this setting we need to evaluate the effect on the two first moments of our coherent states: the transmittance {math}`T` and the excess noise {math}`\xi`.

Taking a simple model where the symbols of Alice are represented by {math}`X` and the symbols of Bob by {math}`Y`, then we have the relation

```{math}
Y = \sqrt{\eta T} X + n
```

where {math}`n` is some white gaussian noise {math}`n \sim \mathcal{N}(0, 1+V_{el}+\eta T \xi)`. Then it's possible to show that

```{math}
\langle X^2 \rangle = V_A
```

```{math}
\langle XY \rangle = \sqrt{\eta T}V_A
```

```{math}
\langle Y^2 \rangle = 1 + V_{el} + \eta T V_A + \eta T \xi
```

so that 

```{math}
T = \frac{1}{\eta} \left(\frac{\langle XY \rangle}{V_A}\right)^2
```

and 

```{math}
\xi = \frac{\langle Y^2 \rangle - 1 - V_{el} - \eta T V_A}{\eta T}
```

The actual exact formula depends on the exact schema of detection and the representation of {math}`X` and {math}`Y`. Also, one does not usually have direct access to {math}`X`, but rather {math}`X'` such that {math}`X = \alpha X'` with {math}`\alpha` a real number, and this {math}`\alpha` might not be known. This coefficient corresponds to the transformation from the arrays on the computer to the actual string of symbols (DAC, modulator, attenuation). This coefficient could be characterized, but this is not mandatory. Instead one can deduce {math}`\alpha` by computing {math}`\langle X'^2 \rangle` and comparing it to the average number of photons per symbol {math}`\langle n \rangle`.

The parameters estimation method will also output the normalized value of {math}`V_{el}`. The action of the parameters estimation method can summarized as:

1. Get the shot noise value by subtracting the electronic noise value to the electronic and shot noise value;
2. Compute the normalized value of the electronic noise {math}`V_{el}`;
3. Measure the coefficient {math}`\alpha` by comparing the variance of Alice symbols to the photon number;
4. Measure the covariance between the symbols of Alice and of Bob, get {math}`T`;
5. Measure the variance of the symbols of Bob, get {math}`\xi`.

Here is an example of use for the parameters estimation:

```{code-block} python

from qosst_bob.parameters_estimation.base import DefaultEstimator

symbols_alice = ...
symbols_bob = ...
photon_number = ...
electronic_symbols = ...
electronic_shot_symbols = ...

t, xi, vel = DefaultEstimator.estimate(symbols_alice, symbols_bob, photon_number, electronic_symbols, electronic_shot_symbols)
```