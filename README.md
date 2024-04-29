# qosst-bob

<center>

![QOSST Logo](qosst_logo_full.png)

<a href='https://qosst-bob.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/qosst-bob/badge/?version=latest' alt='Documentation Status' />
</a>
<a href="https://github.com/qosst/qosst-bob/blob/main/LICENSE"><img alt="Github - License" src="https://img.shields.io/github/license/qosst/qosst-bob"/></a>
<a href="https://github.com/qosst/qosst-bob/releases/latest"><img alt="Github - Release" src="https://img.shields.io/github/v/release/qosst/qosst-bob"/></a>
<a href="https://pypi.org/project/qosst-bob/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/qosst-bob"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/pylint-dev/pylint"><img alt="Linting with pylint" src="https://img.shields.io/badge/linting-pylint-yellowgreen"/></a>
<a href="https://mypy-lang.org/"><img alt="Checked with mypy" src="https://www.mypy-lang.org/static/mypy_badge.svg"></a>
<a href="https://img.shields.io/pypi/pyversions/qosst-bob">
    <img alt="Python Version" src="https://img.shields.io/pypi/pyversions/qosst-bob">
</a>
<img alt="Docstr coverage" src=".docs_badge.svg" />
</center>
<hr/>

This project is part of [QOSST](https://github.com/qosst/qosst).

## Features

`qosst-bob` is the module of QOSST in charge of the functionalities of Bob for CV-QKD. In particular it includes:

* Acquisition of the signal detected by Bob's detector;
* Digital Signal Processing of the acquired signal to recover the symbols sent by Alice;
* Correlations analysis to estimate the parameters of the channel;
* Bob's client to interact with Alice's server;
* Interfaces to this client (scripts and graphical interface).

## Installation

The module can be installed with the following command:

```console
pip install qosst-bob
```

It is also possible to install it directly from the github repository:

```console
pip install git+https://github.com/qosst/qosst-bob
```

It also possible to clone the repository before and install it with pip or poetry

```console
git clone https://github.com/qosst/qosst-bob
cd qosst-alice
poetry install
pip install .
```

## Documentation

The whole documentation can be found at https://qosst-bob.readthedocs.io/en/latest/

## Usage of the client

The client can either be used with the graphical interface, with one of the provided script (optimize, excess noise or transmittance) or in a home-made script.

In any case the first step is to create a configuration file. This can be done with a command line tool shipped with the `qosst-core` package (which is a dependency of `qosst-alice`):

```console
qosst configuration create
```

This will create the default configuration file at the location `config.toml` (you change the location with the `-f` or `--file` option). For more information on the meaning of each parameter in the configuration and how to change them, check the [qosst tutorial](https://qosst.readthedocs.io/en/latest/tutorial.html).

### GUI usage

The GUI can be launched with the following command:

```console
qosst-bob-gui
```

A tutorial on the GUI can be found in the documentation.

### Scripts usage

The command line of the three scripts are given below:

```console
qosst-bob-excess-noise
qosst-bob-optimize
qosst-bob-transmittance
```

A good start is to add the `-h` flag to get information on the command line options. An extensive documentation is also written in the CLI section of the documentation.

### Usage in home-made scripts

It's possible to import Bob's client:

```python
from qosst_bob.bob import Bob
```

and use it in home-made script. Please refer to the documentation for more details.

## License

As for all submodules of QOSST, `qosst-bob` is shipped under the [Gnu General Public License v3](https://www.gnu.org/licenses/gpl-3.0.html).

## Contributing

Contribution are more than welcomed, either by reporting issues or proposing merge requests. Please check the contributing section of the [QOSST](https://github.com/qosst/qosst) project fore more information.
