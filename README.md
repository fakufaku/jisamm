OverIVA Companion Code
======================

Abstract
--------

Hoge

Authors
-------

* Robin Scheibler
* Nobutaka Ono

Algorithms
----------

### Single source

* OverIVA, IP + parametrization
* OverIVA, IP, no parametrization
* FIVE
* IVE [Koldovsky et al.]
* QuickIVE ?

### Multi-source

* OverIVA, IP + parametrization
* OverIVA, IP, no parametrization
* OverIVA, IP2 + parametrization
* OverIVA, Iterative FIVE
* AuxIVA [Ono]
* QuickIVA ? [Koldovsky et al.]

### Source Model

Laplace only.

Summary of Experiments
----------------------

### Experiment 1: Speed contest

Plot runtime of algorithm vs SDR (or SIR).

### Experiment 2: Separation performance

Separation performance for different numbers of sources and microphones.
Similar to experiment in overiva paper, but with more algorithms and SINR points.
This time, we can plot histograms (instead of boxplots).

### Experiment 3: Effect of background and reverberation

We vary two parameters

* The number of interferers (background Gaussianity)
* The distance from microphones to sources (reverberation time)

