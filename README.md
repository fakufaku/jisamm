OverIVA Companion Code
======================

This repository contains the code for the paper "MM Algorithms for Joint Independent Subspace Analysis with Application to Blind Single and Multi-Source Extraction" by [Robin Scheibler](http://robinscheibler.org) and [Nobutaka Ono](http://www.comp.sd.tmu.ac.jp/onolab/index-e.html).

Abstract
--------

In this work, we propose efficient algorithms for joint independent subspace analysis (JISA), an extension of independent component analysis that deals with parallel mixtures, where not all the components are independent.
We derive an algorithmic framework for JISA based on the majorization-minimization (MM) optimization technique (JISA-MM).
We use a well-known inequality for super-Gaussian sources to derive a surrogate function of the negative log-likelihood of the observed data.
The minimization of this surrogate function leads to a variant of the hybrid exact-approximate diagonalization problem, but where multiple demixing vectors are grouped together.
In the spirit of auxiliary function based independent vector analysis (AuxIVA), we propose several updates that can be applied alternately to one, or jointly to two, groups of demixing vectors.

Recently, blind extraction of one or more sources has gained interest as a reasonable way of exploiting larger microphone arrays to achieve better separation.
In particular, several MM algorithms have been proposed for overdetermined IVA (OverIVA).
By applying JISA-MM, we are not only able to rederive these in a general manner, but also find several new algorithms.
We run extensive numerical experiments to evaluate their performance, and compare it to that of full separation with AuxIVA.
We find that algorithms using pairwise updates of two sources, or of one source and the background have the fastest convergence, and are able to separate target sources quickly and precisely from the background.
In addition, we characterize the performance of all algorithms under a large number of noise, reverberation, and background mismatch conditions.

Authors
-------

* [Robin Scheibler](http://robinscheibler.org)
* [Nobutaka Ono](http://www.comp.sd.tmu.ac.jp/onolab/index-e.html)


Summary of Experiments
----------------------

### Experiment 1: Separation performance

Separation performance for different numbers of sources and microphones.
Similar to experiment in overiva paper, but with more algorithms and SINR points.
This time, we can plot histograms (instead of boxplots).

### Experiment 2: Speed contest

Plot runtime of algorithm vs SDR (or SIR).

### Experiment 3: Effect of background and reverberation

We vary three parameters

* The signal-to-interference-and-noise ratio
* The number of interferers (background Gaussianity)
* The distance from microphones to sources (reverberation time)

Reproduce the Results
---------------------

The code can be run serially, or using multiple parallel workers via
[ipyparallel](https://ipyparallel.readthedocs.io/en/latest/).
Moreover, it is possible to only run a few loops to test whether the
code is running or not.

1. Run **test** loops **serially**

        python ./paper_simulation.py ./experiment1_config.json.json -t -s
        python ./paper_simulation.py ./experiment2_config.json.json -t -s

2. Run **test** loops in **parallel**

        # start workers in the background
        # N is the number of parallel process, often "# threads - 1"
        ipcluster start --daemonize -n N

        # run the simulation
        python ./paper_simulation.py ./experiment1_config.json -t
        python ./paper_simulation.py ./experiment2_config.json -t

        # stop the workers
        ipcluster stop

3. Run the **whole** simulation

        # start workers in the background
        # N is the number of parallel process, often "# threads - 1"
        ipcluster start --daemonize -n N

        # run the simulation
        python ./paper_simulation.py ./experiment1_config.json
        python ./paper_simulation.py ./experiment2_config.json

        # stop the workers
        ipcluster stop

The results are saved in a new folder `data/<data>-<time>_five_sim_<flag_or_hash>`
containing the following files

    parameters.json  # the list of global parameters of the simulation
    arguments.json  # the list of all combinations of arguments simulated
    data.json  # the results of the simulation

Figure 1., 2., 3., and 4. from the paper are produced then by running

    ./prepare_figures.sh

