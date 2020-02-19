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

Reproduce the Results
---------------------

The code can be run serially, or using multiple parallel workers via
[ipyparallel](https://ipyparallel.readthedocs.io/en/latest/).
Moreover, it is possible to only run a few loops to test whether the
code is running or not.

1. Run **test** loops **serially**

        python ./paper_simulation.py ./paper_sim_config.json -t -s

2. Run **test** loops in **parallel**

        # start workers in the background
        # N is the number of parallel process, often "# threads - 1"
        ipcluster start --daemonize -n N

        # run the simulation
        python ./paper_simulation.py ./paper_sim_config.json -t

        # stop the workers
        ipcluster stop

3. Run the whole simulation

        # start workers in the background
        # N is the number of parallel process, often "# threads - 1"
        ipcluster start --daemonize -n N

        # run the simulation
        python ./paper_simulation.py ./paper_sim_config.json

        # stop the workers
        ipcluster stop

The results are saved in a new folder `data/<data>-<time>_five_sim_<flag_or_hash>`
containing the following files

    parameters.json  # the list of global parameters of the simulation
    arguments.json  # the list of all combinations of arguments simulated
    data.json  # the results of the simulation

Figure 1., 2., 3., and 4. from the paper are produced then by running

    python ./paper_plot_figures.py data/<data>-<time>_five_sim_<flag_or_hash>

