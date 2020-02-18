import time
import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.bss.common import projection_back

from room_builder import (
    callback_noise_mixer,
    choose_target_locations,
    random_locations,
    convergence_callback,
)
from samples import sampling


config = {
    "seed": 12345,
    "repetitions": 10,
    "sinr": [-5, 0, 5, 10],
    "targets": [1, 2, 4],
    "interferers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "dist_crit_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "mix_params": {"diffuse_ratio": 0.99, "ref_mic": 0},
    "room": {
        "rt60_s": 0.415,
        "critical_distance_m": 1.95,
        "room_kwargs": {
            "dim": [9, 12, 4.5],
            "fs": 16000,
            "absorption": 0.331,
            "max_order": 34,
        },
        "mic_array_geometry_m": [
            [0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.01, 0.017, 0.0],
            [-0.01, 0.017, 0.0],
            [-0.02, 0.0, 0.0],
            [-0.01, -0.017, 0.0],
            [0.01, -0.017, 0.0],
        ],
        "mic_array_location_m": [4.496, 5.889, 2.327],
    },
    "algorithms": {"overiva": {"algo": "overiva", "kwargs": {}}},
}


def exp3_pre_config(config):

    # infer a few arguments
    room_dim = config["room"]["room_kwargs"]["room_dim"]
    mic_array_center = np.array(config["room"]["mic_array_location_m"])
    mic_array = mic_array_center[None, :] + np.array(
        config["room"]["mic_array_geometry_m"]
    )
    critical_distance = config["room"]["critical_distance_m"]

    # master seed
    np.random.seed(config["seed"])

    # choose all the files in advance
    gen_files_seed = int(np.random.randint(2 ** 32, dtype=np.uint32))
    all_wav_files = sampling(
        config["repeat"],
        np.max(config["targets"]) + np.max(config["interferers"]),
        config["samples_list"],
        gender_balanced=True,
        seed=gen_files_seed,
    )

    # sub-seeds
    sub_seeds = []
    for r in range(config["repeat"]):
        sub_seeds.append(int(np.random.randint(2 ** 32)))

    # create all distinct interferers locations
    interferers_locs = []
    for n in range(config["repeat"]):
        interferers_locs.append(
            random_locations(
                np.max(config["n_interferers"]),
                room_dim,
                mic_array_center,
                min_dist=critical_distance,
            ).tolist()
        )

    # create all distinct target locations
    target_locs = {}
    for n in config["n_targets"]:
        target_locs[n] = {}
        for dist_ratio in config["dist_crit_ratio"]:
            dist_mic_target = dist_ratio * critical_distance
            target_locs[n][dist_ratio] = choose_target_locations(
                n, mic_array_center, dist_mic_target
            ).tolist()

    args = []
    for sinr in config["sinr"]:
        for n_targets in config["targets"]:
            for n_interf in config["interferers"]:
                for dist_ratio in config["dist_crit_ratio"]:
                    for r in range(config["repeat"]):

                        # bundle all the room parameters for the simulation
                        room_params = {
                            "room_kwargs": config["room"]["room_kwargs"],
                            "mic_array": mic_array.tolist(),
                            "sources": np.concatenate(
                                (
                                    target_locs[n_targets][dist_ratio],
                                    interferers_locs[r],
                                ),
                                axis=1,
                            ).tolist(),
                            "wav": all_wav_files[r][: n_targets + n_interf],
                        }

                        args.append(
                            (
                                sinr,
                                n_targets,
                                n_interf,
                                dist_ratio,
                                room_params,
                                sub_seeds[r],
                            )
                        )

    return args


def run(args):

    global parameters

    # external library imports
    import numpy
    import pyroomacoustics

    # local imports
    from metrics import si_bss_eval
    from samples import wav_read_center

    # expand arguments
    sinr, n_targets, n_interf, dist_ratio, room_params, seed = args

    n_mics = len(room_params["mic_array"])
    n_sources = n_targets + n_interf

    # this is the underdetermined case. We don't do that.
    if n_mics < n_targets:
        return []

    # set MKL to only use one thread if present
    try:
        import mkl

        mkl.set_num_threads(1)
    except ImportError:
        pass

    # set the RNG seed
    np.random.seed(seed)

    # get all the signals
    source_signals = wav_read_center(room_params["wav"][:n_sources], seed=123)

    # create the room
    room = pra.ShoeBox(**parameters["room"]["room_kwargs"])
    room.add_microphone_array(
        pra.MicrophoneArray(np.array(room_params["mic_array"]).T, room.fs)
    )
    for n in range(n_sources):
        room.add_source(room_params["sources"][n], signal=source_signals[n, :])

    # compute RIRs and RT60
    room.compute_rir()
    rt60 = np.median(
        [
            pra.experimental.measure_rt60(room.rir[0][n], fs=room.fs)
            for n in range(n_targets)
        ]
    )

    # signals after propagation but before mixing
    # (n_sources, n_mics, n_samples)
    premix = room.simulate(return_premix=True)
    n_samples = premix.shape[-1]

    # create the mix (n_mics, n_samples)
    # this routine will also resize the signals in premix
    mix = callback_noise_mixer(
        premix,
        sinr=sinr,
        n_src=n_targets + n_interf,
        n_tgt=n_targets,
        **parameters["mix_params"]
    )

    # create the reference signals
    # (n_sources + 1, n_samples)
    refs = np.zeros((n_targets + 1, n_samples))
    refs[:-1, :] = premix[:n_targets, parameters["mix_params"]["ref_mic"], :]
    refs[-1, :] = np.sum(premix[n_targets:, 0, :], axis=0)

    # STFT parameters
    framesize = parameters["stft_params"]["framesize"]
    hop = parameters["stft_params"]["hop"]
    if parameters["stft_params"]["window"] == "hann":
        win_a = pra.hamming(framesize)
    else:  # default is Hann
        win_a = pra.hann(framesize)
    win_s = pra.transform.compute_synthesis_window(win_a, hop)

    # START BSS
    ###########

    # shape: (n_frames, n_freq, n_mics)
    X_all = pra.transform.analysis(mix.T, framesize, hop, win=win_a)
    X_mics = X_all[:, :, :n_mics]

    # store results in a list, one entry per algorithm
    results = []

    # compute the initial values of SDR/SIR
    init_sdr = []
    init_sir = []

    convergence_callback(
        X_mics,
        X_mics,
        n_targets,
        init_sdr,
        init_sir,
        [],
        refs,
        parameters["mix_params"]["ref_mic"],
        parameters["stft_params"],
        "init",
        False,
    )

    for full_name, params in parameters["algorithm_kwargs"].items():

        name = params["algo"]
        kwargs = params["kwargs"]

        if name == "auxiva_pca" and n_targets == 1:
            # PCA doesn't work for single source scenario
            continue
        elif name in ["ogive", "five"] and n_targets != 1:
            # OGIVE is only for single target
            continue

        results.append(
            {
                "algorithm": full_name,
                "n_targets": n_targets,
                "n_interferers": n_interf,
                "n_mics": n_mics,
                "rt60": rt60,
                "sinr": sinr,
                "seed": seed,
                "sdr": [],
                "sir": [],  # to store the result
                "runtime": np.nan,
                "eval_time": np.nan,
                "n_samples": n_samples,
            }
        )

        # this is used to keep track of time spent in the evaluation callback
        eval_time = []

        def cb(Y):
            convergence_callback(
                Y,
                X_mics,
                n_targets,
                results[-1]["sdr"],
                results[-1]["sir"],
                eval_time,
                refs,
                parameters["mix_params"]["ref_mic"],
                parameters["stft_params"],
                name,
                name not in parameters["overdet_algos"],
            )

        # avoid one computation by using the initial values of sdr/sir
        results[-1]["sdr"].append(init_sdr[0])
        results[-1]["sir"].append(init_sir[0])

        try:
            t_start = time.perf_counter()

            if name == "auxiva":
                # Run AuxIVA
                # this calls full IVA when `n_src` is not provided
                Y = overiva(X_mics, callback=cb, **kwargs)

            elif name == "auxiva_pca":
                # Run AuxIVA
                Y = auxiva_pca(
                    X_mics, n_src=n_targets, callback=cb, proj_back=False, **kwargs
                )

            elif name == "overiva":
                # Run BlinkIVA
                Y = overiva(
                    X_mics, n_src=n_targets, callback=cb, proj_back=False, **kwargs
                )

            elif name == "overiva2":
                # Run BlinkIVA
                Y = overiva(
                    X_mics, n_src=n_targets, callback=cb, proj_back=False, **kwargs
                )

            elif name == "five":
                # Run AuxIVE
                Y = five(X_mics, callback=cb, proj_back=False, **kwargs)

            elif name == "ilrma":
                # Run AuxIVA
                Y = pra.bss.ilrma(X_mics, callback=cb, proj_back=False, **kwargs)

            elif name == "ogive":
                # Run OGIVE
                Y = ogive(X_mics, callback=cb, proj_back=False, **kwargs)

            elif name == "pca":
                # Run PCA
                Y = pca_separation(X_mics, n_src=n_targets)

            else:
                continue

            t_finish = time.perf_counter()

            # The last evaluation
            convergence_callback(
                Y,
                X_mics,
                n_targets,
                results[-1]["sdr"],
                results[-1]["sir"],
                [],
                ref,
                framesize,
                win_s,
                name,
            )

            results[-1]["eval_time"] = np.sum(eval_time)
            results[-1]["runtime"] = t_finish - t_start - results[-1]["eval_time"]

        except:
            import os, json

            pid = os.getpid()
            # report last sdr/sir as np.nan
            results[-1]["sdr"].append(np.nan)
            results[-1]["sir"].append(np.nan)
            # now write the problem to file
            fn_err = os.path.join(
                parameters["_results_dir"], "error_{}.json".format(pid)
            )
            with open(fn_err, "a") as f:
                f.write(json.dumps(results[-1], indent=4))
            # skip to next iteration
            continue

    # restore RNG former state
    np.random.set_state(rng_state)

    return results
