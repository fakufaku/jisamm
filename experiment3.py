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


def exp3_gen_args(config):

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


