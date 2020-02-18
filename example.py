# Copyright (c) 2019 Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Overdetermined Blind Source Separation offline example
======================================================

This script requires the `mir_eval` to run, and `tkinter` and `sounddevice` packages for the GUI option.
"""
import sys
import time

import matplotlib
import numpy as np

# Get the data if needed
from get_data import get_data, samples_dir
import pyroomacoustics as pra
from pyroomacoustics.bss import projection_back
from routines import PlaySoundGUI, grid_layout, random_layout, semi_circle_layout
from room_builder import callback_noise_mixer, convergence_callback
from scipy.io import wavfile

import ive

from get_data import samples_dir
from samples.generate_samples import sampling, wav_read_center

# Once we are sure the data is there, import some methods
# to select and read samples
sys.path.append(samples_dir)


# We concatenate a few samples to make them long enough
if __name__ == "__main__":

    algo_choices = list(ive.algos.keys())
    model_choices = ["laplace", "gauss"]
    init_choices = ["eye", "eig"]

    import argparse

    parser = argparse.ArgumentParser(
        description="Demonstration of blind source extraction using FIVE."
    )
    parser.add_argument(
        "--no_cb", action="store_true", help="Removes callback function"
    )
    parser.add_argument("-b", "--block", type=int, default=2048, help="STFT block size")
    parser.add_argument(
        "-a",
        "--algo",
        type=str,
        default=algo_choices[0],
        choices=algo_choices,
        help="Chooses BSS method to run",
    )
    parser.add_argument(
        "-d",
        "--dist",
        type=str,
        default=model_choices[0],
        choices=model_choices,
        help="IVA model distribution",
    )
    parser.add_argument(
        "-i",
        "--init",
        type=str,
        default=init_choices[0],
        choices=init_choices,
        help="Initialization, eye: identity, eig: principal eigenvectors",
    )
    parser.add_argument("-m", "--mics", type=int, default=5, help="Number of mics")
    parser.add_argument("-s", "--srcs", type=int, default=1, help="Number of sources")
    parser.add_argument(
        "-z", "--interf", type=int, default=10, help="Number of interferers"
    )
    parser.add_argument(
        "--sinr", type=float, default=5, help="Signal-to-interference-and-noise ratio"
    )
    parser.add_argument(
        "-n", "--n_iter", type=int, default=11, help="Number of iterations"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Creates a small GUI for easy playback of the sound samples",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Saves the output of the separation to wav files",
    )
    args = parser.parse_args()

    if args.gui:
        print("setting tkagg backend")
        # avoids a bug with tkinter and matplotlib
        import matplotlib

        matplotlib.use("TkAgg")

    import pyroomacoustics as pra

    # Simulation parameters
    fs = 16000
    absorption, max_order = 0.35, 17  # RT60 == 0.3
    # absorption, max_order = 0.45, 12  # RT60 == 0.2
    n_sources = args.srcs + args.interf
    n_mics = args.mics
    n_sources_target = args.srcs  # the single source case

    # Force an even number of iterations
    if args.n_iter % 2 == 1:
        args.n_iter += 1

    if ive.is_single_source[args.algo]:
        print("IVE only works with a single source. Using only one source.")
        n_sources_target = 1

    # fix the randomness for repeatability
    np.random.seed(30)

    # set the source powers, the first one is half
    source_std = np.ones(n_sources_target)

    SINR = args.sinr  # signal-to-interference-and-noise ratio
    SINR_diffuse_ratio = 0.9999  # ratio of uncorrelated to diffuse noise
    ref_mic = 0  # the reference microphone for SINR and projection back

    # STFT parameters
    framesize = 4096
    hop = framesize // 2
    window = "hamming"
    stft_params = {"framesize": framesize, "hop": hop, "window": window}
    if stft_params["window"] == "hann":
        win_a = pra.hamming(framesize)
    else:  # default is Hann
        win_a = pra.hann(framesize)
    win_s = pra.transform.compute_synthesis_window(win_a, hop)

    # algorithm parameters
    n_iter = args.n_iter

    # param ogive
    ogive_mu = 0.1
    ogive_iter = 4000

    # Geometry of the room and location of sources and microphones
    room_dim = np.array([10, 7.5, 3])
    mic_locs = semi_circle_layout(
        [4.1, 3.76, 1.2], np.pi, 0.20, n_mics, rot=np.pi / 2.0 * 0.99
    )

    target_locs = semi_circle_layout(
        [4.1, 3.755, 1.1], np.pi / 2, 2.0, n_sources_target, rot=0.743 * np.pi
    )
    interferer_locs = random_layout(
        [3.0, 5.5, 1.5], n_sources - n_sources_target, offset=[6.5, 1.0, 0.5], seed=1234
    )
    source_locs = np.concatenate((target_locs, interferer_locs), axis=1)

    # Prepare the signals
    wav_files = sampling(
        1, n_sources, f"{samples_dir}/metadata.json", gender_balanced=True, seed=2222
    )[0]
    signals = wav_read_center(wav_files, seed=123)

    # Create the room itself
    room = pra.ShoeBox(room_dim, fs=fs, absorption=absorption, max_order=max_order)

    # Place a source of white noise playing for 5 s
    for sig, loc in zip(signals, source_locs.T):
        room.add_source(loc, signal=sig)

    # Place the microphone array
    room.add_microphone_array(pra.MicrophoneArray(mic_locs, fs=room.fs))

    # compute RIRs
    room.compute_rir()

    # signals after propagation but before mixing
    # (n_sources, n_mics, n_samples)
    premix = room.simulate(return_premix=True)
    n_samples = premix.shape[-1]

    # create the mix (n_mics, n_samples)
    # this routine will also resize the signals in premix
    mix = callback_noise_mixer(
        premix,
        sinr=SINR,
        n_src=n_sources,
        n_tgt=n_sources_target,
        ref_mic=ref_mic,
        diffuse_ratio=SINR_diffuse_ratio,
    )

    # create the reference signals
    # (n_sources + 1, n_samples)
    refs = np.zeros((n_sources_target + 1, n_samples))
    refs[:-1, :] = premix[:n_sources_target, ref_mic, :]
    refs[-1, :] = np.sum(premix[n_sources_target:, ref_mic, :], axis=0)

    print("Simulation done.")

    # Monitor Convergence
    #####################

    SDR, SIR, eval_time = [], [], []

    def cb_local(Y):
        convergence_callback(
            Y,
            X_mics,
            n_sources_target,
            SDR,
            SIR,
            eval_time,
            refs,
            ref_mic,
            stft_params,
            args.algo,
            not ive.is_determined[args.algo],
        )

    if args.algo.startswith("ogive"):
        callback_checkpoints = list(
            range(1, ogive_iter + ogive_iter // n_iter, ogive_iter // n_iter)
        )
    else:
        if ive.is_dual_update[args.algo]:
            callback_checkpoints = list(range(2, n_iter + 1, 2))
        else:
            callback_checkpoints = list(range(1, n_iter + 1))
    if args.no_cb:
        callback_checkpoints = []

    # START BSS
    ###########

    # shape: (n_frames, n_freq, n_mics)
    X_all = pra.transform.analysis(mix.T, framesize, hop, win=win_a).astype(
        np.complex128
    )
    X_mics = X_all[:, :, :n_mics]

    tic = time.perf_counter()

    # First evaluation of SDR/SIR
    cb_local(X_mics[:, :, :1])

    # Initialization
    if args.init == "eig":
        X0 = ive.pca(X_mics)
    elif args.init == "eye":
        X0 = X_mics
    else:
        raise ValueError("Invalid initialization option")

    # Now run the algorithm
    if args.algo.startswith("ogive"):

        Y = ive.algos[args.algo](
            X0,
            n_iter=ogive_iter,
            step_size=ogive_mu,
            proj_back=False,
            model=args.dist,
            callback=cb_local,
            callback_checkpoints=callback_checkpoints,
        )

    elif ive.is_determined[args.algo] or ive.is_single_source[args.algo]:

        Y = ive.algos[args.algo](
            X0,
            n_iter=n_iter,
            proj_back=False,
            model=args.dist,
            callback=cb_local,
            callback_checkpoints=callback_checkpoints,
        )

    else:

        Y = ive.algos[args.algo](
            X0,
            n_src=n_sources_target,
            n_iter=n_iter,
            proj_back=False,
            model=args.dist,
            callback=cb_local,
            callback_checkpoints=callback_checkpoints,
        )

    # Last evaluation of SDR/SIR
    cb_local(Y)

    # projection back
    z = projection_back(Y, X_mics[:, :, 0])
    Y *= np.conj(z[None, :, :])

    toc = time.perf_counter()

    tot_eval_time = sum(eval_time)

    print("Processing time: {:8.3f} s".format(toc - tic - tot_eval_time))
    print("Evaluation time: {:8.3f} s".format(tot_eval_time))

    # Run iSTFT
    if Y.shape[2] == 1:
        y = pra.transform.synthesis(Y[:, :, 0], framesize, hop, win=win_s)[:, None]
    else:
        y = pra.transform.synthesis(Y, framesize, hop, win=win_s)
    y = y[framesize - hop :, :].astype(np.float64)

    if args.algo != "blinkiva":
        new_ord = np.argsort(np.std(y, axis=0))[::-1]
        y = y[:, new_ord]

    y_hat = y[:, :1]

    # Look at the result
    SDR = np.array(SDR)
    SIR = np.array(SIR)
    for s in range(n_sources_target):
        print(f"SDR: In: {SDR[0, s]:6.2f} dB -> Out: {SDR[-1, s]:6.2f} dB")
    for s in range(n_sources_target):
        print(f"SIR: In: {SIR[0, s]:6.2f} dB -> Out: {SIR[-1, s]:6.2f} dB")

    import matplotlib.pyplot as plt

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.specgram(mix[0], NFFT=1024, Fs=room.fs)
    plt.title("Microphone 0 input")

    plt.subplot(2, 1, 2)
    plt.specgram(y_hat[:, 0], NFFT=1024, Fs=room.fs)
    plt.title("Extracted source")

    plt.tight_layout(pad=0.5)

    plt.figure()
    for s in range(n_sources_target):
        plt.plot([0] + callback_checkpoints, SDR[:, s], label="SDR", marker="*")
        plt.plot([0] + callback_checkpoints, SIR[:, s], label="SIR", marker="o")
    plt.legend()
    plt.tight_layout(pad=0.5)

    if not args.gui:
        plt.show()
    else:
        plt.show(block=False)

    if args.save:
        wavfile.write(
            "bss_iva_mix.wav",
            room.fs,
            pra.normalize(mix[0, :], bits=16).astype(np.int16),
        )
        for i, sig in enumerate(y_hat):
            wavfile.write(
                "bss_iva_source{}.wav".format(i + 1),
                room.fs,
                pra.normalize(sig, bits=16).astype(np.int16),
            )

    if args.gui:

        from tkinter import Tk

        # Make a simple GUI to listen to the separated samples
        root = Tk()
        my_gui = PlaySoundGUI(root, room.fs, mix[0, :], y_hat.T, references=ref[:1, :])
        root.mainloop()
