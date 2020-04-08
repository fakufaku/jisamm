# Copyright (c) 2020 Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
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
Experiment to determine the relation between distance to
microphone and reverberation time.

We observe that the reverberation time increases up to the
critical distance and then stays pretty much flat.
"""
import time
import pyroomacoustics as pra
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from room_builder import inv_sabine


if __name__ == "__main__":

    np.random.seed(2)

    # parameters
    n_sources = 1000
    room_dim = np.array([9, 12, 4.5])
    rt60 = 0.35  # [s]

    # create the room
    reflection, max_order = inv_sabine(rt60, room_dim, pra.constants.get("c"))
    max_order = int(1.2 * max_order)
    print("reflection coefficient", reflection, "max order", max_order)
    room = pra.ShoeBox(
        room_dim, fs=16000, absorption=1 - reflection, max_order=max_order
    )

    # draw a lot of sources inside
    source_locs = np.random.rand(3, n_sources) * room_dim[:, None]
    for n in range(n_sources):
        room.add_source(source_locs[:, n])

    # add the microphone somewhat at the center but with a little noise
    mic_loc = room_dim / 2.0 + np.random.randn(3) * 0.1
    room.add_microphone_array(pra.MicrophoneArray(mic_loc[:, None], room.fs))
    print("mic_loc", mic_loc)

    # simulate
    t1 = time.perf_counter()
    room.compute_rir()
    t2 = time.perf_counter()
    print(f"RIR computation time = {t2 - t1} s")

    # compute all the rt60
    rt60s = np.array(
        [
            pra.experimental.measure_rt60(room.rir[0][n], fs=room.fs)
            for n in range(n_sources)
        ]
    )
    distances = np.linalg.norm(source_locs - mic_loc[:, None], axis=0)

    rt60_med = np.median(rt60s)
    print("Median RT60:", rt60_med)

    critical_distance = 0.057 * np.sqrt(np.prod(room_dim) / rt60_med)
    print("Critical distance:", critical_distance)

    plt.vlines(critical_distance, 0, 1.05 * np.max(rt60s))
    plt.xlim([0, 1.01 * np.max(distances)])

    plt.plot(distances, rt60s, "x")
    plt.show()
