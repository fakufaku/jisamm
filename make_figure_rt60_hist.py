import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Creates histogram plot of RT60 in experiment"
    )
    parser.add_argument("file", type=str, help="File containing the room information")
    args = parser.parse_args()

    with open(args.file, "r") as f:
        room_data = json.load(f)

    rt60s = np.array(room_data["rt60s"])

    sns.set(
        style="whitegrid",
        context="paper",
        font_scale=0.75,
        rc={
            # 'figure.figsize': (3.39, 3.15),
            "lines.linewidth": 1.0,
            # 'font.family': 'sans-serif',
            # 'font.sans-serif': [u'Helvetica'],
            # 'text.usetex': False,
        },
    )

    # Histogram of RT60
    plt.figure(figsize=(1.5 * 1.675, 1.5 * 1.2))
    plt.hist(rt60s * 1000.0)
    plt.xlabel("$T_{60}$ [ms]")
    plt.ylabel("Frequency")
    sns.despine(offset=10, trim=False, left=True, bottom=True)
    plt.xticks([50, 150, 250, 350, 450])

    plt.tight_layout(pad=0.0)

    if not os.path.exists("figures"):
        os.mkdir("figures")

    fig_fn = f"figures/figure0_rt60_hist.pdf"
    plt.savefig(fig_fn)
