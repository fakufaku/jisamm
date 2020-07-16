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
This script takes the output from the simulation and produces a number of plots
used in the publication.
"""
import argparse
import json
import os
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from data_loader import load_data

matplotlib.rc("pdf", fonttype=42)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Plot the data simulated by separake_near_wall"
    )
    parser.add_argument(
        "-p",
        "--pickle",
        action="store_true",
        help="Read the aggregated data table from a pickle cache",
    )
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="Display the plots at the end of data analysis",
    )
    parser.add_argument(
        "dirs",
        type=str,
        nargs="+",
        metavar="DIR",
        help="The directory containing the simulation output files.",
    )

    cli_args = parser.parse_args()
    plot_flag = cli_args.show
    pickle_flag = cli_args.pickle

    df, rt60, parameters = load_data(cli_args.dirs, pickle=pickle_flag)

    # Draw the figure
    print("Plotting...")

    # sns.set(style='whitegrid')
    # sns.plotting_context(context='poster', font_scale=2.)
    # pal = sns.cubehelix_palette(8, start=0.5, rot=-.75)

    df_melt = df.melt(id_vars=df.columns[:-5], var_name="metric")
    # df_melt = df_melt.replace(substitutions)

    # Aggregate the convergence curves
    df_agg = (
        df_melt.groupby(
            by=[
                "Algorithm",
                "Sources",
                "Interferers",
                "SINR",
                "Mics",
                "Iteration",
                "metric",
            ]
        )
        .mean()
        .reset_index()
    )

    all_algos = [
        "OverIVA-IP",
        "OverIVA-IP-NP",
        "OverIVA-IP2",
        "OverIVA-IP2-NP",
        "OverIVA-DX/BG",
        "FIVE",
        "OGIVEs",
        "AuxIVA-IP",
        "AuxIVA-IP2",
    ]

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
    pal = sns.cubehelix_palette(
        4, start=0.5, rot=-0.5, dark=0.3, light=0.75, reverse=True, hue=1.0
    )
    sns.set_palette(pal)

    if not os.path.exists("figures"):
        os.mkdir("figures")

    fig_dir = "figures/{}_{}_{}".format(
        parameters["name"], parameters["_date"], parameters["_git_sha"]
    )

    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    plt_kwargs = {
        (0, 0): {"xlim": [-0.05, 0.6], "xticks": [0.0, 0.3, 0.6]},
        (1, 0): {"xlim": [-0.05, 1.0], "xticks": [0.0, 0.5, 1.0]},
        (2, 0): {"xlim": [-0.05, 1.0], "xticks": [0.0, 0.5, 1.0]},
        (3, 0): {"xlim": [-0.05, 2.0], "xticks": [0.0, 1.0, 2.0]},
        (0, 1): {"xlim": [-0.05, 0.6], "xticks": [0.0, 0.3, 0.6]},
        (1, 1): {"xlim": [-0.05, 1.5], "xticks": [0.0, 0.5, 1.0, 1.5]},
        (2, 1): {"xlim": [-0.05, 3.0], "xticks": [0.0, 1.0, 2.0, 3.0]},
        (3, 1): {"xlim": [-0.05, 4.0], "xticks": [0.0, 2.0, 4.0]},
        (1, 2): {"xlim": [-0.05, 2.0], "xticks": [0.0, 1.0, 2.0]},
        (2, 2): {"xlim": [-0.05, 4.0], "xticks": [0.0, 2.0, 4.0]},
        (3, 2): {"xlim": [-0.05, 6.0], "xticks": [0.0, 2.0, 4.0, 6.0],},
    }

    full_width = 6.93  # inches, == 17.6 cm, double column width
    half_width = 3.35  # inches, == 8.5 cm, single column width

    # Second figure
    # Convergence curves: Time/Iteration vs SDR
    aspect = 0.8
    # height = ((full_width - 0.8) / len(parameters["sinr"])) / aspect
    height = 1.2
    sinr = parameters["sinr"][0]
    n_interferers = 10

    for r, metric in enumerate(["\u0394SI-SDR [dB]", "\u0394SI-SIR [dB]"]):

        select = np.logical_and(df_agg["SINR"] == sinr, df_agg["metric"] == metric)
        select = np.logical_and(select, df_agg["Interferers"] == n_interferers)

        local_algo = df_agg[select]["Algorithm"].unique()
        algo_order = [a for a in all_algos if a in local_algo]

        # select = np.logical_and(df_agg["Interferers"] == 5, select)
        g = sns.FacetGrid(
            df_agg[select],
            row="Sources",
            col="Mics",
            hue="Algorithm",
            hue_order=algo_order,
            hue_kws=dict(
                # marker=["o", "o", "s", "s", "d", "d", "^", "^"],
                # linewidth=[1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5],
            ),
            aspect=aspect,
            height=height,
            sharex=False,
            sharey="row",
            margin_titles=True,
            legend_out=False,
        )
        g.map(plt.plot, "Runtime [s]", "value", markersize=1.5)
        g.set_titles(col_template="{col_name} Mics", row_template="{row_name} Sources")

        # remove empty plot
        g.fig.delaxes(g.axes[2, 0])

        for c in range(4):
            g.facet_axis(1, c).set_title("")
        for c in range(1, 4):
            g.facet_axis(2, c).set_title("")

        g.facet_axis(0, 0).set_ylabel(metric)
        g.facet_axis(1, 0).set_ylabel(metric)
        g.facet_axis(2, 1).set_ylabel(metric)

        g.facet_axis(2, 1).tick_params(axis="y", which="major", labelleft=True)

        # plt.tight_layout(pad=0.5, w_pad=2.0, h_pad=2.0)
        # plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
        g.despine(left=True).add_legend(
            fontsize="x-small", loc="lower left", bbox_to_anchor=[-0.6, -3.3]
        )

        for (c, r), p in plt_kwargs.items():
            g.axes[r][c].set_xlim(p["xlim"]),
            g.axes[r][c].set_xticks(p["xticks"])

        # align the y-axis labels
        # g.fig.align_ylabels(g.axes[:, 0])

        for ext in ["pdf", "png"]:
            if metric[1:].startswith("SI-SDR"):
                metric_lbl = "SI-SDR"
            elif metric[1:].startswith("SI-SIR"):
                metric_lbl = "SI-SIR"
            else:
                metric_lbl = metric

            fig_fn = os.path.join(
                fig_dir,
                f"figure2_conv_interf{n_interferers}_sinr{sinr}_metric{metric_lbl}.{ext}",
            )
            plt.savefig(fig_fn, bbox_inches="tight")
        plt.close()

    if plot_flag:
        plt.show()
