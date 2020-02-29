# Copyright (c) 2019 Robin Scheibler
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

matplotlib.rc("pdf", fonttype=42)

from data_loader import load_data


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
        action="store_false",
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
        "FIVE",
        "OverIVA-IP-Param",
        "OverIVA-IP2-Param",
        "OverIVA-IP-Block",
        "OverIVA-IP2-Block",
        "OverIVA-Demix/BG",
        "OGIVEs",
        "AuxIVA-IP",
        "AuxIVA-IP2",
        "PCA+AuxIVA-IP",
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
        # "improvements": {"ylim": [-5.5, 20.5], "yticks": [-5, 0, 5, 10, 15]},
        # "raw": {"ylim": [-5.5, 20.5], "yticks": [-5, 0, 5, 10, 15]},
        # "runtime": {"ylim": [-0.5, 40.5], "yticks": [0, 10, 20, 30]},
    }

    full_width = 6.93  # inches, == 17.6 cm, double column width
    half_width = 3.35  # inches, == 8.5 cm, single column width

    # Third figure

    # width = aspect * height
    aspect = 1  # width / height
    height = full_width / aspect

    all_algos_fig3 = all_algos

    for n_targets in parameters["n_targets"]:

        print("# Targets:", n_targets)

        select = (
            (df_melt["Iteration_Index"] == 1)
            & (df_melt["Sources"] == n_targets)
            & (df_melt["metric"] == "Success")
        )

        df_agg = df_melt[select]

        def draw_heatmap(*args, **kwargs):
            global ax
            data = kwargs.pop("data")
            d = data.pivot_table(
                index=args[1], columns=args[0], values=args[2], aggfunc=np.mean
            )
            ax_hm = sns.heatmap(d, **kwargs)
            ax_hm.invert_yaxis()
            ax = plt.gca()

        fg = sns.FacetGrid(df_agg, col="SINR", row="Algorithm", margin_titles=True, aspect=1, height=full_width/4.5)
        fg.map_dataframe(
            draw_heatmap, "Interferers", "Distance", "value", cbar=False, vmin=0., vmax=1., square=True
        )
        for ax in fg.axes.flat:
            plt.setp(ax.texts, text="")
        fg.set_titles(col_template="SINR = {col_name} [dB]", row_template="{row_name}")
        fg.set_xlabels("# Interferers")
        fg.set_ylabels("Ratio to Critical Distance")
        for ax in fg.axes.flat:
            plt.setp(ax.texts, bbox=dict(alpha=0.))

        PCM=ax.get_children()[0]
        cbar_ax = fg.fig.add_axes([1.015,0.2, 0.015, 0.6])
        plt.colorbar(PCM, cax=cbar_ax)

        for ext in ["pdf", "png"]:
            fig_fn = os.path.join(fig_dir, f"figure3_success_tgt{n_targets}.{ext}")
            # plt.savefig(fig_fn, bbox_extra_artists=all_artists, bbox_inches="tight")
            plt.savefig(fig_fn, bbox_inches="tight")

        plt.close()

    if plot_flag:
        plt.show()
