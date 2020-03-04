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

    algos_subst_rev = {
        "1": "FIVE",
        "2": "OverIVA-IP",
        "3": "OverIVA-IP2",
        "4": "OverIVA-IP-NP",
        "5": "OverIVA-IP2-NP",
        "6": "OverIVA-Demix/BG",
        "7": "OGIVEs",
        "8": "AuxIVA-IP",
        "9": "AuxIVA-IP2",
    }

    algos_subst = {
        "FIVE": "1",
        "OverIVA-IP": "2",
        "OverIVA-IP2": "3",
        "OverIVA-IP-NP": "4",
        "OverIVA-IP2-NP": "5",
        "OverIVA-Demix/BG": "6",
        "OGIVEs": "7",
        "AuxIVA-IP": "8",
        "AuxIVA-IP2": "9",
    }

    algo_order = {
        1: ["OverIVA-IP", "FIVE", "OGIVEs", "AuxIVA-IP2"],
        2: ["OverIVA-IP", "OverIVA-IP2", "AuxIVA-IP2"],
        3: ["OverIVA-IP", "OverIVA-IP2", "AuxIVA-IP2"],
    }

    if not os.path.exists("figures"):
        os.mkdir("figures")

    fig_dir = "figures/{}_{}_{}".format(
        parameters["name"], parameters["_date"], parameters["_git_sha"]
    )

    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    sns.set(
        style="whitegrid",
        context="paper",
        font_scale=0.5,
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

    # width = aspect * height
    aspect = 1  # width / height
    heights = {
        1: 0.74,
        2: 0.775,
        3: 0.775,
    }
    aspects = {
        1: 1.,
        2: 0.97,
        3: 0.97,
    }

    for n_targets in parameters["n_targets"]:

        height = heights[n_targets]
        aspect = aspects[n_targets]

        print("# Targets:", n_targets)

        select = (
            (df_melt["Iteration_Index"] == 1)
            & (df_melt["Sources"] == n_targets)
            & (df_melt["metric"] == "Success")
        )

        df_agg = df_melt[select].replace(algos_subst)
        """
        df_agg = df_agg[["SINR", "Algorithm", "Interferers", "Distance", "value"]]
        df_agg = df_agg[df_agg["Algorithm"].isin([algos_subst[a] for a in algo_order[n_targets]])]
        """

        def draw_heatmap(*args, **kwargs):
            global ax

            data = kwargs.pop("data")
            d = data.pivot_table(
                index=args[1], columns=args[0], values=args[2], aggfunc=np.mean,
            )

            ax_hm = sns.heatmap(d, **kwargs)
            ax_hm.invert_yaxis()

            ax = plt.gca()
            # import pdb; pdb.set_trace()

        fg = sns.FacetGrid(
            df_agg,
            col="SINR",
            row="Algorithm",
            row_order=[algos_subst[a] for a in algo_order[n_targets]],
            margin_titles=True,
            aspect=aspect,
            height=height,
        )
        fg.map_dataframe(
            draw_heatmap,
            "Interferers",
            "Distance",
            "value",
            cbar=False,
            vmin=0.0,
            vmax=1.0,
            xticklabels=[1, "", "", 4, "", "", 7, "", "", 10],
            yticklabels=[10, "", "", 40, "", "", 70, "", "", 100],
            square=True,
        )

        for the_ax in fg.axes.flat:
            plt.setp(the_ax.texts, text="")
        fg.set_titles(col_template="SINR = {col_name} [dB]", row_template="{row_name}")
        fg.set_xlabels("# Interferers", fontsize="small")
        fg.set_ylabels("Critical Distance [%]")
        for the_ax in fg.axes.flat:
            plt.setp(the_ax.texts, bbox=dict(alpha=0.0))
            plt.setp(the_ax.title, bbox=dict(alpha=0.0))
            for t in the_ax.texts:
                if t.get_text() in algos_subst_rev:
                    t.set_text(algos_subst_rev[t.get_text()])

        n_rows = len(fg.axes)
        for r in range(n_rows):
            for tick in fg.axes[r][0].yaxis.get_major_ticks():
                tick.label.set_fontsize(4)
        n_cols = len(fg.axes[n_rows - 1])
        for c in range(n_cols):
            for tick in fg.axes[n_rows - 1][c].xaxis.get_major_ticks():
                tick.label.set_fontsize(4)

        plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
        if n_targets > 1:
            fg.fig.subplots_adjust(wspace=-0.018)

        # set the colorbar on the side of all plots
        PCM = ax.get_children()[0]
        cbar_ax = fg.fig.add_axes([1.015, 0.2, 0.015, 0.6])
        plt.colorbar(PCM, cax=cbar_ax)

        for ext in ["pdf"]:
            fig_fn = os.path.join(fig_dir, f"figure3_success_tgt{n_targets}.{ext}")
            # plt.savefig(fig_fn, bbox_extra_artists=all_artists, bbox_inches="tight")
            # plt.savefig(fig_fn, bbox_extra_artists=[cbar_ax], bbox_inches="tight")
            plt.savefig(fig_fn, bbox_inches="tight")

        plt.clf()
        plt.close()

    if plot_flag:
        plt.show()
