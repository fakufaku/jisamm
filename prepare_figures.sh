#!/bin/bash

# make the figure with the RT60 distribution
python ./make_figure_rt60_hist.py

# figure with box-plots of SDR/SIR
# SNR 5d dB: data/20200226-225324_speed_contest_608cd46e9e
# SNR 10 dB: data/20200227-114954_speed_contest_f96a1824ad
python ./make_figure1_separation_hist.py \
  data/20200227-114954_speed_contest_f96a1824ad

# figure with wall-clock vs SDR/SIR
# SNR 5d dB: data/20200226-225324_speed_contest_608cd46e9e
# SNR 10 dB: data/20200227-114954_speed_contest_f96a1824ad
python ./make_figure2_speed_contest.py \
  data/20200227-114954_speed_contest_f96a1824ad

# figure with success prob
# all algos except demix/bg: data/20200225-200329_reverb_interf_performance_c0326397b0
# demix/bg only: data/20200305-062305_reverb_interf_performance_009c5c6241
python ./make_figure3_reverb_interf_performance.py \
  data/20200225-200329_reverb_interf_performance_c0326397b0 \
  data/20200305-062305_reverb_interf_performance_009c5c6241
