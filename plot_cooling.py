import os
import re
import math
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def file_number(filename):
    match = re.search(r'_(\d+)\.txt$', filename)
    return int(match.group(1)) if match else 10**9


def load_case_directory(directory):
    """Load only time and surface_temp columns from all .txt files in a directory."""
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    files = sorted(files, key=file_number)
    if not files:
        raise FileNotFoundError(f"No .txt files found in {directory}")

    rows = []
    print(f"Reading {len(files)} files from {directory}")
    for fname in files:
        filepath = os.path.join(directory, fname)
        try:
            df = pd.read_csv(filepath, sep=r"\s+", usecols=['tot_time', 'surface_temp'], low_memory=True)
        except Exception as e:
            print(f"Warning: failed to read {filepath}: {e}")
            continue

        if df.empty:
            continue

        rows.append({
            'tot_time': float(df['tot_time'].iloc[0]),
            'surface_temp': float(df['surface_temp'].iloc[0])
        })
        del df
        gc.collect()

    df_out = pd.DataFrame(rows).sort_values(by='tot_time').reset_index(drop=True)
    return df_out


def plot_surface_temp_comparison(cases, labels, tmin=None, tmax=None, show=True, save_path=None):
    """Plot only the moving average of surface temperature vs time for three cases."""
    if len(cases) != len(labels):
        raise ValueError("cases and labels lists must have the same length")

    plt.figure(figsize=(10, 3.5))  # same aspect ratio
    plt.style.use('seaborn-v0_8-colorblind')

    for case_dir, label in zip(cases, labels):
        df = load_case_directory(case_dir)

        if tmin is None:
            tmin = df['tot_time'].min()
        if tmax is None:
            tmax = df['tot_time'].max()

        df = df[(df['tot_time'] >= tmin) & (df['tot_time'] <= tmax)]
        if df.empty:
            print(f"No data between {tmin} and {tmax} for {label}")
            continue

        # Convert to relative time (years)
        time_rel_years = df['tot_time'] - tmin

        # Compute moving average
        y_series = pd.Series(df['surface_temp'].astype(float).values)
        window = 500  # centered window
        y_ma = y_series.rolling(window=window, center=True, min_periods=1).mean().values

        # Plot only the moving average
        plt.plot(time_rel_years/3.154e7, y_ma, label=label, linewidth=2)

    plt.xlabel('Time (years)',fontsize=13)
    plt.ylabel('Surface Temperature (K)',fontsize=13)
    plt.legend(fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()



def main():
    """Specify three cases and their legend labels, then plot them together."""
    cases = [
        "/nfs/turbo/coe-chengcli/nocl3/case519_CO2_1e6_SO2_2e15_aero_10um_H2_on_scale_10.0/outputs",
        "/nfs/turbo/coe-chengcli/nocl3/case399_CO2_1e6_SO2_2e9_aero_0.1um_H2_on_scale_10.0/outputs",
        "/nfs/turbo/coe-chengcli/nocl3/case506_CO2_1e6_SO2_2e15_aero_0.1um_H2_on_scale_1.0/outputs"
    ]
    labels = [
        "SO2: 2e15 molecules/s/cm$^2$, 10 \u03BCm Aerosol",
        "SO2: 2e9 molecules/s/cm$^2$, 0.1 \u03BCm Aerosol",
        "SO2: 2e15 molecules/s/cm$^2$, 0.1 \u03BCm Aerosol"
    ]

    # Example time range and output path
    tmin = 8e6
    tmax = 1e9
    save_path = "tsurf_comparison.png"

    plot_surface_temp_comparison(cases, labels, tmin, tmax, show=True, save_path=save_path)


if __name__ == "__main__":
    main()

