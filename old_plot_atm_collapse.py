import os
import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import time


def file_number(filename):
    match = re.search(r'_(\d+)\.txt$', filename)
    return int(match.group(1)) if match else 10**9


def load_case_directory(directory, species_candidates=None):
    if species_candidates is None:
        species_candidates = ["xS8", "xS8aer"]

    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    files = sorted(files, key=file_number)
    if not files:
        raise FileNotFoundError(f"No .txt files found in {directory}")

    k_B = 1.380649e-23
    N_avo = 6.022e23
    rows = []

    def _read_relevant_columns(filepath, wanted_cols):
        try:
            header = pd.read_csv(filepath, nrows=0, sep=r"\s+")
            available = list(header.columns)
        except Exception:
            available = []
        cols = [c for c in wanted_cols if c in available]
        if not cols:
            return pd.DataFrame()
        return pd.read_csv(filepath, sep=r"\s+", usecols=cols, low_memory=True)

    print(f"Reading {len(files)} files from {directory}")
    for fname in files:
        filepath = os.path.join(directory, fname)
        wanted = ['tot_time', 'pres', 'temp', 'dz', 'surface_temp'] + species_candidates
        try:
            df = _read_relevant_columns(filepath, wanted)
        except Exception as e:
            print(f"Warning: failed to read {filepath}: {e}")
            continue

        if df.empty:
            continue

        time_0 = float(df['tot_time'].iloc[0])
        surface_pres = float(df['pres'].iloc[0]) if 'pres' in df.columns else math.nan
        surface_temp = float(df['surface_temp'].iloc[0]) if 'surface_temp' in df.columns else math.nan

        # compute N_total to get S8 column abundance
        if all(col in df.columns for col in ['pres', 'temp', 'dz']):
            p = df['pres'].astype(float).values
            T = df['temp'].astype(float).values
            dz = df['dz'].astype(float).values
            with np.errstate(divide='ignore', invalid='ignore'):
                n_m3 = p / (k_B * T)
                N_level = n_m3 * dz
        else:
            N_level = None

        s8_amount = np.nan
        for s in ['xS8', 'xS8aer']:
            if s in df.columns and N_level is not None:
                x = df[s].astype(float).values
                s8_amount = np.nansum(N_level * x) / N_avo
                break

        rows.append({
            'tot_time': time_0,
            'surface_pres': surface_pres,
            'surface_temp': surface_temp,
            'S8': s8_amount
        })

        del df
        gc.collect()

    if not rows:
        raise RuntimeError(f"No valid data rows in {directory}")

    df_out = pd.DataFrame(rows).sort_values(by='tot_time').reset_index(drop=True)
    return df_out


def plot_short_range(df, tmin, tmax, show=True, save_path=None):
    """Plot surface pressure separately, and surface temp & S8 together (dual y-axis)."""
    SECONDS_PER_YEAR = 3.154e7

    # Filter range
    df_short = df[(df['tot_time'] >= tmin) & (df['tot_time'] <= tmax)].copy()
    if df_short.empty:
        print(f"No data between {tmin} and {tmax}")
        return

    # Compute relative time since tmin (in years)
    df_short['time_rel_years'] = (df_short['tot_time'] - tmin) / SECONDS_PER_YEAR

    # Compact figure: 10x3 inches
    fig, axes = plt.subplots(2, 1, figsize=(10, 3.5), sharex=True)

    # --- Surface Pressure ---
    axes[0].plot(df_short['time_rel_years'], df_short['surface_pres'], label='Surface Pressure', color='C0')
    axes[0].set_ylabel('Pressure (Pa)', fontsize=14))
    #axes[0].legend(fontsize=15)

    # --- Temperature and S8 (dual axis) ---
    ax1 = axes[1]
    ax2 = ax1.twinx()
    ax2.set_yscale('log')

    l1 = ax1.plot(df_short['time_rel_years'], df_short['surface_temp'], color='C1', label='Surface Temp (K)')
    l2 = ax2.plot(df_short['time_rel_years'], df_short['S8'], color='C2', linestyle='--', label='S8 Column')

    ax1.set_ylabel('Temperature (K)', color='C1', fontsize=14)
    ax2.set_ylabel('S8 Column (mol/m²)', color='C2', fontsize=14))
    ax1.set_xlabel('Time (years)', fontsize=14))

    # Combine legends and enlarge font
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    #ax1.legend(lines, labels, loc='best', fontsize=15)

    # No plot titles
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main(input_dir, tmin, tmax, show=True, save_path=None):
    """Load directory data and plot surface pressure, temperature, and S8 over a short time range."""
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"{input_dir} is not a valid directory")

    df = load_case_directory(input_dir)
    plot_short_range(df, tmin, tmax, show=show, save_path=save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot surface pressure, temperature, and S8 over a short time range")
    parser.add_argument("input_dir", help="Path to directory with .txt outputs")
    parser.add_argument("--tmin", type=float, required=True, help="Minimum time for plotting window (seconds)")
    parser.add_argument("--tmax", type=float, required=True, help="Maximum time for plotting window (seconds)")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--save", help="Path to save output figure (optional)")
    args = parser.parse_args()

    main(args.input_dir, args.tmin, args.tmax, show=args.show, save_path=args.save)

