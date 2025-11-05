import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import time


def file_number(filename):
    """
    Extract the trailing number before .txt, after the last underscore.
    Example: output_case_foo_3714.txt -> 3714
    If no trailing number is found, return a large sentinel so those files sort last.
    """
    match = re.search(r'_(\d+)\.txt$', filename)
    return int(match.group(1)) if match else 10 ** 9


def process_case_directory(directory, species_candidates=None):
    """
    Read all .txt output files in a case directory (sorted by trailing number) and
    compute time series for derived quantities:
      - surface_pres: pres at the first row (pres[0])
      - cond_net: column-integrated (cond_product - cond_loss)
      - N_total: total molecules per area = sum( (p / (k_B * T)) * dz )
      - species column amounts: for species columns present (e.g. xH2S, xSO2,
        xH2SO4aer, xS8aer) compute sum( N_level * x_species )

    Returns a pandas.DataFrame indexed by time with these columns.
    """
    if species_candidates is None:
        species_candidates = ["xH2S", "xSO2", "xH2SO4aer", "xS8aer", "xS8"]

    # Delegate to load_case_directory which reads files once and returns an aggregated dataframe
    df_out = load_case_directory(directory, species_candidates=species_candidates)
    return df_out


def load_case_directory(directory, species_candidates=None):
    """
    Read all .txt files in `directory` once. Return a tuple (df_agg, raw_dfs) where
    df_agg is the aggregated time-series DataFrame (same format as process_case_directory used to return),
    and raw_dfs is a list of the per-file DataFrames (in sorted order).
    """
    if species_candidates is None:
        species_candidates = ["xH2S", "xSO2", "xH2SO4aer", "xS8aer"]

    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    files = sorted(files, key=file_number)
    if not files:
        raise FileNotFoundError(f"No .txt files found in {directory}")

    # Boltzmann constant (J/K)
    k_B = 1.380649e-23

    rows = []
    # helper to read only relevant columns to save memory
    def _read_relevant_columns(filepath, wanted_cols):
        # read header quickly by reading the first non-empty line and splitting
        available = []
        try:
            with open(filepath, 'r') as fh:
                for _ in range(5):
                    line = fh.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    # assume first useful line is header
                    available = line.split()
                    break
        except Exception:
            available = []

        # fall back to a safe pandas header read if the simple approach failed
        if not available:
            try:
                header = pd.read_csv(filepath, nrows=0, sep=r"\s+")
                available = list(header.columns)
            except Exception:
                available = []

        cols = [c for c in wanted_cols if c in available]
        if not cols:
            # nothing useful
            return pd.DataFrame()
        try:
            return pd.read_csv(filepath, sep=r"\s+", usecols=cols, low_memory=True)
        except Exception:
            # last-resort full read
            return pd.read_csv(filepath, sep=r"\s+", low_memory=True)

    n_files = len(files)
    print(f"Found {n_files} .txt files in {directory}", flush=True)
    start_t = time.time()
    for idx, fname in enumerate(files, start=1):
        filepath = os.path.join(directory, fname)
        # determine columns we may need for the per-file processing
        wanted = ['tot_time', 'pres', 'temp', 'dz', 'surface_temp', 'precip_rate', 'precip_type', 'cond_product', 'cond_loss'] + species_candidates
        try:
            df = _read_relevant_columns(filepath, wanted)
        except Exception as e:
            print(f"Warning: failed to read {filepath}: {e}", flush=True)
            continue

        # progress print every 50 files
        if idx % 50 == 0:
            elapsed = time.time() - start_t
            print(f"Read {idx}/{n_files} files; last file: {fname}; elapsed {elapsed:.1f}s", flush=True)

        time_0 = float(df['tot_time'].iloc[0])

        # surface pres/temp
        surface_pres = float(df['pres'].iloc[0]) if 'pres' in df.columns else math.nan
        surface_temp = float(df['surface_temp'].iloc[0]) if 'surface_temp' in df.columns else math.nan

        # precip
        precip_rate = float(df['precip_rate'].iloc[0]) if 'precip_rate' in df.columns else math.nan
        precip_type = str(df['precip_type'].iloc[0]) if 'precip_type' in df.columns else ''

        # cond sums
        cond_prod_total = df['cond_product'].astype(float).sum() if 'cond_product' in df.columns else math.nan
        cond_loss_total = df['cond_loss'].astype(float).sum() if 'cond_loss' in df.columns else math.nan
        cond_net = cond_prod_total - cond_loss_total if (not math.isnan(cond_prod_total) and not math.isnan(cond_loss_total)) else math.nan

        # N per level and N_total (ensure n * dz computed)
        if all(col in df.columns for col in ['pres', 'temp', 'dz']):
            p = df['pres'].astype(float).values
            T = df['temp'].astype(float).values
            dz = df['dz'].astype(float).values
            with np.errstate(divide='ignore', invalid='ignore'):
                n_m3 = p / (k_B * T)
                N_level = n_m3 * dz
            N_total = np.nansum(N_level)
        else:
            N_level = None
            N_total = math.nan

        species_amounts = {}
        if N_level is not None:
            for s in species_candidates:
                if s in df.columns:
                    x = df[s].astype(float).values
                    # species molecules per area = sum(n * x * dz) == sum(N_level * x)
                    species_amounts[s] = np.nansum(N_level * x)

        # capture some scalar first-row values (surface fluxes) if present
        scalar_firsts = {}
        for s in ['S8_sflx', 'H2SO4_sflx']:
            if s in df.columns:
                try:
                    scalar_firsts[s] = float(df[s].iloc[0])
                except Exception:
                    scalar_firsts[s] = math.nan

        # build the aggregated row for this file
        row = {
            'tot_time': time_0,
            'surface_pres': surface_pres,
            'surface_temp': surface_temp,
            'precip_rate': precip_rate,
            'precip_type': precip_type,
            'cond_net': cond_net,
            'cond_prod_total': cond_prod_total,
            'cond_loss_total': cond_loss_total,
            'N_total': N_total,
        }
        # add any species amounts computed for this file
        row.update(species_amounts)
        # add any scalar first-row values we captured
        row.update(scalar_firsts)
        rows.append(row)

        # free memory from per-file DataFrame promptly
        try:
            del df
            gc.collect()
        except Exception:
            pass

    if not rows:
        raise RuntimeError(f"No readable data rows in {directory}")

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values(by='tot_time').reset_index(drop=True)
    # Do not keep per-file DataFrames in memory (avoid OOM); return aggregated df only
    return df_out


def plot_cases(case_dirs, labels=None, species_to_plot=None, show=True):
    """Given a list of case directories, compute metrics and plot comparisons."""
    if labels is None:
        labels = [os.path.basename(d.rstrip("/")) for d in case_dirs]

    if species_to_plot is None:
        species_to_plot = ["xH2S", "xSO2", "xH2SO4aer", "xS8aer"]

    results = {}
    for d, lab in zip(case_dirs, labels):
        try:
            results[lab] = process_case_directory(d, species_candidates=species_to_plot)
        except Exception as e:
            print(f"Skipping {d}: {e}")

    if not results:
        print("No valid case data to plot. Exiting.")
        return

    # Setup plotting grid: surface_pres, precip_rate, cond_net, N_total, species
    n_species = len(species_to_plot)
    n_plots = 4 + n_species  # surface_pres, precip, cond_net, N_total, + species
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=False)

    # surface pressure
    ax = axes[0]
    for lab, df in results.items():
        if "surface_pres" in df.columns:
            ax.plot(df["tot_time"], df["surface_pres"], label=lab)
    ax.set_ylabel("Surface pressure")
    ax.set_title("Surface pressure vs Time")
    ax.legend()
    # precip rate
    ax = axes[1]
    for lab, df in results.items():
        if "precip_rate" in df.columns:
            # take log10 of precip_rate for plotting; skip non-positive values
            pr = np.array(df["precip_rate"].astype(float).values)
            t = np.array(df["tot_time"].values)
            with np.errstate(divide='ignore', invalid='ignore'):
                pr_log = np.where(pr > 0, np.log10(pr), np.nan)
            ax.plot(t, pr_log, label=lab)
    ax.set_ylabel("log10(Precip rate)")
    ax.set_title("Precipitation rate (log) vs Time")
    ax.legend()

    # cond net
    ax = axes[2]
    for lab, df in results.items():
        if "cond_net" in df.columns:
            # plot log10(-cond_net) so that positive plotted values correspond to destruction
            cn = np.array(df["cond_net"].astype(float).values)
            t = np.array(df["tot_time"].values)
            with np.errstate(divide='ignore', invalid='ignore'):
                mask = np.isfinite(cn) & (cn < 0)
                y = np.full_like(cn, np.nan, dtype=float)
                y[mask] = np.log10(-cn[mask])
            ax.plot(t, y, label=lab)
    ax.set_ylabel("log10(destruction) = log10(-cond_net)")
    ax.set_title("Column net production/destruction (log of destruction) vs Time")
    ax.legend()

    # N_total
    ax = axes[3]
    for lab, df in results.items():
        if "N_total" in df.columns:
            ax.plot(df["tot_time"], df["N_total"], label=lab)
    ax.set_ylabel("N_total (molecules / m^2)")
    ax.set_title("Total molecules per area vs Time")
    ax.legend()

    # species plots (each its own axis)
    for i, s in enumerate(species_to_plot):
        ax = axes[4 + i]
        plotted = False
        for lab, df in results.items():
            if s in df.columns:
                ax.plot(df["tot_time"], df[s], label=lab)
                plotted = True
        ax.set_ylabel(f"Column {s} (molecules / m^2)")
        ax.set_title(f"Column amount of {s} vs Time")
        if plotted:
            ax.legend()
        else:
            ax.text(0.5, 0.5, f"{s} not found in any case", ha="center")

    plt.tight_layout()
    if show:
        try:
            plt.show()
        except Exception as e:
            # e.g., headless environment: save the figure instead
            outfn = "plot_all_output.png"
            try:
                fig.savefig(outfn, dpi=200)
                print(f"Display not available, saved figure to {outfn}")
            except Exception as e2:
                print(f"Failed to show or save figure: {e}; {e2}")


def plot_parameter(directory, parameter, show=True, save_path=None, df_agg=None, raw_dfs=None, cutoff=None):
    """
    Simple single-parameter plot like the working example you provided.
    For scalar parameters (surface_temp, precip_rate, pres) the function takes the first
    row of each file and plots value vs tot_time. For precip_rate it groups by precip_type.
    """
    # If aggregated data was provided, use it; otherwise try to load once
    if df_agg is None:
        try:
            df_agg = load_case_directory(directory)
        except Exception:
            df_agg = None

    # allow short species names to map to aggregated columns
    species_aliases = {
        'H2S': 'xH2S',
        'SO2': 'xSO2',
        'H2SO4': 'xH2SO4aer',
        'S8': 'xS8aer',
    }

    scalar_keys = {"surface_temp", "precip_rate", "pres", "cond_net", "N_total", "surface_pres"} | set(species_aliases.keys())

    plt.figure(figsize=(5, 3))
    # flag to indicate we've already handled dual saving for this parameter (to avoid duplicate saves)
    handled_dual_save = False

    # If aggregated data present and parameter is a scalar, a species alias, or an aggregated species column,
    # plot directly from df_agg (fast). Only fall back to per-file profile reads for true profile parameters.
    if df_agg is not None and (parameter in scalar_keys or parameter in df_agg.columns or parameter in species_aliases):
        if parameter == "precip_rate":
            # group by precip_type and plot log10(precip_rate)
            groups = {}
            for _, row in df_agg.iterrows():
                key = row.get("precip_type", "")
                groups.setdefault(key, []).append((row["tot_time"], row["precip_rate"]))
            for key, values in groups.items():
                values = sorted(values, key=lambda x: x[0])
                x = np.array([v[0] for v in values])
                y = np.array([v[1] for v in values], dtype=float)
                with np.errstate(divide='ignore', invalid='ignore'):
                    y = np.where(y > 0, np.log10(y), np.nan)
                plt.plot(x, y, label=str(key))
        else:
            # determine which aggregated column to plot
            if parameter in species_aliases:
                col = species_aliases[parameter]
            elif parameter in df_agg.columns:
                col = parameter
            elif parameter == "pres":
                col = "surface_pres"
            else:
                col = parameter

            if col not in df_agg.columns:
                print(f"Parameter {parameter} not found in aggregated data (looking for column {col})")
                return

            # Special handling for cond_net: plot log10(-cond_net) and fit that transformed series
            if col == 'cond_net':
                cn = np.array(df_agg['cond_net'].astype(float).values)
                tvals = np.array(df_agg['tot_time'].values)
                with np.errstate(divide='ignore', invalid='ignore'):
                    mask = np.isfinite(cn) & (cn < 0)
                    y = np.full_like(cn, np.nan, dtype=float)
                    y[mask] = np.log10(-cn[mask])
                plt.plot(tvals, y, label='log10(-cond_net)')

                # Fit a linear trend to the transformed series (use only valid points)
                try:
                    valid = ~np.isnan(y)
                    if np.sum(valid) >= 2:
                        A = np.vstack([tvals[valid], np.ones_like(tvals[valid])]).T
                        m, b = np.linalg.lstsq(A, y[valid], rcond=None)[0]
                        y_fit = m * tvals + b
                        # compute R^2 for this fit
                        y_sel = y[valid]
                        y_pred_sel = m * tvals[valid] + b
                        ss_res = np.sum((y_sel - y_pred_sel) ** 2)
                        ss_tot = np.sum((y_sel - np.mean(y_sel)) ** 2)
                        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan

                        plt.plot(tvals, y_fit, color='C1', linestyle='--', label=f"fit: y={m:.3e}*t+{b:.3e}")
                        # annotate formula and R^2
                        txt = f"y={m:.3e}t+{b:.3e}, R^2={r2:.3f}"
                        plt.annotate(txt, xy=(0.02, 0.95), xycoords='axes fraction', va='top')

                        # also determine converted slope and stability if requested
                        # Convert slope from (units per year) to molecules/s^2/cm^2 using original fit_cond_net_trend conversion
                        try:
                            fitmeta = fit_cond_net_trend(df_agg, window_years=20, cutoff=cutoff)
                        except Exception:
                            fitmeta = None
                        if fitmeta is not None:
                            # show stability message requiring R^2 >= 0.9
                            stable = (not np.isnan(fitmeta.get('r2', np.nan)) and fitmeta['r2'] <= 0.1 and fitmeta.get('stable', False))
                            status = 'stable' if stable else 'unstable'
                            plt.annotate(f"Status: {status}; slope={fitmeta['slope_converted']:.3e}", xy=(0.02, 0.90), xycoords='axes fraction', va='top')
                except Exception:
                    pass
            else:
                # For surface_temp, plot raw values and overlay a moving average
                tvals = np.array(df_agg['tot_time'].values)
                if col == 'surface_temp':
                    yvals = np.array(df_agg['surface_temp'].astype(float).values)
                    plt.plot(tvals, yvals, label='surface_temp')
                    try:
                        window = 500
                        y_series = pd.Series(yvals)
                        y_ma = y_series.rolling(window=window, center=True, min_periods=1).mean().values
                        plt.plot(tvals, y_ma, color='C1', linewidth=2.0, label=f'surface_temp MA({window})')
                    except Exception:
                        pass
                elif col in ('S8_sflx', 'H2SO4_sflx'):
                    # these are scalar first-row fluxes captured during load; plot raw and moving average
                    yvals = np.array(df_agg[col].astype(float).values)
                    plt.plot(tvals, yvals, label=col)
                    try:
                        window = 500
                        y_series = pd.Series(yvals)
                        y_ma = y_series.rolling(window=window, center=True, min_periods=1).mean().values
                        plt.plot(tvals, y_ma, color='C1', linewidth=2.0, label=f'{col} MA({window})')
                    except Exception:
                        y_ma = None

                    # Explicitly save both linear and log versions (if save_path provided).
                    if save_path:
                        base_save = save_path
                        # save linear copy
                        try:
                            plt.gcf().savefig(base_save, dpi=200)
                            print(f"Saved linear flux figure to {base_save}")
                        except Exception as e:
                            print(f"Failed to save linear flux figure to {base_save}: {e}")

                        # Now create a separate figure for the log-scaled plot, masking non-positive values.
                        try:
                            fig_log = plt.figure(figsize=(5, 3))
                            ax_log = fig_log.add_subplot(1, 1, 1)
                            # mask non-positive values for log plotting
                            y_pos = np.where(np.array(yvals) > 0, yvals, np.nan)
                            ax_log.plot(tvals, y_pos, label=col)
                            if y_ma is not None:
                                # mask MA as well
                                y_ma_pos = np.where(np.array(y_ma) > 0, y_ma, np.nan)
                                ax_log.plot(tvals, y_ma_pos, color='C1', linewidth=2.0, label=f'{col} MA({window})')
                            ax_log.set_yscale('log')
                            ax_log.set_xlabel('Time')
                            ax_log.set_ylabel(col)
                            ax_log.set_title(f"{col} (log) vs Time")
                            try:
                                ax_log.legend()
                            except Exception:
                                pass
                            alt_save = base_save.replace('.png', '_log.png') if base_save.endswith('.png') else base_save + '_log'
                            fig_log.savefig(alt_save, dpi=200)
                            print(f"Saved log flux figure to {alt_save}")
                            plt.close(fig_log)
                        except Exception as e:
                            print(f"Failed to create/save log flux figure: {e}")

                        # mark handled so outer save block doesn't duplicate
                        handled_dual_save = True
                else:
                    plt.plot(df_agg["tot_time"], df_agg[col], label=col)

                # for species and other scalars (except surface_temp, N_total, and our new fluxes), keep log scale where previously used
                if parameter not in ('surface_temp', 'N_total'):
                    try:
                        plt.yscale('log')
                    except Exception:
                        pass

        plt.xlabel("Time")
        plt.ylabel(parameter)
        plt.title(f"{parameter} vs Time")
        try:
            plt.legend()
        except Exception:
            pass
    else:
        print('param not read in')
    '''
    else:
        # Fallback: iterate files sequentially to avoid storing all DataFrames at once
        data_dict = {}
        files = [f for f in os.listdir(directory) if f.endswith('.txt')]
        files = sorted(files, key=file_number)
        for fname in files:
            filepath = os.path.join(directory, fname)
            try:
                # read only tot_time and the target parameter to keep memory low
                header = pd.read_csv(filepath, nrows=0, sep=r"\s+")
                available = list(header.columns)
                cols = [c for c in ['tot_time', parameter] if c in available]
                if not cols:
                    continue
                df = pd.read_csv(filepath, sep=r"\s+", usecols=cols, low_memory=True)
            except Exception as e:
                print(f"Error reading {fname}: {e}")
                continue
            if "tot_time" not in df.columns or parameter not in df.columns:
                try:
                    del df
                    gc.collect()
                except Exception:
                    pass
                continue
            times = df["tot_time"].values
            values = df[parameter].values
            data_dict.setdefault('profile', []).extend(zip(times, values))
            try:
                del df
                gc.collect()
            except Exception:
                pass

        values = sorted(data_dict.get('profile', []), key=lambda x: x[0])
        if not values:
            print(f"No profile data found for {parameter}")
            return
        x = [v[0] for v in values]
        y = [v[1] for v in values]
        plt.plot(x, y, label=parameter)
        plt.xlabel('Time')
        plt.ylabel(parameter)
        plt.title(f"{parameter} vs Time")

    '''

    plt.tight_layout()
    # Save if requested (skip general save logic if flux-specific dual-save already handled)
    if save_path and not handled_dual_save:
        # Save both linear and log variants for parameters that were previously log-scaled.
        base_save = save_path
        try:
            # Save linear version first
            plt.gcf().savefig(base_save, dpi=200)
            print(f"Saved figure to {base_save}")
        except Exception as e:
            print(f"Failed to save figure to {base_save}: {e}")

        # Determine whether we should also save a linear or log alternative.
        # If yscale is log in the current figure, also save a linear version with same data.
        try:
            ax = plt.gca()
            is_log = ax.get_yscale() == 'log'
        except Exception:
            is_log = False

        # If currently log, also save a linear copy (switch to linear and save with _linear suffix).
        if is_log:
            try:
                ax.set_yscale('linear')
                alt_save = base_save.replace('.png', '_linear.png') if base_save.endswith('.png') else base_save + '_linear'
                plt.gcf().savefig(alt_save, dpi=200)
                print(f"Also saved linear-version figure to {alt_save}")
                # revert to log scale for display if needed
                ax.set_yscale('log')
            except Exception as e:
                print(f"Failed to save linear-version figure: {e}")

        # Additionally, for parameters that are normally plotted in log (precip_rate, cond_net, species),
        # if the current plot is linear (e.g., surface_temp), also produce a log-scaled copy where sensible.
        try:
            if not is_log and parameter not in ('surface_temp', 'N_total'):
                try:
                    ax.set_yscale('log')
                    alt_save2 = base_save.replace('.png', '_log.png') if base_save.endswith('.png') else base_save + '_log'
                    plt.gcf().savefig(alt_save2, dpi=200)
                    print(f"Also saved log-version figure to {alt_save2}")
                    ax.set_yscale('linear')
                except Exception as e:
                    print(f"Failed to save log-version figure: {e}")
        except Exception:
            pass

    if show:
        try:
            plt.show()
        except Exception:
            if not save_path:
                outfn = f"plot_{parameter}.png"
                try:
                    plt.gcf().savefig(outfn, dpi=200)
                    print(f"Saved figure to {outfn} (display not available)")
                except Exception as e:
                    print(f"Failed to show or save figure: {e}")
    else:
        # not showing; close the figure to free memory
        plt.close()


def fit_cond_net_trend(df_agg, window_years=20, cutoff=None):
    """
    Fit a linear trend to the last `window_years` of cond_net vs tot_time.

    Expects df_agg to contain 'tot_time' (in years) and 'cond_net' (molecules / m^2 / s?).
    Returns a dict with slope (per year), intercept, converted slope in target units
    (molecules/s^2/cm^2 assuming input was molecules/m^2 per year), and stability boolean
    if cutoff is provided.
    """
    if 'tot_time' not in df_agg.columns or 'cond_net' not in df_agg.columns:
        return None
    df = df_agg.dropna(subset=['tot_time', 'cond_net']).copy()
    if df.empty:
        return None
    t = np.array(df['tot_time'].values)
    y = np.array(df['cond_net'].values)

    # select last window_years
    t_max = np.max(t)
    cutoff_time = t_max - float(window_years)
    mask = t >= cutoff_time
    if not np.any(mask):
        return None
    t_sel = t[mask]
    y_sel = y[mask]

    # perform linear regression y = m * t + b
    A = np.vstack([t_sel, np.ones_like(t_sel)]).T
    m, b = np.linalg.lstsq(A, y_sel, rcond=None)[0]

    # compute R^2 for the fit
    y_pred = m * t_sel + b
    ss_res = np.sum((y_sel - y_pred) ** 2)
    ss_tot = np.sum((y_sel - np.mean(y_sel)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan

    # Convert slope to molecules / s^2 / cm^2 if input time is in years and cond_net units are molecules / m^2 / s
    # Here we assume tot_time is in years. Convert years->seconds and m^2->cm^2.
    seconds_per_year = 365.25 * 24 * 3600.0
    # m^2 to cm^2 => divide by 1e4
    slope_per_sec = m / seconds_per_year
    slope_converted = slope_per_sec / 1e4

    result = {
        'slope_year': m,
        'intercept_year': b,
        'slope_per_sec': slope_per_sec,
        'slope_converted': slope_converted,
        'n_points': len(t_sel),
        'r2': r2,
    }
    if cutoff is not None:
        # compare absolute values as requested
        # require both slope below cutoff and R^2 >= 0.9 to declare stable
        result['stable'] = (abs(slope_converted) < abs(cutoff)) and (not np.isnan(r2) and r2 >= 0.9)
        result['cutoff'] = cutoff
    else:
        result['stable'] = None
    return result


def main(dirpath):
    """
    Plot a single case directory supplied as the `dirpath` argument.

    This function does not prompt the user. Call it programmatically, e.g.:
        from plot_all import main
        main('/path/to/case/outputs')
    """
    if not dirpath:
        raise ValueError("dirpath must be provided to main()")
    if not os.path.isdir(dirpath):
        raise NotADirectoryError(f"Provided path is not a directory: {dirpath}")
    label = os.path.basename(dirpath.rstrip("/"))
    plot_cases([dirpath], labels=[label])


def main(dirpath, species_to_plot=None, save_dir=None, show=True, cutoff=None):
    """
    Nicely plot the main metrics you requested for a single case directory.

    Parameters:
      dirpath: path to the outputs directory containing .txt files
      species_to_plot: list of species columns to plot (defaults to xH2S, xSO2, xH2SO4aer, xS8aer)
      save_dir: if provided, save each parameter plot as a PNG into this directory
      show: whether to call plt.show() for each plot (set False for headless)
    """
    if not dirpath or not os.path.isdir(dirpath):
        raise NotADirectoryError(f"Provided path is not a directory: {dirpath}")

    if species_to_plot is None:
        species_to_plot = ["xH2S", "xSO2", "xH2SO4aer", "xS8aer"]

    # Also include scalar surface fluxes we want to plot
    flux_params = ["S8_sflx", "H2SO4_sflx"]

    params = ["pres", "precip_rate", "cond_net", "N_total", "surface_temp"] + species_to_plot + flux_params

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # derive a short label for this case from the directory path: use the second-to-last
    # chunk of the provided path (so callers can pass .../case_name/outputs and we pick
    # `case_name`). Fall back to the last chunk if the path is short.
    pnorm = os.path.normpath(dirpath)
    parts = pnorm.split(os.sep)
    label = parts[-2] if len(parts) >= 2 else parts[-1]

    # Load aggregated data once (memory-light)
    try:
        print(f"Starting load_case_directory for {dirpath}", flush=True)
        # ensure the scalar flux columns are included in the read request so we capture first-row values
        read_species = list(species_to_plot) + flux_params
        df_agg = load_case_directory(dirpath, species_candidates=read_species)
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {dirpath}: {e}")

    for p in params:
        fname = None
        if save_dir:
            safe_p = p.replace('/', '_')
            # prepend the case label to each filename to make outputs easy to distinguish
            fname = os.path.join(save_dir, f"{label}_{safe_p}.png")
        print(f"Plotting {p}...")
        try:
            plot_parameter(dirpath, p, show=show, save_path=fname, df_agg=df_agg, cutoff=cutoff)
        except Exception as e:
            print(f"Failed to plot {p}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate case outputs and plot requested diagnostics")
    parser.add_argument('input_dir', nargs='?', help='Path to case outputs directory (containing .txt files)')
    parser.add_argument('--show', action='store_true', help='Show plots interactively (default: no)')
    parser.add_argument('--cutoff', type=float, default=None, help='Cutoff slope (molecules/s^2/cm^2) for stability test')
    args = parser.parse_args()

    if not args.input_dir:
        parser.print_help()
        raise SystemExit(1)

    input_dir = args.input_dir
    # output plots go to ./plots relative to current working dir
    out_dir = os.path.join(os.getcwd(), 'plots')
    os.makedirs(out_dir, exist_ok=True)

    main(input_dir, save_dir=out_dir, show=args.show, cutoff=args.cutoff)

