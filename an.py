import os
import re
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

def _detect_hpc_workers(default=None):
    """Detect number of workers from HPC scheduler environment variables."""
    # SLURM
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"])
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        val = os.environ["SLURM_JOB_CPUS_PER_NODE"]
        try:
            return int(val.split("(")[0])  # handle formats like "16(x2)"
        except:
            return int(val)

    # PBS / Torque
    if "PBS_NP" in os.environ:
        return int(os.environ["PBS_NP"])

    # LSF
    if "LSB_DJOB_NUMPROC" in os.environ:
        return int(os.environ["LSB_DJOB_NUMPROC"])

    # Fallback
    return default or os.cpu_count() or 1

def _process_case(args):
    case, base_dir, years = args
    outputs_path = os.path.join(base_dir, case, "outputs")
    if not os.path.isdir(outputs_path):
        return None, case, "no_outputs"

    pattern = re.compile(r"_(\d+)\.txt$")
    max_file = None
    max_num = -1

    try:
        with os.scandir(outputs_path) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                m = pattern.search(entry.name)
                if m:
                    num = int(m.group(1))
                    if num > max_num:
                        max_num = num
                        max_file = entry.path
    except Exception as e:
        return None, case, f"scan_error: {e}"

    if not max_file:
        return None, case, "no_output_files"

    try:
        with open(max_file, "r") as f:
            f.readline()
            second = f.readline()
            if not second:
                return None, case, "empty_file"

            elapsed_seconds = float(second.split()[0])
            elapsed_years = elapsed_seconds / (365 * 24 * 3600)
            return (case, elapsed_seconds, elapsed_years, max_file), case, None
    except Exception as e:
        return None, case, f"read_error: {e}"

def find_long_cases(base_dir=".", years=5, csv_file="long_cases.csv", workers=None):
    SECONDS_THRESHOLD = years * 365 * 24 * 3600
    cases = [c for c in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, c))]

    results = []
    errors = []

    if workers is None:
        workers = _detect_hpc_workers()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_case, (case, base_dir, years)): case for case in cases}
        for fut in as_completed(futures):
            res, case, err = fut.result()
            if res:
                _, secs, yrs, fpath = res
                if secs > SECONDS_THRESHOLD:
                    results.append(res)
            elif err:
                errors.append((case, err))

    if csv_file:
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Case", "Elapsed_Seconds", "Elapsed_Years", "Last_File"])
            writer.writerows(results)

        print(f"\nFound {len(results)} cases over {years} years. Results saved to {csv_file}")
    else:
        print(f"\nFound {len(results)} cases over {years} years.")

    if errors:
        print("\nCases with issues:")
        for case, err in errors:
            print(f"{case}: {err}")

    return results


import csv
import re

def write_sorted_case_names(csv_file, output_file):
    """
    Reads a CSV with a 'Case' column, sorts cases by numeric ID, and writes only the
    case names (one per line) to a new file.

    Parameters
    ----------
    csv_file : str
        Path to input CSV.
    output_file : str
        Path to write sorted case names.
    """
    cases = []

    # Extract case names from CSV
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_name = row["Case"]
            match = re.match(r"case(\d+)", case_name)
            if match:
                num = int(match.group(1))
                cases.append((num, case_name))

    # Sort by the numeric part
    cases.sort(key=lambda x: x[0])

    # Write only the case names
    with open(output_file, "w") as f:
        for _, case_name in cases:
            f.write(case_name + "\n")

    print(f"Wrote {len(cases)} case names to {output_file}")

cases = find_long_cases(base_dir=".", years=5, csv_file="analysis/long_cases.csv")
write_sorted_case_names("analysis/long_cases.csv", "analysis/sordid_cases.txt")
