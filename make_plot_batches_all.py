import os
import re
import shutil

def main():
    template_file = "batch_template2.sh"
    submit_file = "submit_all_plots.sh"

    # read template
    with open(template_file, "r") as f:
        template_lines = f.readlines()

    # find case directories
    case_dirs = [
        d for d in os.listdir(".")
        if os.path.isdir(d) and re.match(r"case\d+_.+", d)
    ]

    batch_files = []

    for case_dir in case_dirs:
        abs_path = os.path.abspath(case_dir)
        # copy template
        new_lines = template_lines[:-1]  # everything except last line
        new_lines.append(f"python3 plot_all.py {abs_path}/outputs --cutoff 1e-7\n")

        batch_filename = f"plot_{case_dir}.sh"
        with open(batch_filename, "w") as f:
            f.writelines(new_lines)

        batch_files.append(batch_filename)

    # write submit_all_plots.sh
    with open(submit_file, "w") as f:
        for bf in batch_files:
            f.write(f"sbatch {bf}\n")

    print(f"Generated {len(batch_files)} batch files and {submit_file}")

if __name__ == "__main__":
    main()
