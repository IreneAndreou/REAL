import os
import shutil
import yaml
import uproot

def count_entries(file_path):
    """Count the number of entries in the ROOT file."""
    try:
        with uproot.open(file_path) as f:
            tree = f["ntuple"]  # Update this with the actual tree name
            return tree.num_entries
    except Exception as e:
        print(f"Error counting entries in {file_path}: {e}")
        return 0

def preprocessing(source_dir, filenames, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

    for filename in filenames:
        src_file = os.path.join(source_dir, filename, "nominal/merged.root")
        dest_file = os.path.join(dest_dir, f"{filename}.root")

        shutil.copy(src_file, dest_file)
        print(f"Copied {src_file} to {dest_file}")

        print(f"Processing {dest_file}")
        os.system(
            "python scripts/scaling.py "
            f"--params configs/Run3_2022/params.yaml "
            f"--file_path {dest_file}"
        )


def merge_files(filenames, dest_dir, target_file):
    """Merge files and check the number of entries."""
    print(f"Merging files into {target_file}...")

    # Count entries in each file
    total_entries_before = 0
    for filename in filenames:
        scaled_file = os.path.join(dest_dir, f"{filename}_scaled.root")
        entries = count_entries(scaled_file)
        total_entries_before += entries
        print(f"File: {scaled_file}, Entries: {entries}")

    # Merge files
    files_to_merge = " ".join(
        [os.path.join(dest_dir, f"{filename}_scaled.root") for filename in filenames]
    )
    os.system(f"hadd -f {target_file} {files_to_merge}")

    # Count entries in the merged file
    total_entries_after = count_entries(target_file)

    print(f"Total entries before merging: {total_entries_before}")
    print(f"Total entries after merging: {total_entries_after}")

    if total_entries_before != total_entries_after:
        print("WARNING: Entry count mismatch after merging!")
    else:
        print("Entry count matches after merging.")

def main():
    # Load configuration
    with open("configs/Run3_2022/input_files.yaml", "r") as f:
        config = yaml.safe_load(f)

    source_dir = config["source_dir"]
    dest_dir = config["destination_dir"]

    # Preprocess and copy files for data and mc
    for category, filenames in config["source_files"].items():
        print(f"Processing {category} files...")
        preprocessing(source_dir, filenames, dest_dir)

        # Merge files
        target_file = os.path.join(
            dest_dir, f"{category}_all_events_Run3_2022.root"
        )
        merge_files(filenames, dest_dir, target_file)

if __name__ == "__main__":
    main()
