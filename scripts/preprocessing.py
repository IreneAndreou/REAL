import os
import shutil
import yaml


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
    print(f"Merging files into {target_file}...")
    files_to_merge = " ".join(
        [os.path.join(dest_dir, f"{filename}_scaled.root") for filename in filenames]
    )
    os.system(f"hadd -f {target_file} {files_to_merge}")  #TODO: check why data throws TObjArray::At and index 13 out of bounds error


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
