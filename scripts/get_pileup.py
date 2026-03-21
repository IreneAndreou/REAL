import os
import pandas as pd

# =========================
# CONFIG
# =========================

PILEUP_CSV = "/vols/cms/ia2318/REAL/data_January26/pileup_2022_2023.csv"

BASE_DIR = "/vols/cms/ia2318/REAL/data_January26"

ERAS = ["Run3_2022", "Run3_2022EE", "Run3_2023", "Run3_2023BPix"]
REGIONS = ["determination_region", "validation_region"]

# =========================
# LOAD PILEUP MAP
# =========================

def load_pileup_map(csv_file):
    pileup_map = {}

    with open(csv_file) as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.strip().split(",")

            run = int(parts[0].split(":")[0])
            lumi = int(parts[1].split(":")[0])
            avgpu = float(parts[7])

            pileup_map[(run, lumi)] = avgpu

    print(f"Loaded {len(pileup_map)} (run,lumi) entries")
    return pileup_map


# =========================
# PROCESS ONE FILE
# =========================

def process_file(filepath, pileup_map):
    print(f"Processing: {filepath}")

    df = pd.read_parquet(filepath)

    # Detect MC vs DATA
    is_mc = df["run"].nunique() == 1 and df["run"].iloc[0] == 1

    if is_mc:
        df["pileup"] = 0.0
        print(" -> MC detected (run=1), setting pileup=0")

    else:
        # fast vectorized lookup
        df["pileup"] = [
            pileup_map.get((r, l), -1.0)
            for r, l in zip(df["run"], df["lumi"])
        ]

        missing = (df["pileup"] < 0).sum()
        if missing > 0:
            print(f" -> WARNING: {missing} missing PU entries")

    # new file path
    new_filepath = filepath.replace("data_January26", "data_with_pileup_January26")
    os.makedirs(os.path.dirname(new_filepath), exist_ok=True)
    df.to_parquet(new_filepath, index=False)
    print(f" -> Saved with pileup: {new_filepath}")


# =========================
# MAIN LOOP
# =========================

def main():
    pileup_map = load_pileup_map(PILEUP_CSV)

    for era in ERAS:
        for region in REGIONS:

            base_path = os.path.join(BASE_DIR, era, region)

            if not os.path.exists(base_path):
                continue

            for subdir in os.listdir(base_path):
                sub_path = os.path.join(base_path, subdir)

                if not os.path.isdir(sub_path):
                    continue

                # loop over parquet files
                for file in os.listdir(sub_path):
                    if file.endswith(".parquet"):
                        filepath = os.path.join(sub_path, file)
                        process_file(filepath, pileup_map)


if __name__ == "__main__":
    main()