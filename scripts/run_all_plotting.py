import os
import sys
import yaml

plotting_type = input("What plotting do you want? (1: EarlyRun3 or 2: 2024): ").strip()
if plotting_type not in ["1", "2"]:
    print("Invalid choice. Please enter '1' or '2'.")
    sys.exit(1)
plotting_type = "" if plotting_type == "1" else "_2024"

config_file = f"configs/plotting{plotting_type}.yaml"
era = "EarlyRun3" if plotting_type == "" else "Run3_2024"

to_run = [
    {"channel": "tt", "process": "QCD", "binary": False, "leading": True, "regions": "all"},
    {"channel": "mt", "process": "QCD", "binary": False, "leading": False, "regions": "all"},
    {"channel": "mt", "process": "Wjets", "binary": False, "leading": False, "regions": "determination"},
    {"channel": "mt", "process": "WjetsMC", "binary": True, "leading": False, "regions": "all"},
    {"channel": "mt", "process": "ttbarMC", "binary": True, "leading": False, "regions": "determination"},
    {"channel": "et", "process": "QCD", "binary": False, "leading": False, "regions": "all"},
    {"channel": "et", "process": "Wjets", "binary": False, "leading": False, "regions": "determination"},
    {"channel": "et", "process": "WjetsMC", "binary": True, "leading": False, "regions": "all"},
    {"channel": "et", "process": "ttbarMC", "binary": True, "leading": False, "regions": "determination"},
]
for config in to_run:
    channel = config["channel"]
    process = config["process"]
    binary = "" if not config["binary"] else "--binary"
    leading = "" if not config["leading"] else "--leading"
    regions = config["regions"]
    region = "validation"
    if regions != "all":
        binary += f" --region {regions}"
        region = regions
    os.system(f"python scripts/plotting.py --config {config_file} --channel {channel} --process {process} --global_variables True {binary} --paper_plots")
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    output_dir = config_data['output_dir'].format(channel=channel, ff_process=process, global_str="Global")
    os.system(f"python scripts/non_closures.py --output-dir {output_dir} --channel {channel} --process {process} --region {region} {leading} --eras {era}")
