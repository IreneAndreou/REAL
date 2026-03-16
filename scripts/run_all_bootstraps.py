import os
import sys

training_type = input("What training do you want to create bootstraps for? (1: EarlyRun3 or 2: 2024): ").strip()
if training_type not in ["1", "2"]:
    print("Invalid choice. Please enter '1' or '2'.")
    sys.exit(1)
training_type = "" if training_type == "1" else "_2024"

config_file = f"configs/training{training_type}.yaml"

to_run = [
    {"channel": "tt", "process": "QCD", "global_variables": True, "binary": False},
    {"channel": "mt", "process": "QCD", "global_variables": True, "binary": False},
    {"channel": "mt", "process": "Wjets", "global_variables": True, "binary": False},
    {"channel": "mt", "process": "WjetsMC", "global_variables": True, "binary": True},
    {"channel": "mt", "process": "ttbarMC", "global_variables": True, "binary": True},
    {"channel": "et", "process": "QCD", "global_variables": True, "binary": False},
    {"channel": "et", "process": "Wjets", "global_variables": True, "binary": False},
    {"channel": "et", "process": "WjetsMC", "global_variables": True, "binary": True},
    {"channel": "et", "process": "ttbarMC", "global_variables": True, "binary": True},
]
for config in to_run:
    channel = config["channel"]
    process = config["process"]
    global_variables = config["global_variables"]
    binary = "" if not config["binary"] else "--binary"
    os.system(f"python scripts/bootstrap_inputs.py --config {config_file} --channel {channel} --process {process} --global_variables {str(global_variables)} {binary}")
    os.system(f"python scripts/bootstrap_submission.py --output_dir /vols/cms/ia2318/REAL/outputs/best_models/ARCReview_withGlobal/{channel}_{process}/ --ref_model /vols/cms/ia2318/REAL/outputs/best_models/ARCReview_withGlobal/{channel}_{process}/best_model.pkl")
