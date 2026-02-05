import os

to_run = [
    {"channel": "tt", "process": "QCD", "global_variables": False, "binary": False},
    {"channel": "tt", "process": "QCD", "global_variables": True, "binary": False},
    {"channel": "mt", "process": "QCD", "global_variables": False, "binary": False},
    {"channel": "mt", "process": "QCD", "global_variables": True, "binary": False},
    {"channel": "mt", "process": "Wjets", "global_variables": False, "binary": False},
    {"channel": "mt", "process": "Wjets", "global_variables": True, "binary": False},
    {"channel": "mt", "process": "WjetsMC", "global_variables": False, "binary": True},
    {"channel": "mt", "process": "WjetsMC", "global_variables": True, "binary": True},
    {"channel": "mt", "process": "ttbarMC", "global_variables": False, "binary": True},
    {"channel": "mt", "process": "ttbarMC", "global_variables": True, "binary": True},
    {"channel": "et", "process": "QCD", "global_variables": False, "binary": False},
    {"channel": "et", "process": "QCD", "global_variables": True, "binary": False},
    {"channel": "et", "process": "Wjets", "global_variables": False, "binary": False},
    {"channel": "et", "process": "Wjets", "global_variables": True, "binary": False},
    {"channel": "et", "process": "WjetsMC", "global_variables": False, "binary": True},
    {"channel": "et", "process": "WjetsMC", "global_variables": True, "binary": True},
    {"channel": "et", "process": "ttbarMC", "global_variables": False, "binary": True},
    {"channel": "et", "process": "ttbarMC", "global_variables": True, "binary": True},
]
for config in to_run:
    channel = config["channel"]
    process = config["process"]
    global_variables = config["global_variables"]
    binary = "" if not config["binary"] else "--binary"
    os.system(f"python scripts/training.py --config configs/training.yaml --channel {channel} --process {process} --global_variables {str(global_variables)} {binary}")
