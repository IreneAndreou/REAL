from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import mplhep as hep
hep.style.use("CMS")

# Arguement parsing
parser = argparse.ArgumentParser(description="Plot non-closure for ML and Classical reweighting.")
parser.add_argument("--leading", action="store_true", help="Whether to plot leading or subleading tau metrics.")
parser.add_argument("--channel", type=str, default="tt", help="Channel to read metrics from (e.g. 'tt', 'mt', 'et').")
parser.add_argument("--process", type=str, default="QCD", help="Process to read metrics from (e.g. 'QCD', 'Wjets', 'ttbarMC').")
parser.add_argument("--region", type=str, default="all", help="Which regions to plot (e.g. 'all', 'determination', 'validation').")
parser.add_argument("--output-dir", type=str, required=True, help="Directory to save non-closure plots to.")
parser.add_argument("--eras", type=str, default="", help="Era label to include in plot (e.g. 'EarlyRun3', 'Run3_2024').")
args = parser.parse_args()

CMS_LABEL = dict(data=True, label="", com=13.6, loc=0, lumi=62.4 if args.eras == "EarlyRun3" else 109.08)
if args.process in ["WjetsMC", "ttbarMC"]:
    CMS_LABEL = dict(data=True, label="Simulation", com=13.6, loc=0, lumi=62.4 if args.eras == "EarlyRun3" else 109.08)


def read_metrics_txt(path):
    """Parse one {feature}_..._metrics.txt file and return arrays for ML(withGlobal) and Classical."""
    txt = open(path, "r").read()

    def grab_array(name):
        # matches: "Ratio values (X): [ ... ]" possibly spanning multiple lines
        m = re.search(rf"{re.escape(name)}:\s*\[([^\]]+)\]", txt, flags=re.MULTILINE | re.DOTALL)
        if not m:
            raise ValueError(f"Could not find array block '{name}' in {path}")
        arr_str = m.group(1).replace("\n", " ")
        arr_str = m.group(1).replace("--", "0")  # handle missing values denoted by "--"
        return np.fromstring(arr_str, sep=" ")

    r_ml = grab_array("Ratio values (ML Reweighting withGlobal)")
    e_ml = grab_array("Ratio errors (ML Reweighting withGlobal)")
    r_cl = grab_array("Ratio values (Classical Reweighting)")
    e_cl = grab_array("Ratio errors (Classical Reweighting)")
    bin_centers = grab_array("Bin centers")
    bin_widths = grab_array("Bin widths")
    ml_diff = re.findall(r"Max relative difference between ML reweightings:\s*([0-9.eE+-]+)", txt)
    ml_diff = float(ml_diff[0]) if ml_diff else np.nan
    closure_diff = re.findall(r"Max closure for Global model:\s*([0-9.eE+-]+)", txt)
    closure_diff = float(closure_diff[0]) if closure_diff else np.nan

    # optional: feature name
    mfeat = re.search(r"Feature:\s*([A-Za-z0-9_]+)", txt)
    feature = mfeat.group(1) if mfeat else "feature"

    # Parse constant fit results
    fit_cl = re.search(r"Classical Reweighting:\s*([0-9.eE+-]+)\s*±\s*([0-9.eE+-]+)", txt)
    fit_ml = re.search(r"ML Reweighting \(withGlobal\):\s*([0-9.eE+-]+)\s*±\s*([0-9.eE+-]+)", txt)
    const_cl = (float(fit_cl.group(1)), float(fit_cl.group(2))) if fit_cl else None
    const_ml = (float(fit_ml.group(1)), float(fit_ml.group(2))) if fit_ml else None

    return feature, r_ml, e_ml, r_cl, e_cl, bin_centers, bin_widths, ml_diff, closure_diff, const_ml, const_cl


def plot_nonclosure_centered(feature, r_ml, e_ml, r_cl, e_cl, bin_centers=None, bin_widths=None, outpath=None, const_ml=None, const_cl=None, tau_suffix="sublead"):
    # Non-closure defined as 1 - ratio (centered at 0)
    feature_key = feature
    d_ml = 1.0 - r_ml
    d_cl = 1.0 - r_cl

    x = np.arange(len(d_ml)) if bin_centers is None else np.asarray(bin_centers)

    fig, ax = plt.subplots(figsize=(12, 10))
    x = np.arange(len(d_ml)) if bin_centers is None else np.asarray(bin_centers)
    bin_widths = np.asarray(bin_widths) if bin_widths is not None else np.ones_like(x) * (x[1] - x[0])

    fig, ax = plt.subplots(figsize=(12, 10))

    # Always plot the binned step data up to threshold
    half = bin_widths / 2
    x_step = np.concatenate([x - half, [x[-1] + half[-1]]])
    def to_step(y): return np.concatenate([y, [y[-1]]])

    d_ml_quad = np.sqrt(np.abs(d_ml)**2 + e_ml**2)
    d_cl_quad = np.sqrt(np.abs(d_cl)**2 + e_cl**2)

    # For pt/mt_tot/pt_tt, replace bins above threshold with constant fit value added in quadrature
    use_const = feature in ["pt", "mt_tot", "pt_tt"] and const_ml is not None and const_cl is not None
    if use_const:
        threshold = 200 if tau_suffix == "lead" else 150
        nc_ml = abs(1.0 - const_ml[0])
        nc_cl = abs(1.0 - const_cl[0])
        syst_ml = np.sqrt(nc_ml**2 + const_ml[1]**2)
        syst_cl = np.sqrt(nc_cl**2 + const_cl[1]**2)

        above = x >= threshold
        d_ml_quad = np.where(above, syst_ml, d_ml_quad)
        d_cl_quad = np.where(above, syst_cl, d_cl_quad)

        # For stat uncertainty overlay: use fit stat error above threshold
        e_ml_plot = np.where(above, const_ml[1], e_ml)
        e_cl_plot = np.where(above, const_cl[1], e_cl)

        # Extend last bin edge to infinity for display
        x_step_plot = x_step.copy()
        x_step_plot[-1] = x_step[-2] + (x_step[-2] - x_step[-3])  # visual extension

        # Use bin edges below threshold as ticks, plus the threshold edge labelled infinity
        below_edges = x[~above] - half[~above]
        tick_positions = np.concatenate([below_edges, [threshold], [x_step_plot[-1]]])
        tick_labels = [str(int(t)) for t in below_edges] + [str(int(threshold)), r"$\infty$"]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(x_step[0], x_step_plot[-1])
    else:
        e_ml_plot = e_ml
        e_cl_plot = e_cl
        x_step_plot = x_step
        ax.set_xlim(x_step[0], x_step[-1])


    ax.fill_between(x_step_plot, 0, to_step(d_ml_quad), step="post", color="#5790fc", alpha=0.15, zorder=1)
    ax.fill_between(x_step_plot, 0, to_step(d_cl_quad), step="post", color="#fc5757", alpha=0.15, zorder=1)
    ax.step(x_step_plot, to_step(d_ml_quad), color="#5790fc", linewidth=1.8, where="post", label="MUFFIN Method", linestyle="-", zorder=2)
    ax.step(x_step_plot, to_step(d_cl_quad), color="#fc5757", linewidth=1.8, where="post", label=r"$\text{F}_{\text{F}}$ Method", linestyle="--", zorder=2)
    ax.plot(x, d_ml_quad, "o", color="#5790fc", markersize=5, zorder=3)
    ax.plot(x, d_cl_quad, "o", color="#fc5757", markersize=5, zorder=3)

    # Dashed lines showing pure statistical uncertainty per bin
    ax.step(x_step_plot, to_step(e_ml_plot), color="#5790fc", linewidth=1.2, where="post",
            linestyle="-.", zorder=2)
    ax.step(x_step_plot, to_step(e_cl_plot), color="#fc5757", linewidth=1.2, where="post",
            linestyle="-.", zorder=2)

    # Add to legend entry for stat-only uncertainty
    stat_line_handle = Line2D([0], [0], color="black", linewidth=1.5, 
                            marker="|", markersize=12, markeredgewidth=1.5,
                            linestyle="None",
                            label=r"Statistical Uncertainty")
    ymax = max(d_ml_quad.max(), d_cl_quad.max())
    # Classical as one-sided non-closure band (positive side only)
    # upper = np.abs(d_cl)
    # x_new = x - xerrors
    # x_new = np.concatenate([x_new, [x[-1] + xerrors[-1]]])
    # upper = np.concatenate([upper, [upper[-1]]])

    # ax.fill_between(
    #     x_new,
    #     0,
    #     upper,
    #     step="post", # matches binned style
    #     color="#fc5757",
    #     alpha=0.25,
    #     label=r"$\text{F}_{\text{F}}$ Method",
    #     zorder=1
    # )

    latex_feature_names = {
        "pt": r"$p_{\mathrm{T}}$ (GeV)",
        "mt_tot": r"$m_{\mathrm{T}}^{\mathrm{tot}}$ (GeV)",
        "pt_tt": r"$p_{\mathrm{T}}^{\tau\tau}$ (GeV)",
        'BDT_raw_score_tau': r'$\tau_h$ BDT score',
        'BDT_raw_score_higgs': r'Higgs BDT score',
        'BDT_raw_score_fake': r'Misidentified $\tau_h$ BDT score'
    }
    feature = latex_feature_names.get(feature, feature)

    ax.set_ylabel(r"Total Uncertainty ($\sqrt{(1-r)^2 + \sigma_r^2}$)", fontsize=32)
    ax.set_xlabel(feature, fontsize=32)
    # make sure x-axis and y-axis ticks do not overlap
    ax.tick_params(axis='x', which='major', pad=10, labelsize=32)
    ax.tick_params(axis='y', which='major', pad=10, labelsize=32)


    ymax = max(ymax * 1.2, 0.05)  # add some headroom, but enforce a minimum for visibility
    ax.set_ylim(0.0, ymax)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(tick*100)}%" for tick in yticks])

    ml_solid = Line2D([0], [0], color="#5790fc", linewidth=1.8, label="MUFFIN Method", linestyle="-")
    cl_solid = Line2D([0], [0], color="#fc5757", linewidth=1.8, label=r"$\text{F}_{\text{F}}$ Method", linestyle="--")
    stat_line_handle = Line2D([0], [0], color="black", linewidth=1.5,
                              linestyle="-.", label="Statistical\n" + r"Uncertainty ($\sigma_{\mathrm{r}}$)")

    # Add space at top for legend
    ax.set_ylim(0.0, ymax * 1.35)  # extend y-axis to make room for legend
    ax.legend(handles=[ml_solid, cl_solid, stat_line_handle], fontsize=30, loc="upper right", frameon=False)
    ax.grid(True, axis="y", alpha=0.2)
    hep.cms.label(**CMS_LABEL, ax=ax, fontsize=32)

    # Add region text labels in top-left corner
    region_text = {
        "determination": "Determination region",
        "validation": "Validation region",
    }.get(args.region, f"{str(args.region).capitalize()} region")

    ax.text(
        0.02, 0.95, region_text,
        ha="left", va="top", fontsize=30, transform=ax.transAxes,
        fontweight="bold"
    )

    bdt_region_text = {
        "BDT_raw_score_higgs": r"$H \to \tau\tau$ enriched region",
        "BDT_raw_score_tau": r"$Z \to \tau\tau$ enriched region",
        "BDT_raw_score_fake": r"$j \to \tau_h$ enriched region",
    }.get(feature_key)

    if bdt_region_text is not None:
        ax.text(
            0.02, 0.85, bdt_region_text,
            ha="left", va="top", fontsize=30, transform=ax.transAxes,
            fontweight="bold"
        )

    plt.tight_layout()
    if outpath:
        fig.savefig(outpath)
        plt.close(fig)
    else:
        plt.show()

    print(f"Plot saved to {outpath}")


if args.eras == "EarlyRun3":
    features_to_plot = ["pt", "mt_tot", "pt_tt", "BDT_raw_score_tau", "BDT_raw_score_higgs", "BDT_raw_score_fake"]
elif args.eras == "Run3_2024":
    features_to_plot = ["pt", "mt_tot", "pt_tt"]  # only kinematic variables for EarlyRun3 since BDTs not used in that analysis
for feature in features_to_plot:
    suffix = "lead" if args.leading else "sublead"
    feature, r_ml, e_ml, r_cl, e_cl, bin_centers, bin_widths, ml_diff, closure_diff, const_ml, const_cl = read_metrics_txt(f"{args.output_dir}/Run3_Combined/{args.region}/{feature}_subtraction_reweighting_{suffix}_metrics.txt")
    plot_nonclosure_centered(feature, r_ml, e_ml, r_cl, e_cl, bin_centers=bin_centers, bin_widths=bin_widths, outpath=f"{args.output_dir}/Run3_Combined/{args.region}/{feature}_nonclosure.pdf", const_ml=const_ml, const_cl=const_cl, tau_suffix=suffix)

# access all metrics_txt files in the determination directory and extract ML relative differences -- determined from determination region
# channels_processes = ["tt_QCD", "mt_QCD", "et_QCD", "mt_Wjets", "et_Wjets", "mt_WjetsMC", "et_WjetsMC", "mt_ttbarMC", "et_ttbarMC"]
# for cp in channels_processes:
#     ml_diffs = []
#     closure_diffs = []
#     metrics_files = glob.glob(f"/vols/cms/ia2318/REAL/outputs/plots/ARCReview_noGlobal_DannysComments/ARCReview/{cp}/Run3_Combined/*/*_metrics.txt")
#     for path in metrics_files:
#         feature, _, _, _, _, _, _, ml_diff, closure_diff = read_metrics_txt(path)
#         if feature.startswith("BDT_raw_score"):
#             continue  # skip BDT score features since they are not used in the final analysis
#         if feature in "n_jets n_bjets n_prebjets pt phi eta jpt_pt decay_Mode".split():
#             continue  # info used in training
#         # if ml_diff >= 20:
#         #     print(f"Warning: ML relative difference for {feature} in {path} is {ml_diff:.1f}%, which is quite high and may indicate an issue.")
#         ml_diffs.append((feature, ml_diff/100))  # convert percentage to fraction
#         closure_diffs.append((feature, closure_diff/100))  # convert percentage to fraction
#     print(f"Channel/Process: {cp}")
#     print(f"ML relative differences: {[f'{feat}: {diff:.2%}' for feat, diff in ml_diffs]}")
#     print(f"Global model closures: {[f'{feat}: {diff:.2%}' for feat, diff in closure_diffs]}")
#     print(f"Average ML relative difference: {np.nanmean([diff for _, diff in ml_diffs]):.2%}")
#     print(f"Average Global model closure: {np.nanmean([diff for _, diff in closure_diffs]):.2%}")
#     print("-" * 50)

# # Plot histogram of max relative differences between ML reweightings across features
# features, diffs = zip(*ml_diffs)
# plt.figure(figsize=(8, 4))
# plt.hist(diffs, bins=np.linspace(0, max(diffs)*1.1, 100), color="#5790fc", alpha=0.7)
# plt.ylabel("Max relative difference")
# plt.xlabel("Percentage")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.savefig("/vols/cms/ia2318/REAL/outputs/plots/ARCReview_noGlobal/ARCReview/ml_relative_differences.pdf")
# plt.close()
# print("ML relative differences plot saved to /vols/cms/ia2318/REAL/outputs/plots/ARCReview_noGlobal/ARCReview/ml_relative_differences.pdf")

# # Plot histogram of max closures for Global model across features
# features, closures = zip(*closure_diffs)
# plt.figure(figsize=(8, 4))
# plt.hist(closures, bins=np.linspace(0, max(closures)*1.1, 100), color="#fc5757", alpha=0.7)
# plt.ylabel("Max closure")
# plt.xlabel("Percentage")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.savefig("/vols/cms/ia2318/REAL/outputs/plots/ARCReview_noGlobal/ARCReview/global_model_closures.pdf")
# plt.close()
# print("Global model closures plot saved to /vols/cms/ia2318/REAL/outputs/plots/ARCReview_noGlobal/ARCReview/global_model_closures.pdf")
