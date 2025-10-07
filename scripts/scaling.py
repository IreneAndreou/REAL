import os
import pandas as pd
import yaml
import logging
import argparse
from pathlib import Path
import pyarrow.parquet as pq

# Set up argument parser
parser = argparse.ArgumentParser(description="Scaling MC weights in parquet files.")
parser.add_argument("--params", required=True, type=str, help="Path to the YAML file containing parameters.")
parser.add_argument("--file_path", required=True, type=str, help="Path to the parquet file to be processed.")
parser.add_argument("--dest_file", required=True, type=str, help="Destination file name.")
args = parser.parse_args()

# ----------------------- Logging ----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------- Helpers -------------------------


def load_params(yaml_file):
    """Load parameters from a YAML file."""
    try:
        with open(yaml_file, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"YAML file '{yaml_file}' not found.")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file '{yaml_file}': {e}")
        raise


def get_columns(file_path):
    """Get the columns of a Parquet file using schema-only column listing"""
    return pq.ParquetFile(file_path).schema_arrow.names


def scaling(input_file, dest_file, params):
    """Apply scaling to the ROOT file."""
    input_file = Path(input_file)
    dest_file = Path(dest_file)
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    common_branches = ['event', 'run', 'lumi', 'original_index_1', 'original_index_2', 'charge_1', 'charge_2', 'pt_1', 'eta_1', 'phi_1', 'mass_1', 'pt_2', 'eta_2', 'phi_2', 'mass_2', 'os', 'dR', 'dphi', 'pt_tt', 'pt_vis', 'phi_vis', 'eta_vis', 'mt_1', 'mt_2', 'mt_lep', 'mt_tot', 'm_vis', 'met_pt', 'met_phi', 'met_covXX', 'met_covXY', 'met_covYY', 'met_dphi_1', 'met_dphi_2', 'trg_doubletau', 'trg_doubletauandjet', 'trg_doubletauandjet_2', 'trg_singletau', 'trg_singletau_2', 'idDeepTau2018v2p5VSjet_1', 'idDeepTau2018v2p5VSmu_1', 'idDeepTau2018v2p5VSe_1', 'idDeepTau2018v2p5VSjet_2', 'idDeepTau2018v2p5VSmu_2', 'idDeepTau2018v2p5VSe_2', 'idDeepTau2018v2p5noDAVSjet_1', 'idDeepTau2018v2p5noDAVSmu_1', 'idDeepTau2018v2p5noDAVSe_1', 'idDeepTau2018v2p5noDAVSjet_2', 'idDeepTau2018v2p5noDAVSmu_2', 'idDeepTau2018v2p5noDAVSe_2', 'rawDeepTau2018v2p5VSjet_1', 'rawDeepTau2018v2p5VSmu_1', 'rawDeepTau2018v2p5VSe_1', 'rawDeepTau2018v2p5VSjet_2', 'rawDeepTau2018v2p5VSmu_2', 'rawDeepTau2018v2p5VSe_2', 'rawDeepTau2018v2p5noDAVSjet_1', 'rawDeepTau2018v2p5noDAVSmu_1', 'rawDeepTau2018v2p5noDAVSe_1', 'rawDeepTau2018v2p5noDAVSjet_2', 'rawDeepTau2018v2p5noDAVSmu_2', 'rawDeepTau2018v2p5noDAVSe_2', 'rawPNetVSjet_1', 'rawPNetVSmu_1', 'rawPNetVSe_1', 'rawPNetVSjet_2', 'rawPNetVSmu_2', 'rawPNetVSe_2', 'decayMode_1', 'decayMode_2', 'decayModePNet_1', 'decayModePNet_2', 'probDM0PNet_1', 'probDM1PNet_1', 'probDM2PNet_1', 'probDM10PNet_1', 'probDM11PNet_1', 'probDM0PNet_2', 'probDM1PNet_2', 'probDM2PNet_2', 'probDM10PNet_2', 'probDM11PNet_2', 'n_jets', 'n_prebjets', 'n_bjets', 'mjj', 'jdeta', 'sjdphi', 'dijetpt', 'jpt_1', 'jeta_1', 'jphi_1', 'jpt_2', 'jeta_2', 'jphi_2', 'seeding_n_jets', 'seeding_mjj', 'seeding_jdeta', 'seeding_sjdphi', 'seeding_dijetpt', 'seeding_jpt_1', 'seeding_jeta_1', 'seeding_jphi_1', 'seeding_jpt_2', 'seeding_jeta_2', 'seeding_jphi_2', 'aco_pi_pi', 'aco_pi_rho', 'aco_pi_a1', 'aco_rho_pi', 'aco_rho_rho', 'aco_rho_a1', 'aco_a1_pi', 'aco_a1_rho', 'aco_a1_a1', 'aco_pi_a1_FASTMTT_NoMassConstraint', 'aco_rho_a1_FASTMTT_NoMassConstraint', 'aco_a1_pi_FASTMTT_NoMassConstraint', 'aco_a1_rho_FASTMTT_NoMassConstraint', 'aco_pi_a1_FASTMTT_MassConstraint', 'aco_rho_a1_FASTMTT_MassConstraint', 'aco_a1_pi_FASTMTT_MassConstraint', 'aco_a1_rho_FASTMTT_MassConstraint', 'PV_x', 'PV_y', 'PV_z', 'PVBS_x', 'PVBS_y', 'PVBS_z', 'ip_x_1', 'ip_y_1', 'ip_z_1', 'ip_x_2', 'ip_y_2', 'ip_z_2', 'ip_LengthSig_1', 'ip_LengthSig_2', 'hasRefitSV_1', 'hasRefitSV_2', 'sv_x_1', 'sv_y_1', 'sv_z_1', 'sv_x_2', 'sv_y_2', 'sv_z_2', 'pi_pt_1', 'pi_eta_1', 'pi_phi_1', 'pi_mass_1', 'pi_charge_1', 'pi_pdgId_1', 'pi_Energy_1', 'pi2_pt_1', 'pi2_eta_1', 'pi2_phi_1', 'pi2_mass_1', 'pi2_charge_1', 'pi2_pdgId_1', 'pi2_Energy_1', 'pi3_pt_1', 'pi3_eta_1', 'pi3_phi_1', 'pi3_mass_1', 'pi3_charge_1', 'pi3_pdgId_1', 'pi3_Energy_1', 'pi0_pt_1', 'pi0_eta_1', 'pi0_phi_1', 'pi0_mass_1', 'pi0_charge_1', 'pi0_pdgId_1', 'pi0_Energy_1', 'pi_pt_2', 'pi_eta_2', 'pi_phi_2', 'pi_mass_2', 'pi_charge_2', 'pi_pdgId_2', 'pi_Energy_2', 'pi2_pt_2', 'pi2_eta_2', 'pi2_phi_2', 'pi2_mass_2', 'pi2_charge_2', 'pi2_pdgId_2', 'pi2_Energy_2', 'pi3_pt_2', 'pi3_eta_2', 'pi3_phi_2', 'pi3_mass_2', 'pi3_charge_2', 'pi3_pdgId_2', 'pi3_Energy_2', 'pi0_pt_2', 'pi0_eta_2', 'pi0_phi_2', 'pi0_mass_2', 'pi0_charge_2', 'pi0_pdgId_2', 'pi0_Energy_2', 'FastMTT_mass', 'FastMTT_pt', 'FastMTT_pt_1', 'FastMTT_pt_2', 'FastMTT_mass_constraint', 'FastMTT_pt_constraint', 'FastMTT_pt_1_constraint', 'FastMTT_pt_2_constraint', 'weight', 'w_DoubleTauJetTrigger', 'w_DoubleTauTrigger', 'pion_E_split_1', 'pion_E_split_2', 'gen_boson_pT', 'gen_boson_mass', 'gen_boson_eta', 'gen_boson_phi', 'gen_taunus_pT', 'gen_taunus_phi', 'genPartFlav_1', 'genPartFlav_2', 'genPart_pt_1', 'genPart_eta_1', 'genPart_phi_1', 'genPart_pdgId_1', 'genPart_pt_2', 'genPart_eta_2', 'genPart_phi_2', 'genPart_pdgId_2', 'genVisTau_pt_1', 'genVisTau_eta_1', 'genVisTau_phi_1', 'genVisTau_mass_1', 'genVisTau_pt_2', 'genVisTau_eta_2', 'genVisTau_phi_2', 'genVisTau_mass_2', 'gen_decayMode_1', 'gen_decayMode_2', 'genIP_1_x', 'genIP_1_y', 'genIP_1_z', 'genIP_2_x', 'genIP_2_y', 'genIP_2_z', 'GenVsReco_PVBS_dxy', 'GenVsReco_PVBS_dz', 'GenVsReco_PV_dxy', 'GenVsReco_PV_dz', 'w_DY_soup', 'w_WJ_soup', 'w_DY_NLO_soup', 'w_Pileup', 'w_Zpt_Reweighting', 'w_Top_pt_Reweighting', 'w_ggH_QuarkMass_Effects', 'w_Electron_ID', 'w_Electron_Reco', 'w_Muon_ID', 'w_Muon_Isolation', 'w_Tau_ID', 'w_Tau_e_FakeRate', 'w_Tau_mu_FakeRate', 'w_Trigger']
    p = str(input_file)
    if "/mt/" in p or "/et/" in p:
        common_branches += ['iso_1', 'trg_singlemuon', 'trg_mt_cross', 'trg_singleelectron', 'trg_et_cross']
    # Retrieve scaling parameters (lumi)
    lumi = params.get("lumi", None)
    if lumi is None:
        logging.error(f"Luminosity not provided or invalid in the parameters file for {input_file}.")
        return False
    try:
        lumi = float(lumi)
    except Exception:
        logging.error(f"Invalid luminosity value '{lumi}' in parameters file for {input_file}.")
        return False

    sample = input_file.parent.parent.name

    # Selected columns
    try:
        available_columns = set(get_columns(input_file))
    except Exception as e:
        logging.error(f"Failed to inspect schema for {input_file}: {e}")
        return False
    selected_columns = [col for col in common_branches if col in available_columns]

    try:
        df = pd.read_parquet(input_file, columns=selected_columns)
    except Exception as e:
        logging.error(f"Failed to read {input_file}: {e}")
        return False
    is_data = sample.startswith(("Tau", "SingleMuon", "Muon", "EGamma"))

    if not is_data:
        s = params.get(sample, {})
        xs = s.get("xs", None)
        evt = s.get("eff", None)

        try:
            xs = float(xs) if xs else None
            evt = float(evt) if evt else None
        except Exception:
            logging.error(f"Non-numeric xs/eff for sample '{sample}': xs={xs}, eff={evt}")
            return False
        if xs is None or evt is None:
            logging.error(f"Missing xs/eff for sample '{sample}'.")
            return False
        if evt == 0:
            logging.error(f"Effective events is zero for sample '{sample}'.")
            return False

        # Calculate scaling
        wt_sf = lumi * xs / evt
        logging.info(f"Sample: {sample}, Cross-section: {xs}, Effective Events: {evt}, Weight SF: {wt_sf}")
        df["wt_sf"] = wt_sf * df["weight"]
    else:
        df["wt_sf"] = df["weight"]

    # Atomic write
    tmp = dest_file.with_suffix(dest_file.suffix + ".tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, dest_file)  # atomic on  POSIX
    except Exception as e:
        logging.error(f"Failed to write scaled data to {dest_file}: {e}")
        try:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            return False
        except Exception as e2:
            logging.error(f"Failed to clean up temp file {tmp}: {e2}")
            return False
    logging.info(f"Wrote scaled file: {dest_file}")
    return True

# Call the main function with parsed arguments


if __name__ == "__main__":
    params = load_params(args.params)
    ok = scaling(args.file_path, args.dest_file, params)
    if not ok:
        raise SystemExit(1)
