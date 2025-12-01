import numpy as np
import pandas as pd

from modular_analysis.shared.data_models import ExperimentalDesign, GroupInfo, DesignType
from modular_analysis.statistical_analysis.tests.mixed_anova import MixedANOVA
from modular_analysis.statistical_analysis.tests.frequency_analysis_dependent import (
    FrequencyAnalyzerDependent,
)
from modular_analysis.statistical_analysis.tests.attenuation_analysis_dependent import (
    AttenuationAnalyzerDependent,
)
from modular_analysis.shared.config import ProtocolConfig


BETWEEN_LEVELS = ["WT", "Het", "KO"]
WITHIN_LEVELS = ["NoHeat", "Heat"]
SUBJECTS_PER_GROUP = 4


def build_design():
    groups = []
    factor_mapping = {}
    for between in BETWEEN_LEVELS:
        for within in WITHIN_LEVELS:
            name = f"{within}_{between}"
            groups.append(GroupInfo(name=name, folder_path=""))
            factor_mapping[name] = {"factor1": between, "factor2": within}

    manifest_rows = []
    for between in BETWEEN_LEVELS:
        for subj_idx in range(SUBJECTS_PER_GROUP):
            subject_id = f"{between}_S{subj_idx}"
            for within in WITHIN_LEVELS:
                manifest_rows.append(
                    {
                        "Subject_ID": subject_id,
                        "Group": between,
                        "Condition": within,
                        "Filename": f"{subject_id}_{within}.abf",
                    }
                )
    manifest = pd.DataFrame(manifest_rows)

    design = ExperimentalDesign(
        design_type=DesignType.MIXED_FACTORIAL,
        groups=groups,
        factor1_name="Group",
        factor2_name="Condition",
        factor_mapping=factor_mapping,
        between_factor_name="Group",
        within_factor_name="Condition",
        pairing_manifest=manifest,
    )
    return design


def build_measurement_tables(design, seed: int = 0):
    rng = np.random.default_rng(seed)
    between_effect = {"WT": 0.0, "Het": 2.5, "KO": 5.0}

    data = {}
    for group in design.groups:
        between = design.factor_mapping[group.name]["factor1"]
        within = design.factor_mapping[group.name]["factor2"]
        rows = []
        for subj_idx in range(SUBJECTS_PER_GROUP):
            subject_id = f"{between}_S{subj_idx}"
            marginal_val = 50 + between_effect[between] + rng.normal(0, 0.2, size=20)
            interaction_bonus = 6.0 if (between == "KO" and within == "Heat") else 0.0
            interaction_val = (
                marginal_val
                + (2.0 if within == "Heat" else 0.0)
                + interaction_bonus
                + rng.normal(0, 0.8, size=20)
            )
            rows.append(
                pd.DataFrame(
                    {
                        "Subject_ID": [subject_id] * 20,
                        "marginal_measure": marginal_val,
                        "interaction_measure": interaction_val,
                    }
                )
            )
        data[group.name] = pd.concat(rows, ignore_index=True)
    return data


class MockMixedANOVA(MixedANOVA):
    def __init__(self, synthetic_tables):
        super().__init__()
        self.synthetic_tables = synthetic_tables

    def _load_combined_data(self, group_name: str, base_path: str) -> pd.DataFrame:
        return self.synthetic_tables.get(group_name, pd.DataFrame()).copy()


def run_mixed_anova(design):
    measurement_tables = build_measurement_tables(design)
    analyzer = MockMixedANOVA(measurement_tables)
    results = analyzer.run_analysis(design, None, base_path="")

    marginal = [r for r in results if "Marginal" in r.test_name]
    assert marginal, "Mixed ANOVA should yield marginal contrasts"
    interaction = [r for r in results if "simple effect" in r.test_name]
    assert interaction, "Simple effects should still run for interaction signal"
    print(
        f"Mixed ANOVA: {len(marginal)} marginal comparisons, "
        f"{len(interaction)} simple-effect comparisons"
    )


def _build_step_table(between: str, within: str, subject_ids, steps: int, base: float, rng):
    rows = []
    for step in range(steps):
        row = {}
        for subj_id in subject_ids:
            subject_col = f"Subject_{subj_id}"
            value_col = f"Values_{subj_id}"
            row[subject_col] = subj_id
            row[value_col] = base + rng.normal(0, 0.1)
        rows.append(row)
    return pd.DataFrame(rows)


def build_frequency_tables(design, seed: int = 1, interaction_bonus: bool = False):
    rng = np.random.default_rng(seed)
    tables = {}
    steps = 4
    for between in BETWEEN_LEVELS:
        subject_ids = [f"{between}_S{i}" for i in range(SUBJECTS_PER_GROUP)]
        for within in WITHIN_LEVELS:
            base = 5.0 + (2.0 if between == "KO" else 1.0 if between == "Het" else 0.0)
            base += 1.0 if within == "Heat" else 0.0
            if interaction_bonus and between == "KO" and within == "Heat":
                base += 2.0
            tables[f"{within}_{between}"] = _build_step_table(
                between, within, subject_ids, steps, base, rng
            )
    return tables


def run_frequency_test(design):
    freq_tables = build_frequency_tables(design)
    analyzer = FrequencyAnalyzerDependent(protocol_config=ProtocolConfig(min_current=1.0, step_size=1.0))
    # Directly call mixed-model pointwise analyzer to avoid I/O
    anova_stats, posthoc_stats = analyzer._run_pointwise_mixed_model_frequency(
        freq_tables, analysis_type="current", design=design
    )
    marginal = [p for p in posthoc_stats if p.get("Test_Type", "").startswith("Marginal")]
    assert marginal, "Frequency mixed factorial should produce marginal tests"
    print(f"Dependent frequency analyzer: {len(marginal)} marginal comparisons generated")


def run_global_frequency_posthoc_test(design):
    freq_tables = build_frequency_tables(design, seed=5, interaction_bonus=True)
    analyzer = FrequencyAnalyzerDependent(protocol_config=ProtocolConfig(min_current=1.0, step_size=1.0))
    global_result = analyzer._run_unified_mixed_effects_frequency(freq_tables, "current", design)
    assert global_result, "Global mixed-effects model failed to return results"
    posthocs = global_result.get("posthocs", [])
    assert posthocs, "Global mixed-effects model should produce post-hoc contrasts"
    simple = [p for p in posthocs if p.get("Posthoc_Mode") == "Simple"]
    assert simple, "Expected simple-effect contrasts when interaction is present"
    print(f"Global mixed-effects posthocs: {len(simple)} simple-effects, {len(posthocs)} total")


def build_attenuation_tables(design, seed: int = 2):
    rng = np.random.default_rng(seed)
    tables = {}
    ap_count = 10
    for between in BETWEEN_LEVELS:
        subject_ids = [f"{between}_S{i}" for i in range(SUBJECTS_PER_GROUP)]
        for within in WITHIN_LEVELS:
            rows = []
            for ap_idx in range(ap_count):
                row = {}
                for subj_id in subject_ids:
                    subject_col = f"Subject_{subj_id}"
                    value_col = f"Values_{subj_id}"
                    row[subject_col] = subj_id
                    baseline = 10.0 - ap_idx
                    bonus = 3.0 if between == "KO" else 1.5 if between == "Het" else 0.0
                    row[value_col] = baseline + bonus + rng.normal(0, 0.2)
                rows.append(row)
            tables[f"{within}_{between}"] = pd.DataFrame(rows)
    return tables


def run_attenuation_test(design):
    attenuation_tables = build_attenuation_tables(design)
    analyzer = AttenuationAnalyzerDependent()
    anova_stats, posthoc_stats = analyzer._run_pointwise_mixed_model_attenuation(
        attenuation_tables, design
    )
    marginal = [p for p in posthoc_stats if p.get("Test_Type", "").startswith("Marginal")]
    assert marginal, "Attenuation mixed factorial should produce marginal tests"
    print(f"Dependent attenuation analyzer: {len(marginal)} marginal comparisons generated")


def main():
    design = build_design()
    run_mixed_anova(design)
    run_frequency_test(design)
    run_global_frequency_posthoc_test(design)
    run_attenuation_test(design)
    print("Dependent synthetic tests completed successfully.")


if __name__ == "__main__":
    main()

