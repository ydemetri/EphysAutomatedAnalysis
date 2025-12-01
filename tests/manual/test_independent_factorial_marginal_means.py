import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from modular_analysis.shared.data_models import GroupInfo, ExperimentalDesign, DesignType
from modular_analysis.statistical_analysis.designs import DesignManager
from modular_analysis.statistical_analysis.tests.two_way_anova import TwoWayANOVA
from modular_analysis.statistical_analysis.tests.frequency_analysis import (
    FrequencyAnalyzer,
)
from modular_analysis.statistical_analysis.tests.attenuation_analysis import (
    AttenuationAnalyzer,
)
from modular_analysis.shared.config import ProtocolConfig


FACTOR1_LEVELS = ["Low", "Medium", "High"]
FACTOR2_LEVELS = ["Before", "After"]
UNBALANCED_FACTOR1_EFFECTS = {"Low": 0.0, "Medium": 5.0, "High": 10.0}
UNBALANCED_FACTOR2_EFFECTS = {"Before": 0.0, "After": 12.0}
UNBALANCED_COUNTS = {
    "Before": {"Low": 40, "Medium": 5, "High": 5},
    "After": {"Low": 3, "Medium": 30, "High": 60},
}


def build_design():
    groups = []
    factor_mapping = {}
    for f1 in FACTOR1_LEVELS:
        for f2 in FACTOR2_LEVELS:
            group_name = f"{f1}_{f2}"
            groups.append(GroupInfo(name=group_name, folder_path=""))
            factor_mapping[group_name] = {"factor1": f1, "factor2": f2}

    design = DesignManager.create_factorial_2x2(
        groups=groups,
        factor1_name="Dose",
        factor2_name="Time",
        factor_mapping=factor_mapping,
        name="Synthetic 3x2",
    )
    return design


def build_measurement_data(design, seed: int = 0) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    factor1_effect = {"Low": 0.0, "Medium": 4.0, "High": 8.0}

    measurement_data = {}
    for idx, group in enumerate(design.groups):
        mapping = design.factor_mapping[group.name]
        f1 = mapping["factor1"]
        f2 = mapping["factor2"]

        base = 50.0
        n = 80
        main_values = np.full(n, base + factor1_effect[f1]) + rng.normal(
            0, 0.1, size=n
        )

        interaction_bonus = 6.0 if (f1 == "High" and f2 == "After") else 0.0
        interaction_values = (
            base
            + factor1_effect[f1]
            + (2.0 if f2 == "After" else 0.0)
            + interaction_bonus
            + rng.normal(0, 0.8, size=n)
        )

        unbalanced = np.full(n, np.nan)
        count = UNBALANCED_COUNTS[f2][f1]
        mean_val = (
            UNBALANCED_FACTOR1_EFFECTS[f1] + UNBALANCED_FACTOR2_EFFECTS[f2]
        )
        noise = rng.normal(0, 0.05, size=count)
        unbalanced[:count] = mean_val + noise

        measurement_data[group.name] = pd.DataFrame(
            {
                "marginal_measure": main_values,
                "interaction_measure": interaction_values,
                "lsm_unbalanced": unbalanced,
            }
        )
    return measurement_data


class MockTwoWayANOVA(TwoWayANOVA):
    def __init__(self, synthetic_data: Dict[str, pd.DataFrame]):
        super().__init__()
        self.synthetic_data = synthetic_data

    def _load_combined_data(self, group_name: str, base_path: str) -> pd.DataFrame:
        return self.synthetic_data.get(group_name, pd.DataFrame()).copy()


def run_two_way_test(design):
    measurement_data = build_measurement_data(design)
    analyzer = MockTwoWayANOVA(measurement_data)
    results = analyzer.run_analysis(design, None, base_path="")

    marginal_results = [
        r for r in results if "Marginal t-test" in r.test_name
    ]
    assert marginal_results, "Expected marginal tests for independent factorial"

    # Validate LS-mean computation for unbalanced measurement
    lsm_results = [
        r
        for r in marginal_results
        if r.measurement == "lsm_unbalanced" and "Dose" in r.test_name
    ]
    assert lsm_results, "Expected marginal Dose contrasts for lsm_unbalanced"
    high_low = next(
        r
        for r in lsm_results
        if r.group1_name == "Dose=High" and r.group2_name == "Dose=Low"
    )
    avg_factor2 = sum(UNBALANCED_FACTOR2_EFFECTS.values()) / len(
        UNBALANCED_FACTOR2_EFFECTS
    )
    expected_high = UNBALANCED_FACTOR1_EFFECTS["High"] + avg_factor2
    expected_low = UNBALANCED_FACTOR1_EFFECTS["Low"] + avg_factor2
    assert math.isclose(
        high_low.group1_mean, expected_high, rel_tol=0.2, abs_tol=0.5
    ), f"High LS-mean mismatch: {high_low.group1_mean} vs {expected_high}"
    assert math.isclose(
        high_low.group2_mean, expected_low, rel_tol=0.2, abs_tol=0.5
    ), f"Low LS-mean mismatch: {high_low.group2_mean} vs {expected_low}"

    interaction_results = [
        r
        for r in results
        if "Pairwise t-test (Within" in r.test_name
        or "Pairwise t-test (Within" in r.test_name
    ]
    assert interaction_results, "Expected simple-effect comparisons"

    print(
        f"Two-way ANOVA: {len(marginal_results)} marginal comparisons, "
        f"{len(interaction_results)} simple-effect comparisons"
    )


def build_frequency_data(design, seed: int = 1, steps: int = 5, interaction_bonus: bool = False) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    factor1_effect = {"Low": 0.0, "Medium": 2.0, "High": 4.0}
    factor2_effect = {"Before": 0.0, "After": 1.5}

    data = {}
    for group in design.groups:
        f1 = design.factor_mapping[group.name]["factor1"]
        f2 = design.factor_mapping[group.name]["factor2"]
        rows = []
        for step in range(steps):
            base = 5.0 + step
            mean = base + factor1_effect[f1] + factor2_effect[f2]
            if interaction_bonus and f1 == "High" and f2 == "After":
                mean += 2.5
            values = mean + rng.normal(0, 0.2, size=6)
            rows.append({f"Values_{i}": values[i] for i in range(len(values))})
        data[group.name] = pd.DataFrame(rows)
    return data


def run_frequency_test(design):
    freq_data = build_frequency_data(design)
    analyzer = FrequencyAnalyzer(ProtocolConfig(min_current=0.0, step_size=1.0))

    freq_results = analyzer._run_pointwise_anova_frequency(
        freq_data, analysis_type="current", design=design
    )

    marginal = [
        r for r in freq_results if r.get("Test_Type") == "Marginal t-test"
    ]
    assert marginal, "Frequency analysis should produce marginal tests"
    print(f"Frequency analyzer: {len(marginal)} marginal comparisons generated")


def run_global_frequency_posthoc_test(design):
    freq_data = build_frequency_data(design, seed=5, steps=6, interaction_bonus=True)
    analyzer = FrequencyAnalyzer(ProtocolConfig(min_current=0.0, step_size=1.0))
    result = analyzer._run_unified_mixed_effects_frequency(freq_data, "current", design)
    assert result, "Global mixed model returned no result"
    posthocs = result.get("posthocs", [])
    assert posthocs, "Expected post-hoc contrasts for factorial design"
    simple = [p for p in posthocs if p.get("Posthoc_Mode") == "Simple"]
    assert simple, "Interaction-driven simple effects should be present"
    print(f"Global factorial frequency posthocs: {len(simple)} simple-effects, {len(posthocs)} total")


def run_single_factor_interaction_posthoc_test():
    groups = [GroupInfo(name=name, folder_path="") for name in ["G0", "G1", "G2"]]
    design = ExperimentalDesign(
        design_type=DesignType.INDEPENDENT_MULTI_GROUP,
        groups=groups,
        name="Single factor synthetic",
    )
    slopes = {"G0": -0.2, "G1": 0.0, "G2": 0.2}
    freq_data: Dict[str, pd.DataFrame] = {}
    steps = 6
    cells = 4
    for group in groups:
        rows = []
        for step_idx in range(steps):
            base = 30.0 + step_idx
            mean = base + slopes[group.name] * step_idx
            row = {f"Values_{i}": mean + (i - 1.5) * 0.05 for i in range(cells)}
            rows.append(row)
        freq_data[group.name] = pd.DataFrame(rows)
    analyzer = FrequencyAnalyzer(ProtocolConfig(min_current=0.0, step_size=1.0))
    result = analyzer._run_unified_mixed_effects_frequency(freq_data, "current", design)
    assert result, "Single-factor global mixed model returned no result"
    effects = dict(zip(result["effects"]["Effect"], result["effects"]["p-value"]))
    assert effects.get("Genotype:Current", 1.0) < 0.05, "Expected Genotype:Current interaction"
    assert result.get("posthocs"), "Interaction-triggered contrasts should be emitted"
    print(f"Single-factor global posthocs: {len(result['posthocs'])} contrasts")


def build_attenuation_data(design, seed: int = 2) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    factor1_effect = {"Low": 0.0, "Medium": 1.5, "High": 3.0}

    data = {}
    for group in design.groups:
        f1 = design.factor_mapping[group.name]["factor1"]
        rows = []
        for ap_idx in range(10):
            base = 10.0 - ap_idx
            values = base + factor1_effect[f1] + rng.normal(0, 0.3, size=6)
            rows.append({f"Value_{i}": values[i] for i in range(len(values))})
        data[group.name] = pd.DataFrame(rows)
    return data


def run_attenuation_test(design):
    attenuation_data = build_attenuation_data(design)
    analyzer = AttenuationAnalyzer()

    attenuation_results = analyzer._run_pointwise_anova_attenuation(
        attenuation_data, design=design
    )

    marginal = [
        r for r in attenuation_results if r.get("Test_Type") == "Marginal t-test"
    ]
    assert marginal, "Attenuation analysis should produce marginal tests"
    print(f"Attenuation analyzer: {len(marginal)} marginal comparisons generated")


def main():
    design = build_design()
    run_two_way_test(design)
    run_frequency_test(design)
    run_global_frequency_posthoc_test(design)
    run_single_factor_interaction_posthoc_test()
    run_attenuation_test(design)
    print("Synthetic tests completed successfully.")


if __name__ == "__main__":
    main()

