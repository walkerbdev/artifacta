"""
End-to-end tests for Genomics domain
DNA Sequence Analysis with parameter sweep

Run with: pytest tests/domains/test_genomics.py -v
"""

import os
import time

import numpy as np
import pytest

import artifacta as ds
from tests.helpers.notebook_html import create_genomics_notebook

# Path to example code
EXAMPLE_CODE_PATH = os.path.join(os.path.dirname(__file__), "../examples/genomics")


def run_parameter_sweep(project_name, base_config, param_variations, run_fn):
    """
    Helper to run multiple experiments with parameter variations.

    Args:
        project_name: Project name for the runs
        base_config: Base configuration dict
        param_variations: List of dicts with parameter overrides
        run_fn: Function(config, seed) that executes the run logic
    """
    for idx, variation in enumerate(param_variations):
        config = {**base_config, **variation}
        # Use project name in run name to avoid collisions across different sweep tests
        run_name = f"{project_name}-run-{idx + 1}"
        ds.init(project=project_name, name=run_name, config=config)
        # Pass seed to make results vary but be reproducible
        run_fn(config, seed=42 + idx)
        time.sleep(0.3)


@pytest.mark.e2e
def test_genomics(api_url):
    """Test 4: Genomics - DNA Sequence Analysis with parameter sweep"""

    def run_genomics_pipeline(config, seed=42):
        run = ds.get_run()

        # Set seed for reproducibility but vary across runs
        np.random.seed(seed)

        # Distribution: Variant quality scores - vary based on min_quality_score
        min_qual = config.get("min_quality_score", 30)
        # Higher min quality → higher average quality scores
        quality_mean = min_qual + 10
        ds.log(
            "variant_quality",
            ds.Distribution(
                values=np.random.gamma(quality_mean / 2, 2, 200).tolist(),
                groups=None,
            ),
        )

        # Note: Timeline/events primitive removed
        # Event durations should be logged as scalars if timing is important

        # Table: Top variants
        # Handle Table special case: convert 'rows' to 'data' and fix columns format
        cols = ["Chromosome", "Position", "Type", "Quality", "Depth"]
        col_types = ["string", "number", "string", "number", "number"]
        columns = [{"name": name, "type": typ} for name, typ in zip(cols, col_types)]

        ds.log(
            "top_variants",
            ds.Table(
                columns=columns,
                data=[
                    ["chr1", 12345678, "SNP", 95.2, 42],
                    ["chr2", 98765432, "INDEL", 88.7, 38],
                    ["chr3", 45678901, "SNP", 92.1, 51],
                    ["chr5", 23456789, "SNP", 97.8, 46],
                ],
            ),
        )

        # Log genomics variant calling code artifact with metadata
        run.log_input(
            EXAMPLE_CODE_PATH,
            metadata={
                "pipeline": "variant-calling",
                "reference_genome": "hg38",
                "variant_caller": "GATK",
                "language": "Python",
                "version": "4.2.0",
            },
        )

    base_config = {"variant_caller": "gatk", "reference_genome": "hg38"}
    param_variations = [
        {"min_quality_score": 30, "min_read_depth": 10},
        {"min_quality_score": 20, "min_read_depth": 10},
        {"min_quality_score": 30, "min_read_depth": 5},
    ]

    run_parameter_sweep("genomics-sweep", base_config, param_variations, run_genomics_pipeline)

    # Create rich notebook with sequence statistics
    sequence_stats = {
        "total_variants": 1247,
        "snps": 1089,
        "indels": 158,
        "avg_quality": 87.3,
        "avg_depth": 42.5,
        "transition_transversion_ratio": 2.1,
    }
    create_genomics_notebook(
        project_id="genomics-sweep",
        sequence_stats=sequence_stats,
        api_url=api_url,
    )
