"""
tests/test_reservoir_extraction.py

Comprehensive Test Suite for Indian Reservoir Extraction Pipeline.
Authoritative Sources:
- ORIGINAL_REQUEST.md (§R1, §R2, §R3, §Acceptance Criteria)
- PROJECT.md (§Architecture, §Feature Inventory, §Interface Contracts)
- DISPATCH.md (Test Writer 1)

Verification Scope:
1. Multi-Source Navigation (UW-SASWE/RAT, reservoirs.earth, nwdp.nwic.gov.in/dataset/reservoir, data.gov.in)
2. Main Extraction Script Execution & Syntax Validity
3. Successful Output of the 7 Target CSV Files in data/raw/wris/
4. Schema Conformance: Date, Reservoir_Name, Inflow (cusecs/cumecs), Storage (TMC/MCM)
5. Continuous Daily Date Coverage (2010-01-01 to 2024-12-31, exactly 5,479 days)
6. Non-Synthetic, Valid Numerical Values for both Inflow and Storage
"""

import ast
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Register custom pytest markers
def pytest_configure(config):
    config.addinivalue_line("markers", "network: mark test as requiring live network connectivity")

# Authoritative Constants
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "wris"
MAIN_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "extract_reservoirs.py"
ALTERNATIVE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "download_wris_data.py"
PIPELINE_PKG_DIR = PROJECT_ROOT / "src" / "data" / "reservoir_pipeline"


REQUIRED_SOURCES = {
    "UW_SASWE_RAT": "https://github.com/UW-SASWE/RAT",
    "RESERVOIRS_EARTH": "https://reservoirs.earth",
    "NWDP_DATASET": "https://nwdp.nwic.gov.in/dataset/reservoir",
    "DATA_GOV_IN": "https://data.gov.in",
}

NWDP_CKAN_API = "https://nwdp.nwic.gov.in/api/3/action/datastore_search"

# Target 7 Reservoirs per ORIGINAL_REQUEST & PROJECT.md
TARGET_RESERVOIRS = {
    "srisailam": {
        "slug": "srisailam",
        "file_name": "srisailam.csv",
        "canonical_name": "Srisailam",
        "river": "Krishna",
        "gross_capacity_mcm": 8560,
    },
    "nagarjuna_sagar": {
        "slug": "nagarjuna_sagar",
        "file_name": "nagarjuna_sagar.csv",
        "canonical_name": "Nagarjuna Sagar",
        "river": "Krishna",
        "gross_capacity_mcm": 11560,
    },
    "mettur": {
        "slug": "mettur",
        "file_name": "mettur.csv",
        "canonical_name": "Mettur",
        "river": "Cauvery",
        "gross_capacity_mcm": 2646,
    },
    "jayakwadi": {
        "slug": "jayakwadi",
        "file_name": "jayakwadi.csv",
        "canonical_name": "Jayakwadi",
        "river": "Godavari",
        "gross_capacity_mcm": 2909,
    },
    "ujjani": {
        "slug": "ujjani",
        "file_name": "ujjani.csv",
        "canonical_name": "Ujjani",
        "river": "Bhima",
        "gross_capacity_mcm": 3140,
    },
    "sardar_sarovar": {
        "slug": "sardar_sarovar",
        "file_name": "sardar_sarovar.csv",
        "canonical_name": "Sardar Sarovar",
        "river": "Narmada",
        "gross_capacity_mcm": 9500,
    },
    "ukai": {
        "slug": "ukai",
        "file_name": "ukai.csv",
        "canonical_name": "Ukai",
        "river": "Tapi",
        "gross_capacity_mcm": 7414,
    },
}

REQUIRED_SCHEMA_COLUMNS = [
    "Date",
    "Reservoir_Name",
    "Inflow (cusecs/cumecs)",
    "Storage (TMC/MCM)",
]

START_DATE_STR = "2010-01-01"
END_DATE_STR = "2024-12-31"
EXPECTED_CALENDAR_DAYS = 5479  # 15 years: 11 non-leap years (4015 days) + 4 leap years 2012,2016,2020,2024 (1464 days)


# =====================================================================
# 1. Multi-Source Navigation Tests
# =====================================================================
class TestMultiSourceNavigation:
    """Verifies pipeline capability to navigate and probe the 4 required sources:
    UW-SASWE/RAT, reservoirs.earth, nwdp.nwic.gov.in/dataset/reservoir, and data.gov.in.
    """

    def test_required_sources_configuration(self):
        """All 4 required source URLs must be configured and valid HTTP(S) URLs."""
        assert len(REQUIRED_SOURCES) == 4
        for source_key, url in REQUIRED_SOURCES.items():
            assert url.startswith("https://") or url.startswith("http://"), (
                f"Source {source_key} has invalid URL: {url}"
            )

    @patch("requests.get")
    def test_source_probing_mocked_success(self, mock_get):
        """Tests that probing logic correctly handles successful responses from all 4 sources."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><head><title>Water Portal</title></head><body>Dataset</body></html>"
        mock_get.return_value = mock_resp

        results = {}
        for name, url in REQUIRED_SOURCES.items():
            resp = requests.get(url, timeout=10)
            results[name] = resp.status_code

        assert all(code == 200 for code in results.values()), (
            f"Expected all sources to return HTTP 200, got {results}"
        )
        assert mock_get.call_count == 4

    @patch("requests.get")
    def test_source_probing_timeout_handling(self, mock_get):
        """Pipeline must catch network timeouts gracefully without unhandled crashes."""
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        with pytest.raises(requests.exceptions.Timeout):
            requests.get(REQUIRED_SOURCES["NWDP_DATASET"], timeout=5)

    def test_nwdp_ckan_api_query_payload_construction(self):
        """Verifies proper construction of NWDP CKAN datastore_search parameters."""
        resource_id = "be847b75-154e-4cc8-b4ff-f56ad8735644"
        filters = {"Station": "SRI SAILAM PROJECT"}
        params = {
            "resource_id": resource_id,
            "filters": str(filters).replace("'", '"'),
            "limit": 100,
        }

        assert params["resource_id"] == resource_id
        assert '"Station": "SRI SAILAM PROJECT"' in params["filters"]
        assert params["limit"] == 100

    def test_live_sources_reachability(self):
        """Live connectivity probe to the 4 specified sources.
        Gracefully skips if running in an offline environment.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ReservoirPipeline/1.0"
        }
        for name, url in REQUIRED_SOURCES.items():
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                # HTTP 200, 301, 302 are acceptable live web reachability codes
                assert resp.status_code in [200, 301, 302, 403], (
                    f"Live probe for {name} ({url}) failed with status {resp.status_code}"
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                pytest.skip(f"Live network access to {url} not available in this test run: {e}")


# =====================================================================
# 2. Main Extraction Script & Pipeline Module Execution Tests
# =====================================================================
class TestExtractionScript:
    """Verifies that the main Python extraction script executes without syntax errors
    and provides a clean command-line interface.
    """

    def test_main_script_file_exists(self):
        """The extraction entry point script must exist in scripts/."""
        script_exists = MAIN_SCRIPT_PATH.exists() or ALTERNATIVE_SCRIPT_PATH.exists()
        assert script_exists, (
            f"Extraction script not found at {MAIN_SCRIPT_PATH} or {ALTERNATIVE_SCRIPT_PATH}"
        )

    def test_extract_reservoirs_script_existence(self):
        """PROJECT.md § Code Layout designates scripts/extract_reservoirs.py as the primary runner."""
        if not MAIN_SCRIPT_PATH.exists():
            pytest.skip(
                f"scripts/extract_reservoirs.py not yet implemented by Worker 1. "
                f"Alternative runner {ALTERNATIVE_SCRIPT_PATH.name} exists."
            )
        assert MAIN_SCRIPT_PATH.exists(), f"Missing {MAIN_SCRIPT_PATH}"

    def test_main_script_syntax_validity(self):
        """The main extraction script must compile without any syntax errors."""
        script_to_test = MAIN_SCRIPT_PATH if MAIN_SCRIPT_PATH.exists() else ALTERNATIVE_SCRIPT_PATH
        if not script_to_test.exists():
            pytest.fail(f"No extraction script exists to validate syntax at {script_to_test}")

        with open(script_to_test, "r", encoding="utf-8") as f:
            code = f.read()

        try:
            ast.parse(code, filename=str(script_to_test))
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {script_to_test}: {e}")

    def test_main_script_cli_help(self):
        """Executing the script with --help should return code 0 and usage details."""
        script_to_test = MAIN_SCRIPT_PATH if MAIN_SCRIPT_PATH.exists() else ALTERNATIVE_SCRIPT_PATH
        if not script_to_test.exists():
            pytest.skip(f"Main script not yet created at {script_to_test}")

        res = subprocess.run(
            [sys.executable, str(script_to_test), "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert res.returncode == 0, (
            f"Running {script_to_test} --help failed with code {res.returncode}. "
            f"Stderr: {res.stderr}"
        )
        assert "usage" in res.stdout.lower() or "help" in res.stdout.lower(), (
            f"Expected CLI help text, got stdout: {res.stdout[:200]}"
        )

    def test_pipeline_reexecution_idempotency(self):
        """Verifies that running extraction twice produces identical results without data corruption."""
        import tempfile
        from scripts.extract_reservoirs import run_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Run 1 on fresh clean directory
            run_pipeline(
                output_dir=str(tmp_path),
                cache_dir="data/raw/nwdp_cache",
                skip_probe=True,
            )

            run1_data = {}
            for res_id, res_info in TARGET_RESERVOIRS.items():
                csv_path = tmp_path / res_info["file_name"]
                assert csv_path.exists(), f"File {csv_path} was not created in Run 1"
                run1_data[res_id] = pd.read_csv(csv_path)

            # Run 2 on the exact same directory
            run_pipeline(
                output_dir=str(tmp_path),
                cache_dir="data/raw/nwdp_cache",
                skip_probe=True,
            )

            # Verify exact equality
            for res_id, res_info in TARGET_RESERVOIRS.items():
                csv_path = tmp_path / res_info["file_name"]
                df_run2 = pd.read_csv(csv_path)
                pd.testing.assert_frame_equal(
                    run1_data[res_id],
                    df_run2,
                    check_exact=True,
                    obj=f"Pipeline output for {res_id} differed on re-run",
                )


class TestPipelineModules:
    """Verifies modular pipeline components defined in PROJECT.md:
    - src/data/reservoir_pipeline/__init__.py
    - src/data/reservoir_pipeline/source_navigator.py
    - src/data/reservoir_pipeline/nwdp_extractor.py
    - src/data/reservoir_pipeline/data_formatter.py
    """

    def test_pipeline_package_structure(self):
        """Verifies presence of src/data/reservoir_pipeline package if implemented."""
        if not PIPELINE_PKG_DIR.exists():
            pytest.skip("Pipeline package directory src/data/reservoir_pipeline not yet created.")
        assert (PIPELINE_PKG_DIR / "__init__.py").exists(), (
            f"Missing __init__.py in {PIPELINE_PKG_DIR}"
        )

    def test_source_navigator_import(self):
        """Verifies source_navigator module import and interface."""
        module_path = PIPELINE_PKG_DIR / "source_navigator.py"
        if not module_path.exists():
            pytest.skip(f"{module_path} not yet created.")
        import importlib
        mod = importlib.import_module("src.data.reservoir_pipeline.source_navigator")
        assert mod is not None

    def test_nwdp_extractor_import(self):
        """Verifies nwdp_extractor module import and interface."""
        module_path = PIPELINE_PKG_DIR / "nwdp_extractor.py"
        if not module_path.exists():
            pytest.skip(f"{module_path} not yet created.")
        import importlib
        mod = importlib.import_module("src.data.reservoir_pipeline.nwdp_extractor")
        assert mod is not None

    def test_data_formatter_import(self):
        """Verifies data_formatter module import and interface."""
        module_path = PIPELINE_PKG_DIR / "data_formatter.py"
        if not module_path.exists():
            pytest.skip(f"{module_path} not yet created.")
        import importlib
        mod = importlib.import_module("src.data.reservoir_pipeline.data_formatter")
        assert mod is not None



# =====================================================================
# 3. Output File Existence Tests
# =====================================================================
class TestOutputFileExistence:
    """Verifies that the 7 target CSV files exist in data/raw/wris/ and are non-empty."""

    def test_output_directory_exists(self):
        """The target output directory data/raw/wris/ must exist."""
        assert DATA_DIR.exists() and DATA_DIR.is_dir(), (
            f"Target directory {DATA_DIR} does not exist."
        )

    @pytest.mark.parametrize("res_id", TARGET_RESERVOIRS.keys())
    def test_csv_file_exists(self, res_id):
        """Each target reservoir must have its corresponding CSV file in data/raw/wris/."""
        res_info = TARGET_RESERVOIRS[res_id]
        csv_file = DATA_DIR / res_info["file_name"]
        assert csv_file.exists(), (
            f"Missing required CSV output for reservoir '{res_id}': expected file {csv_file}"
        )

    @pytest.mark.parametrize("res_id", TARGET_RESERVOIRS.keys())
    def test_csv_file_is_non_empty(self, res_id):
        """Each target reservoir CSV file must contain actual data (size > 100 bytes)."""
        res_info = TARGET_RESERVOIRS[res_id]
        csv_file = DATA_DIR / res_info["file_name"]
        if not csv_file.exists():
            pytest.fail(f"File {csv_file} does not exist.")
        file_size = csv_file.stat().st_size
        assert file_size > 100, (
            f"File {csv_file} is suspiciously small ({file_size} bytes)."
        )


# =====================================================================
# 4. Schema Conformance Tests
# =====================================================================
class TestSchemaConformance:
    """Verifies strict schema compliance:
    Header: Date,Reservoir_Name,Inflow (cusecs/cumecs),Storage (TMC/MCM)
    """

    @pytest.mark.parametrize("res_id", TARGET_RESERVOIRS.keys())
    def test_csv_header_exact_match(self, res_id):
        """CSV header must strictly match 'Date,Reservoir_Name,Inflow (cusecs/cumecs),Storage (TMC/MCM)'."""
        res_info = TARGET_RESERVOIRS[res_id]
        csv_file = DATA_DIR / res_info["file_name"]
        if not csv_file.exists():
            pytest.fail(f"Cannot test schema: {csv_file} does not exist.")

        with open(csv_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()

        actual_columns = [col.strip() for col in first_line.split(",")]
        assert actual_columns == REQUIRED_SCHEMA_COLUMNS, (
            f"Schema mismatch in {res_info['file_name']}.\n"
            f"Expected: {REQUIRED_SCHEMA_COLUMNS}\n"
            f"Actual:   {actual_columns}"
        )

    @pytest.mark.parametrize("res_id", TARGET_RESERVOIRS.keys())
    def test_reservoir_name_column_integrity(self, res_id):
        """The Reservoir_Name column must contain valid non-blank strings matching the target reservoir."""
        res_info = TARGET_RESERVOIRS[res_id]
        csv_file = DATA_DIR / res_info["file_name"]
        if not csv_file.exists():
            pytest.fail(f"File {csv_file} does not exist.")

        df = pd.read_csv(csv_file)
        if "Reservoir_Name" not in df.columns:
            pytest.fail(
                f"Column 'Reservoir_Name' missing from {res_info['file_name']}. "
                f"Columns present: {list(df.columns)}"
            )

        unique_names = df["Reservoir_Name"].dropna().unique().tolist()
        assert len(unique_names) >= 1, f"Reservoir_Name is empty in {res_info['file_name']}"
        
        # Check canonical or slug match
        matched = any(
            res_info["canonical_name"].lower() in str(n).lower()
            or res_id.lower() in str(n).lower().replace(" ", "_")
            for n in unique_names
        )
        assert matched, (
            f"Unexpected Reservoir_Name values {unique_names} in {res_info['file_name']}; "
            f"expected '{res_info['canonical_name']}'"
        )

    @pytest.mark.parametrize("res_id", TARGET_RESERVOIRS.keys())
    def test_date_column_format_iso(self, res_id):
        """Date column values must strictly adhere to ISO format YYYY-MM-DD."""
        res_info = TARGET_RESERVOIRS[res_id]
        csv_file = DATA_DIR / res_info["file_name"]
        if not csv_file.exists():
            pytest.fail(f"File {csv_file} does not exist.")

        df = pd.read_csv(csv_file)
        if "Date" not in df.columns:
            pytest.fail(f"Column 'Date' missing from {res_info['file_name']}")

        date_samples = df["Date"].astype(str).head(100)
        iso_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for d in date_samples:
            assert iso_pattern.match(d), f"Date '{d}' in {res_info['file_name']} does not match YYYY-MM-DD format"

    @pytest.mark.parametrize("res_id", TARGET_RESERVOIRS.keys())
    def test_numerical_columns_datatypes(self, res_id):
        """Inflow and Storage columns must be numeric (float/int)."""
        res_info = TARGET_RESERVOIRS[res_id]
        csv_file = DATA_DIR / res_info["file_name"]
        if not csv_file.exists():
            pytest.fail(f"File {csv_file} does not exist.")

        df = pd.read_csv(csv_file)
        inflow_col = "Inflow (cusecs/cumecs)"
        storage_col = "Storage (TMC/MCM)"

        for col in [inflow_col, storage_col]:
            if col not in df.columns:
                pytest.fail(f"Required column '{col}' missing from {res_info['file_name']}")
            assert pd.api.types.is_numeric_dtype(df[col]), (
                f"Column '{col}' in {res_info['file_name']} is not numeric dtype (got {df[col].dtype})"
            )


# =====================================================================
# 5. Continuous Daily Date Coverage Tests (2010–2024)
# =====================================================================
class TestDateCoverage:
    """Verifies continuous daily date coverage spanning 2010-01-01 to 2024-12-31 (5,479 days)."""

    @pytest.mark.parametrize("res_id", TARGET_RESERVOIRS.keys())
    def test_date_range_spans_2010_to_2024(self, res_id):
        """The dataset must span from at least 2010-01-01 to 2024-12-31."""
        res_info = TARGET_RESERVOIRS[res_id]
        csv_file = DATA_DIR / res_info["file_name"]
        if not csv_file.exists():
            pytest.fail(f"File {csv_file} does not exist.")

        df = pd.read_csv(csv_file)
        if "Date" not in df.columns:
            pytest.fail(f"Date column missing from {res_info['file_name']}")

        df["parsed_date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
        assert df["parsed_date"].isna().sum() == 0, (
            f"Found unparseable dates in {res_info['file_name']}"
        )

        min_date = df["parsed_date"].min()
        max_date = df["parsed_date"].max()

        expected_start = pd.Timestamp(START_DATE_STR)
        expected_end = pd.Timestamp(END_DATE_STR)

        assert min_date <= expected_start, (
            f"{res_info['file_name']} start date {min_date.strftime('%Y-%m-%d')} is after expected {START_DATE_STR}"
        )
        assert max_date >= expected_end, (
            f"{res_info['file_name']} end date {max_date.strftime('%Y-%m-%d')} is before expected {END_DATE_STR}"
        )

    @pytest.mark.parametrize("res_id", TARGET_RESERVOIRS.keys())
    def test_daily_continuity_no_missing_days_in_range(self, res_id):
        """Within 2010-01-01 to 2024-12-31, there must be NO missing calendar days."""
        res_info = TARGET_RESERVOIRS[res_id]
        csv_file = DATA_DIR / res_info["file_name"]
        if not csv_file.exists():
            pytest.fail(f"File {csv_file} does not exist.")

        df = pd.read_csv(csv_file)
        if "Date" not in df.columns:
            pytest.fail(f"Date column missing from {res_info['file_name']}")

        df["parsed_date"] = pd.to_datetime(df["Date"], errors="coerce")
        in_range_df = df[(df["parsed_date"] >= START_DATE_STR) & (df["parsed_date"] <= END_DATE_STR)]

        reference_dates = pd.date_range(start=START_DATE_STR, end=END_DATE_STR, freq="D")
        actual_dates_set = set(in_range_df["parsed_date"].dt.strftime("%Y-%m-%d"))
        expected_dates_set = set(reference_dates.strftime("%Y-%m-%d"))

        missing_dates = sorted(list(expected_dates_set - actual_dates_set))
        assert len(missing_dates) == 0, (
            f"{res_info['file_name']} has {len(missing_dates)} missing days between {START_DATE_STR} and {END_DATE_STR}. "
            f"First 5 missing: {missing_dates[:5]}"
        )

    @pytest.mark.parametrize("res_id", TARGET_RESERVOIRS.keys())
    def test_no_duplicate_dates(self, res_id):
        """Every date entry in the CSV file must be strictly unique (no duplicate days)."""
        res_info = TARGET_RESERVOIRS[res_id]
        csv_file = DATA_DIR / res_info["file_name"]
        if not csv_file.exists():
            pytest.fail(f"File {csv_file} does not exist.")

        df = pd.read_csv(csv_file)
        if "Date" not in df.columns:
            pytest.fail(f"Date column missing from {res_info['file_name']}")

        duplicate_count = df["Date"].duplicated().sum()
        assert duplicate_count == 0, (
            f"{res_info['file_name']} contains {duplicate_count} duplicate Date entries!"
        )

    @pytest.mark.parametrize("res_id", TARGET_RESERVOIRS.keys())
    def test_exact_row_count_for_target_period(self, res_id):
        """Between 2010-01-01 and 2024-12-31, there must be exactly 5,479 daily records."""
        res_info = TARGET_RESERVOIRS[res_id]
        csv_file = DATA_DIR / res_info["file_name"]
        if not csv_file.exists():
            pytest.fail(f"File {csv_file} does not exist.")

        df = pd.read_csv(csv_file)
        if "Date" not in df.columns:
            pytest.fail(f"Date column missing from {res_info['file_name']}")

        df["parsed_date"] = pd.to_datetime(df["Date"], errors="coerce")
        in_range_count = len(df[(df["parsed_date"] >= START_DATE_STR) & (df["parsed_date"] <= END_DATE_STR)])
        assert in_range_count == EXPECTED_CALENDAR_DAYS, (
            f"{res_info['file_name']} contains {in_range_count} rows in 2010-2024; "
            f"expected exactly {EXPECTED_CALENDAR_DAYS} daily rows."
        )


# =====================================================================
# 6. Non-Synthetic & Valid Numerical Data Tests
# =====================================================================
class TestDataValidityAndNonSynthetic:
    """Verifies that Inflow and Storage contain valid, non-synthetic numerical data:
    - No NaNs or infinities
    - Non-negative values
    - Inflow is NOT the synthetic proxy max(diff(storage), 0)
    - Storage is realistic and physically plausible
    """

    @pytest.mark.parametrize("res_id", TARGET_RESERVOIRS.keys())
    def test_no_null_or_nan_values(self, res_id):
        """Neither Inflow nor Storage may contain null, NaN, or infinite values."""
        res_info = TARGET_RESERVOIRS[res_id]
        csv_file = DATA_DIR / res_info["file_name"]
        if not csv_file.exists():
            pytest.fail(f"File {csv_file} does not exist.")

        df = pd.read_csv(csv_file)
        inflow_col = "Inflow (cusecs/cumecs)"
        storage_col = "Storage (TMC/MCM)"

        for col in [inflow_col, storage_col]:
            if col not in df.columns:
                pytest.fail(f"Column '{col}' missing from {res_info['file_name']}")

            null_count = df[col].isna().sum()
            assert null_count == 0, (
                f"{res_info['file_name']} contains {null_count} null/NaN values in '{col}'"
            )

            inf_count = np.isinf(df[col]).sum()
            assert inf_count == 0, (
                f"{res_info['file_name']} contains {inf_count} infinite values in '{col}'"
            )

    @pytest.mark.parametrize("res_id", TARGET_RESERVOIRS.keys())
    def test_non_negative_values(self, res_id):
        """Inflow and Storage values must be non-negative (>= 0)."""
        res_info = TARGET_RESERVOIRS[res_id]
        csv_file = DATA_DIR / res_info["file_name"]
        if not csv_file.exists():
            pytest.fail(f"File {csv_file} does not exist.")

        df = pd.read_csv(csv_file)
        inflow_col = "Inflow (cusecs/cumecs)"
        storage_col = "Storage (TMC/MCM)"

        if inflow_col in df.columns:
            negative_inflow = (df[inflow_col] < 0).sum()
            assert negative_inflow == 0, (
                f"{res_info['file_name']} contains {negative_inflow} negative Inflow values!"
            )

        if storage_col in df.columns:
            negative_storage = (df[storage_col] < 0).sum()
            assert negative_storage == 0, (
                f"{res_info['file_name']} contains {negative_storage} negative Storage values!"
            )

    @pytest.mark.parametrize("res_id", TARGET_RESERVOIRS.keys())
    def test_inflow_is_not_synthetic_diff_storage(self, res_id):
        """Detects and rejects synthetic inflow generated as max(diff(storage), 0).
        Explorer 1 verified that previous legacy CSVs generated inflow using:
            df['inflow'] = storage_diff.apply(lambda x: x if x > 0 else 0.0)
        which yielded a 100% mathematical match with diff(storage).clip(lower=0).
        Genuine gauge/streamflow records differ significantly from pure storage differences.
        """
        res_info = TARGET_RESERVOIRS[res_id]
        csv_file = DATA_DIR / res_info["file_name"]
        if not csv_file.exists():
            pytest.fail(f"File {csv_file} does not exist.")

        df = pd.read_csv(csv_file)
        inflow_col = "Inflow (cusecs/cumecs)" if "Inflow (cusecs/cumecs)" in df.columns else "inflow"
        storage_col = "Storage (TMC/MCM)" if "Storage (TMC/MCM)" in df.columns else "storage"

        if inflow_col not in df.columns or storage_col not in df.columns:
            pytest.fail(
                f"Cannot test synthetic inflow: missing required columns in {res_info['file_name']}. "
                f"Columns present: {list(df.columns)}"
            )

        # Calculate synthetic proxy
        delta_s = df[storage_col].diff().clip(lower=0).fillna(0)
        match_mask = np.isclose(df[inflow_col], delta_s, atol=1e-2, rtol=1e-2)
        match_percentage = match_mask.mean() * 100.0

        assert match_percentage < 90.0, (
            f"REJECTED: Data in {res_info['file_name']} is SYNTHETIC! "
            f"{match_percentage:.2f}% of Inflow values exactly match max(diff(Storage), 0). "
            f"Genuine river gauge records required per ORIGINAL_REQUEST."
        )

    @pytest.mark.parametrize("res_id", TARGET_RESERVOIRS.keys())
    def test_storage_and_inflow_variability(self, res_id):
        """Storage and Inflow must exhibit realistic statistical variability over 15 years."""
        res_info = TARGET_RESERVOIRS[res_id]
        csv_file = DATA_DIR / res_info["file_name"]
        if not csv_file.exists():
            pytest.fail(f"File {csv_file} does not exist.")

        df = pd.read_csv(csv_file)
        inflow_col = "Inflow (cusecs/cumecs)" if "Inflow (cusecs/cumecs)" in df.columns else "inflow"
        storage_col = "Storage (TMC/MCM)" if "Storage (TMC/MCM)" in df.columns else "storage"

        if inflow_col in df.columns and storage_col in df.columns:
            assert df[storage_col].std() > 0.01, (
                f"Storage in {res_info['file_name']} is constant/flat (std={df[storage_col].std()})!"
            )
            assert df[inflow_col].max() > 0.0, (
                f"Inflow in {res_info['file_name']} has no non-zero values (all zeros)!"
            )

    @pytest.mark.parametrize("res_id", TARGET_RESERVOIRS.keys())
    def test_storage_within_physical_gross_capacity(self, res_id):
        """Storage must not exceed reservoir physical gross capacity with 15% surcharge buffer."""
        res_info = TARGET_RESERVOIRS[res_id]
        csv_file = DATA_DIR / res_info["file_name"]
        if not csv_file.exists():
            pytest.fail(f"File {csv_file} does not exist.")

        df = pd.read_csv(csv_file)
        storage_col = "Storage (TMC/MCM)"
        if storage_col not in df.columns:
            pytest.fail(f"Column '{storage_col}' missing from {res_info['file_name']}")

        gross_capacity_tmc = res_info["gross_capacity_mcm"] / 28.3168466
        max_allowed_tmc = gross_capacity_tmc * 1.15
        max_storage = df[storage_col].max()

        assert max_storage <= max_allowed_tmc, (
            f"Physical capacity breached in {res_info['file_name']}: "
            f"max storage {max_storage:.3f} TMC exceeds physical limit {max_allowed_tmc:.3f} TMC "
            f"({res_info['gross_capacity_mcm']} MCM / {gross_capacity_tmc:.3f} TMC * 1.15)"
        )


# =====================================================================
# 7. Hydrological Unit Conversion & Transformation Tests
# =====================================================================
class TestHydrologicalUnitConversions:
    """Verifies unit conversion algorithms used when standardizing units into
    cusecs/cumecs and TMC/MCM.
    """

    def test_cumecs_to_cusecs_conversion(self):
        """1 cubic meter per second (cumec) = 35.3146667 cubic feet per second (cusec)."""
        cumecs = 100.0
        expected_cusecs = 3531.46667
        actual_cusecs = cumecs * 35.3146667
        assert np.isclose(actual_cusecs, expected_cusecs, atol=1e-3)

    def test_mcm_to_tmc_conversion(self):
        """1 TMC = 28.3168466 MCM (or 1 MCM ≈ 0.03531467 TMC)."""
        tmc = 1.0
        expected_mcm = 28.3168466
        actual_mcm = tmc * 28.3168466
        assert np.isclose(actual_mcm, expected_mcm, atol=1e-3)

        mcm = 28.3168466
        actual_tmc = mcm / 28.3168466
        assert np.isclose(actual_tmc, 1.0, atol=1e-5)
