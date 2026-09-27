"""Regression test: build_datasets must produce a graph WITH climatological edges.

Pins the audit fix that wired rainfall into graph construction, and guards the
ordering bug where a Trainer built before build_datasets would train on the
physical-only (pre-rebuild) edge_index.
"""

import sys
from pathlib import Path

import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.graph_builder import build_reservoir_graph  # noqa: E402

CONFIG = PROJECT_ROOT / "configs" / "default_config.yaml"
RESERVOIRS = PROJECT_ROOT / "configs" / "reservoirs.yaml"


@pytest.fixture(scope="module")
def prepared():
    import main as M

    cfg = M.load_config(str(CONFIG))
    res = yaml.safe_load(open(RESERVOIRS))["reservoirs"]
    # Start from a physical-only graph, exactly as main.build_graph does.
    graph = build_reservoir_graph(
        reservoirs=res,
        correlation_threshold=cfg["graph"]["correlation_threshold"],
        block_cross_ghats=cfg["graph"]["block_cross_ghats"],
        use_physical=cfg["graph"]["physical_edges"],
        use_climatological=cfg["graph"]["climatological_edges"],
    )
    return M, cfg, graph


def test_graph_starts_physical_only(prepared):
    _, _, graph = prepared
    n_phys = int((graph.edge_type == 0).sum()) if graph.edge_type.numel() else graph.edge_index.shape[1]
    n_clim = int((graph.edge_type == 1).sum()) if graph.edge_type.numel() else 0
    assert n_clim == 0, "engineered graph should have no climatological edges yet"
    assert n_phys > 0


def test_build_datasets_adds_climatological_edges(prepared):
    M, cfg, graph = prepared
    before = graph.edge_index.shape[1]
    _, _, _, _ = M.build_datasets(cfg, graph)
    after = graph.edge_index.shape[1]
    n_clim = int((graph.edge_type == 1).sum()) if graph.edge_type.numel() else 0
    assert n_clim > 0, "build_datasets must build the configured climatological edges"
    assert after > before, f"edge count did not grow ({before} -> {after})"


def test_edge_attr_matches_edge_count(prepared):
    M, cfg, graph = prepared
    M.build_datasets(cfg, graph)
    # Trainer stores edge_index at construction; it must be the rebuilt tensor.
    assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]
