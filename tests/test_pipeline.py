"""Unit & integration tests verifying TRD specifications and pipeline integrity."""
import numpy as np
import pytest
import torch

from src.data.preprocessing import (
    FeaturePipeline,
    Scaler,
    build_splits,
    graph_features,
    interpolate_missing,
    resample_uniform,
    sequence_features,
)
from src.inference.controller import OSActionController
from src.inference.gate import GateConfig, PostProcessGate, SystemState
from src.models import build_model, count_parameters
from src.models.stgcn import build_hand_adjacency


def test_hand_adjacency():
    A = build_hand_adjacency()
    assert A.shape == (3, 21, 21)
    assert not np.isnan(A).any()


def test_feature_dimensions():
    seq = np.random.randn(32, 21, 3).astype(np.float32)
    s_feat = sequence_features(seq, use_global=True)
    assert s_feat.shape == (32, 131)

    s_feat_noglobal = sequence_features(seq, use_global=False)
    assert s_feat_noglobal.shape == (32, 126)

    g_feat = graph_features(seq, use_global=True)
    assert g_feat.shape == (10, 32, 21)

    g_feat_noglobal = graph_features(seq, use_global=False)
    assert g_feat_noglobal.shape == (6, 32, 21)


def test_models_forward():
    x_seq = torch.randn(2, 32, 131)
    bilstm = build_model("bilstm", num_classes=8, dropout=0.3)
    out_bilstm = bilstm(x_seq)
    assert out_bilstm.shape == (2, 8)

    tcn = build_model("tcn", num_classes=8, dropout=0.2)
    out_tcn = tcn(x_seq)
    assert out_tcn.shape == (2, 8)

    x_graph = torch.randn(2, 10, 32, 21)
    stgcn = build_model("stgcn", num_classes=8, dropout=0.3)
    out_stgcn = stgcn(x_graph)
    assert out_stgcn.shape == (2, 8)


def test_postprocess_gate():
    cfg = GateConfig(confidence_threshold=0.80, consecutive_windows=3, cooldown_s=0.9)
    gate = PostProcessGate(cfg, background_id=7)

    # 1. Background gesture (class 7) -> No fire
    probs_bg = np.zeros(8, dtype=np.float32)
    probs_bg[7] = 1.0
    dec = gate.update(probs_bg, motion_ok=True, now=1.0)
    assert not dec.fire

    # 2. Gesture class 1 (Swipe Right) with high confidence 0.90
    probs_swipe = np.zeros(8, dtype=np.float32)
    probs_swipe[1] = 0.90

    # Window 1 -> consistency 1/3 -> No fire
    d1 = gate.update(probs_swipe, motion_ok=True, now=1.1)
    assert not d1.fire and d1.consistency == 1

    # Window 2 -> consistency 2/3 -> No fire
    d2 = gate.update(probs_swipe, motion_ok=True, now=1.2)
    assert not d2.fire and d2.consistency == 2

    # Window 3 -> consistency 3/3 -> FIRE!
    d3 = gate.update(probs_swipe, motion_ok=True, now=1.3)
    assert d3.fire and d3.class_id == 1

    # Cooldown check: Window 4 at now=1.5 (still in cooldown 0.9s) -> No fire
    d4 = gate.update(probs_swipe, motion_ok=True, now=1.5)
    assert not d4.fire
