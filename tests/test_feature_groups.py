"""Regression tests for feature-group definitions."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_groups import (CUCD_ID_GROUPS, STIN_GROUPS, get_feature_groups,
                            feature_group_indices)


def _assert_partition(groups, expected_n):
    flat = [f for feats in groups.values() for f in feats]
    assert len(flat) == expected_n, (len(flat), expected_n)
    assert len(flat) == len(set(flat)), "duplicate feature across groups"
    assert all(feats for feats in groups.values()), "empty group"


def main():
    _assert_partition(CUCD_ID_GROUPS, 22)
    _assert_partition(STIN_GROUPS, 30)
    assert set(CUCD_ID_GROUPS) == {"time", "ids", "window", "raw_telemetry"}
    assert set(STIN_GROUPS) == {
        "flow_volume", "packet_length", "iat_timing", "tcp_flags",
        "tcp_window_block"}

    cucd_features = [f for feats in CUCD_ID_GROUPS.values() for f in feats]
    stin_features = [f for feats in STIN_GROUPS.values() for f in feats]
    g = get_feature_groups("CuCD-ID", cucd_features)
    assert [f for feats in g.values() for f in feats] == cucd_features
    idx = feature_group_indices(cucd_features, g)
    assert sorted(i for ids in idx.values() for i in ids) == list(range(22))

    g2 = get_feature_groups("STIN", stin_features)
    assert sum(len(v) for v in g2.values()) == 30

    try:
        get_feature_groups("STIN", stin_features[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("missing feature must raise")

    try:
        get_feature_groups("unknown", [])
    except ValueError:
        pass
    else:
        raise AssertionError("unknown dataset must raise")

    print("test_feature_groups: OK")


if __name__ == "__main__":
    main()
