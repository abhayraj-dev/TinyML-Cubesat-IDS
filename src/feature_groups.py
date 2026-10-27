"""Provenance groups for feature-group ablation and XAI aggregation.

The groups are disjoint and exhaustive for each dataset.  CuCD-ID separates
the circular time index, CCSDS identifiers, 20-second sliding-window
aggregates, and raw telemetry.  STIN has no time/identifier/window
provenance fields; its groups describe network-flow feature families.
"""
from __future__ import annotations


CUCD_ID_GROUPS = {
    "time": ["TimeRadians"],
    "ids": ["MsgId", "CmdCode", "ApId"],
    "window": [
        "FlowLengthInWindow",
        "SlidingWindowMeanIntervalSec",
        "SlidingWindowMaxIntervalSec",
        "SlidingWindowMinIntervalSec",
        "MessageCountInWindow",
        "UniqueMessageIDsInWindow",
        "AverageMessageLengthInWindow",
        "MessageRateInWindow",
        "StdDevMessageLengthInWindow",
    ],
    "raw_telemetry": [
        "MsgLength",
        "MemoryAnonMB",
        "MemoryFileMB",
        "MemoryKernelstackMB",
        "MemoryPageTableMB",
        "MemorySocketMB",
        "MemoryPercpuMB",
        "MemoryShmemMB",
        "MemorySlabUnreclaimableMB",
    ],
}

STIN_GROUPS = {
    "flow_volume": [
        "fl_dur", "fw_pk", "l_fw_pkt", "l_bw_pkt", "fl_byt_s", "bw_pkt_s",
    ],
    "packet_length": [
        "pkt_len_min", "pkt_len_max", "pkt_len_std", "fw_hdr_len",
    ],
    "iat_timing": [
        "fl_iat_min", "bw_iat_tot", "bw_iat_min",
    ],
    "tcp_flags": [
        "bw_psh_flag", "fw_urg_flag", "bw_urg_flag", "fin_cnt",
        "syn_cnt", "psh_cnt", "urg_cnt", "ece_cnt",
    ],
    "tcp_window_block": [
        "down_up_ratio", "fw_byt_blk_avg", "fw_pkt_blk_avg",
        "fw_blk_rate_avg", "bw_byt_blk_avg", "bw_pkt_blk_avg",
        "bw_blk_rate_avg", "fw_win_byt", "bw_win_byt",
    ],
}


def get_feature_groups(dataset: str, feature_names) -> dict[str, list[str]]:
    """Return ordered, disjoint, exhaustive groups for the deployed feature list."""
    feature_names = list(feature_names)
    if dataset == "CuCD-ID":
        groups = CUCD_ID_GROUPS
    elif dataset == "STIN":
        groups = STIN_GROUPS
    else:
        raise ValueError(f"Unknown dataset for feature groups: {dataset}")
    group_features = [f for feats in groups.values() for f in feats]
    if len(group_features) != len(set(group_features)):
        raise ValueError(f"{dataset}: duplicate features across groups")
    missing = [f for f in feature_names if f not in set(group_features)]
    extra = [f for f in group_features if f not in set(feature_names)]
    if missing or extra:
        raise ValueError(
            f"{dataset}: feature groups do not partition feature_names; "
            f"missing={missing}, extra={extra}"
        )
    return {name: [f for f in feature_names if f in set(feats)]
            for name, feats in groups.items()}


def feature_group_indices(feature_names, groups: dict[str, list[str]]) -> dict[str, list[int]]:
    """Map group names to column indices in the deployed feature order."""
    pos = {f: i for i, f in enumerate(feature_names)}
    return {name: [pos[f] for f in feats] for name, feats in groups.items()}
