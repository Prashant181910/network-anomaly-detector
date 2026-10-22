"""Synthetic network-flow generator.

Produces flow-level records that mimic features found in public IDS datasets
(NSL-KDD / CIC-IDS style): duration, protocol, bytes, packets, TCP flags,
connection-rate statistics and error rates. Four attack families are simulated:

  * dos      - SYN/UDP flood: huge packet counts, tiny payloads, high conn rate
  * portscan - many short connections to many ports, high reset/error rate
  * bruteforce - repeated short connections to one service (ssh/ftp/http-auth)
  * exfil    - long flow with very large outbound byte volume

Swap `generate_flows()` for a loader of a real dataset (e.g. NSL-KDD or
CIC-IDS2017 CSV) as long as the column names in FEATURES are present.
"""
import numpy as np
import pandas as pd

PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = ["http", "https", "dns", "ssh", "ftp", "smtp", "other"]

NUMERIC = [
    "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts",
    "syn_ratio", "rst_ratio", "conn_count_2s", "srv_count_2s",
    "distinct_dst_ports", "error_rate", "bytes_per_pkt", "byte_ratio",
]
CATEGORICAL = ["protocol", "service"]
FEATURES = NUMERIC + CATEGORICAL


def _normal(rng, n):
    proto = rng.choice(PROTOCOLS, n, p=[0.75, 0.22, 0.03])
    service = rng.choice(SERVICES, n, p=[0.30, 0.35, 0.15, 0.04, 0.03, 0.05, 0.08])
    duration = rng.lognormal(0.5, 1.2, n)
    src_pkts = rng.poisson(12, n) + 1
    dst_pkts = rng.poisson(15, n) + 1
    src_bytes = rng.lognormal(6.5, 1.1, n)
    dst_bytes = rng.lognormal(7.8, 1.5, n)
    return pd.DataFrame({
        "duration": duration, "src_bytes": src_bytes, "dst_bytes": dst_bytes,
        "src_pkts": src_pkts, "dst_pkts": dst_pkts,
        "syn_ratio": rng.beta(1, 12, n), "rst_ratio": rng.beta(1, 25, n),
        "conn_count_2s": rng.poisson(4, n), "srv_count_2s": rng.poisson(3, n),
        "distinct_dst_ports": rng.poisson(1.5, n) + 1,
        "error_rate": rng.beta(1, 30, n),
        "protocol": proto, "service": service, "label": "normal",
    })


def _dos(rng, n):
    return pd.DataFrame({
        "duration": rng.exponential(0.3, n), "src_bytes": rng.lognormal(4.5, 0.5, n),
        "dst_bytes": rng.lognormal(2.0, 1.0, n),
        "src_pkts": rng.poisson(900, n) + 50, "dst_pkts": rng.poisson(2, n),
        "syn_ratio": rng.beta(30, 2, n), "rst_ratio": rng.beta(2, 10, n),
        "conn_count_2s": rng.poisson(250, n) + 50, "srv_count_2s": rng.poisson(240, n) + 50,
        "distinct_dst_ports": rng.poisson(1, n) + 1, "error_rate": rng.beta(8, 3, n),
        "protocol": rng.choice(["tcp", "udp", "icmp"], n, p=[0.6, 0.25, 0.15]),
        "service": rng.choice(["http", "https", "dns", "other"], n), "label": "dos",
    })


def _portscan(rng, n):
    return pd.DataFrame({
        "duration": rng.exponential(0.05, n), "src_bytes": rng.lognormal(3.5, 0.4, n),
        "dst_bytes": rng.lognormal(2.5, 0.8, n),
        "src_pkts": rng.poisson(2, n) + 1, "dst_pkts": rng.poisson(1, n),
        "syn_ratio": rng.beta(15, 3, n), "rst_ratio": rng.beta(10, 3, n),
        "conn_count_2s": rng.poisson(60, n) + 20, "srv_count_2s": rng.poisson(3, n),
        "distinct_dst_ports": rng.poisson(120, n) + 30, "error_rate": rng.beta(10, 2, n),
        "protocol": rng.choice(["tcp", "udp"], n, p=[0.9, 0.1]), "service": "other",
        "label": "portscan",
    })


def _bruteforce(rng, n):
    return pd.DataFrame({
        "duration": rng.gamma(2, 0.6, n), "src_bytes": rng.lognormal(5.2, 0.4, n),
        "dst_bytes": rng.lognormal(5.0, 0.4, n),
        "src_pkts": rng.poisson(10, n) + 4, "dst_pkts": rng.poisson(9, n) + 3,
        "syn_ratio": rng.beta(2, 8, n), "rst_ratio": rng.beta(3, 8, n),
        "conn_count_2s": rng.poisson(35, n) + 10, "srv_count_2s": rng.poisson(34, n) + 10,
        "distinct_dst_ports": np.ones(n), "error_rate": rng.beta(6, 4, n),
        "protocol": "tcp", "service": rng.choice(["ssh", "ftp", "http"], n, p=[0.6, 0.25, 0.15]),
        "label": "bruteforce",
    })


def _exfil(rng, n):
    return pd.DataFrame({
        "duration": rng.lognormal(4.0, 0.7, n), "src_bytes": rng.lognormal(15.0, 0.8, n),
        "dst_bytes": rng.lognormal(5.0, 0.8, n),
        "src_pkts": rng.poisson(8000, n) + 500, "dst_pkts": rng.poisson(300, n) + 20,
        "syn_ratio": rng.beta(1, 30, n), "rst_ratio": rng.beta(1, 40, n),
        "conn_count_2s": rng.poisson(2, n), "srv_count_2s": rng.poisson(2, n),
        "distinct_dst_ports": np.ones(n), "error_rate": rng.beta(1, 40, n),
        "protocol": "tcp", "service": rng.choice(["https", "dns", "other"], n, p=[0.5, 0.3, 0.2]),
        "label": "exfil",
    })


def generate_flows(n_normal=20000, anomaly_fraction=0.12, seed=42):
    rng = np.random.default_rng(seed)
    n_anom = int(n_normal * anomaly_fraction / (1 - anomaly_fraction))
    split = rng.dirichlet([3, 2.5, 2, 1.5]) * n_anom
    n_dos, n_scan, n_bf, n_ex = [max(1, int(x)) for x in split]
    df = pd.concat([
        _normal(rng, n_normal), _dos(rng, n_dos), _portscan(rng, n_scan),
        _bruteforce(rng, n_bf), _exfil(rng, n_ex),
    ], ignore_index=True)
    # Make detection non-trivial: ~40% of attacks are "stealthy" (low-and-slow),
    # with their numeric features blended toward normal behaviour.
    base = [c for c in NUMERIC if c not in ("bytes_per_pkt", "byte_ratio")]
    df[base] = df[base].astype(float)
    atk = np.where(df["label"] != "normal")[0]
    stealth = rng.choice(atk, int(0.40 * len(atk)), replace=False)
    nrm = np.where(df["label"] == "normal")[0]
    donors = rng.choice(nrm, len(stealth), replace=True)
    alpha = rng.uniform(0.15, 0.65, (len(stealth), 1))   # share of attack signal kept
    mixed = alpha * df.loc[stealth, base].to_numpy() + (1 - alpha) * df.loc[donors, base].to_numpy()
    df.loc[stealth, base] = mixed
    df["bytes_per_pkt"] = (df["src_bytes"] + df["dst_bytes"]) / (df["src_pkts"] + df["dst_pkts"] + 1)
    df["byte_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)
    df["is_anomaly"] = (df["label"] != "normal").astype(int)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    flows = generate_flows()
    flows.to_csv("traffic.csv", index=False)
    print(flows["label"].value_counts())
    print("saved traffic.csv", flows.shape)
