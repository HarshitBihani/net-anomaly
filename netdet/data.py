"""NSL-KDD: network connection records labelled normal or one of many attack types.

Why this dataset: its official TEST file contains attack types that never appear in the TRAIN file. That is the
situation a real detector faces (tomorrow's attack is not in yesterday's training data), so it lets us compare a model that
learns known attacks with one that only learns what normal looks like.
Caveat: the traffic was recorded in the late 1990s. It is a standard teaching benchmark, not a picture of modern networks.
"""
from __future__ import annotations
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

COLUMNS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
           "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
           "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count",
           "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
           "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
           "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
           "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in COLUMNS if c not in CATEGORICAL]

FAMILY = {}
for fam, names in {
    "DoS": "back land neptune pod smurf teardrop apache2 udpstorm processtable worm mailbomb",
    "Probe": "satan ipsweep nmap portsweep mscan saint",
    "R2L": "guess_passwd ftp_write imap phf multihop warezmaster warezclient spy xlock xsnoop snmpguess snmpgetattack httptunnel sendmail named",
    "U2R": "buffer_overflow loadmodule rootkit perl sqlattack xterm ps",
}.items():
    for n in names.split():
        FAMILY[n] = fam
FAMILY["normal"] = "normal"


def _read(name: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA, name), header=None, names=COLUMNS + ["attack", "difficulty"])
    df["family"] = df["attack"].map(FAMILY).fillna("other")
    df["is_attack"] = (df["attack"] != "normal").astype(int)
    return df


def load():
    train, test = _read("KDDTrain+.txt"), _read("KDDTest+.txt")
    seen = set(train.attack.unique())
    test["unseen_attack"] = (~test.attack.isin(seen) & (test.is_attack == 1)).astype(int)
    train["unseen_attack"] = 0
    return train, test
