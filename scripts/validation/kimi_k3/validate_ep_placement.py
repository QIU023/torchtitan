# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check contiguous EP groups against Slurm's physical NVLink domains."""
import argparse
import json
import re
import subprocess


def check_groups(nodes, domains, *, ranks_per_node, ep, pp):
    ranks = [node for node in nodes for _ in range(ranks_per_node)]
    if len(ranks) != ep * pp:
        raise ValueError(f"Expected {ep * pp} ranks, got {len(ranks)}")
    groups = []
    for stage in range(pp):
        hosts = ranks[stage * ep : (stage + 1) * ep]
        group_domains = {domains[host] for host in hosts}
        if len(group_domains) != 1:
            raise ValueError(
                f"EP group {stage} crosses NVLink domains: {[(host, domains[host]) for host in dict.fromkeys(hosts)]}"
            )
        groups.append(
            {
                "stage": stage,
                "domain": next(iter(group_domains)),
                "nodes": list(dict.fromkeys(hosts)),
                "ranks": ep,
            }
        )
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--ep", type=int, default=32)
    parser.add_argument("--ranks-per-node", type=int, default=4)
    args = parser.parse_args()
    nodes = subprocess.check_output(
        ["scontrol", "show", "hostnames", args.nodes], text=True
    ).split()
    rows = subprocess.check_output(
        ["scontrol", "show", "node", "-o", args.nodes], text=True
    ).splitlines()
    domains = {}
    for row in rows:
        node = re.search(r"\bNodeName=(\S+)", row).group(1)
        features = re.search(r"\bActiveFeatures=(\S+)", row).group(1).split(",")
        nvl = [f for f in features if f.startswith("nvlblk")]
        if len(nvl) != 1:
            raise ValueError(f"{node}: expected one NVLink-domain feature, got {nvl}")
        domains[node] = nvl[0]
    groups = check_groups(
        nodes, domains, ranks_per_node=args.ranks_per_node, ep=args.ep, pp=args.pp
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "policy": "each EP group lies inside one NVLink domain",
                "groups": groups,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
