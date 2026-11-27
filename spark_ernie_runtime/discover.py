"""Automatic 200GbE QSFP / RoCEv2 ConnectX-7 cable & ring topology discovery for 2x-4x DGX Spark.

Probes each Spark host over SSH (or locally for rank 0) using standard Linux
network/RDMA tools (`ibdev2netdev`, `ip -j addr`, `ip -j link`, `ethtool` / sysfs),
detects active 200GbE CX7 links, matches point-to-point peer subnets across nodes,
solves the physical ring order for 4-node switchless rings, and writes a clean
machine-local `~/.config/spark-ernie/cluster.json`.
"""
from __future__ import annotations

import ipaddress
import json
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import CLUSTER_FILE, save_cluster

PROBE_REMOTE_PY = r"""
import json, os, socket, subprocess, glob

def run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""

# 1. Map RDMA HCA -> netdev via sysfs (works even without rdma-core ibdev2netdev)
netdev_to_hca = {}
hca_to_netdev = {}
for p in sorted(glob.glob("/sys/class/infiniband/*/device/net/*")):
    parts = p.split("/")
    hca, nd = parts[4], parts[7]
    netdev_to_hca[nd] = hca
    hca_to_netdev[hca] = nd

# Also parse ibdev2netdev if available (more accurate for multi-port CX7)
ib2nd = run(["ibdev2netdev"])
for line in ib2nd.splitlines():
    toks = line.split()
    if len(toks) >= 5 and toks[3] == "==>":
        hca, nd = toks[0], toks[4]
        netdev_to_hca[nd] = hca
        hca_to_netdev[hca] = nd

# 2. Collect IPv4 addresses, link state, MTU, and speed
ip_json = run(["ip", "-j", "-4", "addr", "show"])
ifaces = []
try:
    addr_data = json.loads(ip_json) if ip_json else []
except Exception:
    addr_data = []

for item in addr_data:
    name = item.get("ifname", "")
    if name == "lo":
        continue
    oper = item.get("operstate", "UNKNOWN")
    mtu = item.get("mtu", 1500)
    speed = -1
    try:
        with open(f"/sys/class/net/{name}/speed") as f:
            speed = int(f.read().strip())
    except Exception:
        pass
    hca = netdev_to_hca.get(name, "")
    addrs = []
    for a in item.get("addr_info", []):
        if a.get("family") == "inet":
            addrs.append(f"{a['local']}/{a['prefixlen']}")
    ifaces.append({
        "ifname": name,
        "operstate": oper,
        "mtu": mtu,
        "speed_mbps": speed,
        "hca": hca,
        "addrs": addrs,
    })

# 3. Check optional patched switchless 4-HCA NCCL library
nccl_candidates = [
    os.environ.get("SPARK_NCCL_LIB", ""),
    "/opt/nccl-switchless-4hca/libnccl.so.2.30.7",
    "/opt/nccl-switchless-4hca/libnccl.so.2",
]
nccl_lib = next((c for c in nccl_candidates if c and os.path.exists(c)), "")

# 4. Check GPU & unified RAM
mem_kb = 0
try:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                mem_kb = int(line.split()[1])
                break
except Exception:
    pass

gpu_name = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])

print(json.dumps({
    "hostname": socket.gethostname(),
    "mem_gib": round(mem_kb / 1048576, 1),
    "gpu": gpu_name.splitlines()[0] if gpu_name else "unknown",
    "nccl_lib": nccl_lib,
    "ifaces": ifaces,
}))
"""


@dataclass
class LinkEdge:
    src_idx: int
    dst_idx: int
    src_hca: str
    dst_hca: str
    src_if: str
    dst_if: str
    src_ip: str
    dst_ip: str
    subnet: str
    speed_mbps: int
    mtu: int


def _is_local_host(host: str) -> bool:
    if host in ("localhost", "127.0.0.1", "::1"):
        return True
    hn = socket.gethostname()
    return host in (hn, f"{hn}.local")


def probe_host(host: str, ssh_user: str | None = None, timeout: int = 10) -> dict[str, Any]:
    if _is_local_host(host):
        out = subprocess.check_output(["python3", "-c", PROBE_REMOTE_PY], text=True, timeout=timeout)
    else:
        target = f"{ssh_user}@{host}" if ssh_user and "@" not in host else host
        out = subprocess.check_output(
            ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}", target, "python3", "-"],
            input=PROBE_REMOTE_PY,
            text=True,
            timeout=timeout + 5,
        )
    data = json.loads(out)
    data["ssh_host"] = host
    return data


def _find_edges(probes: list[dict[str, Any]]) -> list[LinkEdge]:
    """Match active RoCEv2 HCA interfaces across hosts that share an IPv4 subnet."""
    subnet_map: dict[str, list[tuple[int, dict[str, Any], str]]] = {}
    for idx, p in enumerate(probes):
        for iface in p.get("ifaces", []):
            hca = iface.get("hca")
            if not hca:
                continue
            if iface.get("operstate") not in ("UP", "UNKNOWN"):
                continue
            for cidr in iface.get("addrs", []):
                net = ipaddress.IPv4Interface(cidr).network
                if net.is_loopback or net.is_link_local:
                    continue
                # Point-to-point CX7 rails are typically /24../30
                if net.prefixlen >= 24:
                    subnet_map.setdefault(str(net), []).append((idx, iface, cidr.split("/")[0]))

    edges: list[LinkEdge] = []
    for subnet, endpoints in sorted(subnet_map.items()):
        if len(endpoints) != 2:
            continue
        (i, if_i, ip_i), (j, if_j, ip_j) = endpoints
        if i == j:
            continue
        edges.append(
            LinkEdge(
                src_idx=i,
                dst_idx=j,
                src_hca=if_i["hca"],
                dst_hca=if_j["hca"],
                src_if=if_i["ifname"],
                dst_if=if_j["ifname"],
                src_ip=ip_i,
                dst_ip=ip_j,
                subnet=subnet,
                speed_mbps=min(if_i.get("speed_mbps", -1), if_j.get("speed_mbps", -1)),
                mtu=min(if_i.get("mtu", 1500), if_j.get("mtu", 1500)),
            )
        )
    return edges


def _solve_ring_order(n: int, adj: dict[int, set[int]]) -> list[int] | None:
    """Find a Hamiltonian cycle starting at node 0 for a 4-node switchless ring."""
    if n < 3:
        return list(range(n))
    path = [0]
    used = {0}

    def dfs(u: int) -> bool:
        if len(path) == n:
            return 0 in adj.get(u, set())
        for v in sorted(adj.get(u, set())):
            if v not in used:
                used.add(v)
                path.append(v)
                if dfs(v):
                    return True
                path.pop()
                used.remove(v)
        return False

    return path if dfs(0) else None


def _solve_dual_pairs(adj: dict[int, set[int]]) -> list[tuple[int, int]] | None:
    """Find two disjoint linked 2-node pairs among 4 hosts (e.g. 2x2 parallelized wiring)."""
    for v in sorted(adj.get(0, set())):
        rem = [x for x in range(4) if x not in (0, v)]
        a, b = rem[0], rem[1]
        if b in adj.get(a, set()):
            return [(0, v), (a, b)]
    return None


def _build_pair_subcluster(
    probes: list[dict[str, Any]],
    pair_edges: dict[tuple[int, int], list[LinkEdge]],
    orig_a: int,
    orig_b: int,
    ssh_user: str,
) -> dict[str, Any]:
    """Build a self-contained 2-node `direct` cluster dict for one TP=2 pair."""
    order = [orig_a, orig_b]
    nodes_out: list[dict[str, Any]] = []
    for prank, oidx in enumerate(order):
        peer_oidx = order[1 - prank]
        p = probes[oidx]
        el = sorted(pair_edges[(oidx, peer_oidx)], key=lambda x: x.subnet)
        hlist = [e.src_hca for e in el]
        subnets = {e.src_hca: e.subnet for e in el}
        links = [
            {
                "peer_rank": 1 - prank,
                "local_hca": e.src_hca,
                "local_if": e.src_if,
                "local_ip": e.src_ip,
                "peer_hca": e.dst_hca,
                "peer_ip": e.dst_ip,
                "subnet": e.subnet,
                "speed_mbps": e.speed_mbps,
                "mtu": e.mtu,
            }
            for e in el
        ]
        first = links[0]
        nodes_out.append({
            "rank": prank,
            "ssh_host": p["ssh_host"],
            "hostname": p["hostname"],
            "gpu": p["gpu"],
            "mem_gib": p["mem_gib"],
            "host_ip": first["local_ip"],
            "oob_iface": first["local_if"],
            "nccl_ib_hca": "=" + ",".join(hlist),
            "sparknet_peer_hcas": {str(1 - prank): hlist},
            "sparknet_subnets": subnets,
            "links": links,
        })
    return {
        "schema_version": 1,
        "topology": "direct",
        "world_size": 2,
        "ssh_user": ssh_user,
        "master_addr": nodes_out[0]["host_ip"],
        "nccl_lib_path": "",
        "nodes": nodes_out,
    }


def _pick_oob_iface(probe: dict[str, Any], rdma_ifs: set[str], prefer_ring0: bool) -> tuple[str, str]:
    """Pick the OOB/control-plane interface (ring0 > non-RDMA LAN > first RDMA rail)."""
    ifaces = probe.get("ifaces", [])
    if prefer_ring0:
        for iface in ifaces:
            if iface["ifname"] == "ring0" and iface.get("addrs"):
                return "ring0", iface["addrs"][0].split("/")[0]
    # Next: first UP non-RDMA interface with a private/routable IPv4 (e.g., eth/enp/tailscale/br)
    for iface in ifaces:
        if iface["ifname"] in rdma_ifs or iface["ifname"].startswith(("docker", "veth", "virbr")):
            continue
        if iface.get("operstate") == "UP" and iface.get("addrs"):
            return iface["ifname"], iface["addrs"][0].split("/")[0]
    # Fallback: first RDMA rail (ideal for 2-node direct cable!)
    for iface in ifaces:
        if iface["ifname"] in rdma_ifs and iface.get("addrs"):
            return iface["ifname"], iface["addrs"][0].split("/")[0]
    for iface in ifaces:
        if iface.get("addrs"):
            return iface["ifname"], iface["addrs"][0].split("/")[0]
    return "", ""


def discover_cluster(
    hosts: list[str],
    ssh_user: str | None = None,
    save: bool = True,
    out_path: Path = CLUSTER_FILE,
) -> dict[str, Any]:
    """Discover 2x direct or 4x switchless ring 200GbE CX7 topology across hosts."""
    if len(hosts) not in (2, 4):
        raise ValueError(f"Expected 2 or 4 DGX Spark hosts, got {len(hosts)}")

    probes = [probe_host(h, ssh_user=ssh_user) for h in hosts]
    edges = _find_edges(probes)

    adj: dict[int, set[int]] = {i: set() for i in range(len(hosts))}
    pair_edges: dict[tuple[int, int], list[LinkEdge]] = {}
    for e in edges:
        adj[e.src_idx].add(e.dst_idx)
        adj[e.dst_idx].add(e.src_idx)
        pair_edges.setdefault((e.src_idx, e.dst_idx), []).append(e)
        pair_edges.setdefault((e.dst_idx, e.src_idx), []).append(
            LinkEdge(
                src_idx=e.dst_idx,
                dst_idx=e.src_idx,
                src_hca=e.dst_hca,
                dst_hca=e.src_hca,
                src_if=e.dst_if,
                dst_if=e.src_if,
                src_ip=e.dst_ip,
                dst_ip=e.src_ip,
                subnet=e.subnet,
                speed_mbps=e.speed_mbps,
                mtu=e.mtu,
            )
        )

    pairs_out: list[dict[str, Any]] = []
    if len(hosts) == 2:
        topology = "direct"
        order = [0, 1]
        if not pair_edges.get((0, 1)):
            raise RuntimeError(
                "No direct 200GbE RoCEv2 link detected between the 2 hosts. "
                "Verify QSFP cables are plugged in and both ends share a /24 or /30 IPv4 subnet."
            )
    else:
        ring_order = _solve_ring_order(4, adj)
        if ring_order is not None:
            topology = "ring4"
            order = ring_order
            # Also extract 2 independent direct TP=2 pairs (0<->1 and 2<->3, picking max-rail pairing!)
            cand_a = [(ring_order[0], ring_order[1]), (ring_order[2], ring_order[3])]
            cand_b = [(ring_order[0], ring_order[3]), (ring_order[1], ring_order[2])]
            rails_a = sum(len(pair_edges.get(p, ())) for p in cand_a)
            rails_b = sum(len(pair_edges.get(p, ())) for p in cand_b)
            chosen_pairs = cand_a if rails_a >= rails_b else cand_b
            pairs_out = [
                _build_pair_subcluster(probes, pair_edges, a, b, ssh_user or "")
                for a, b in chosen_pairs
            ]
        else:
            dual_pairs = _solve_dual_pairs(adj)
            if dual_pairs is None:
                raise RuntimeError(
                    f"Could not form a 4-node ring or 2x2 parallel pairs from discovered links: {adj}. "
                    "Verify QSFP cables connect either a 4-node ring or two 2-node direct pairs."
                )
            topology = "dual_direct"
            order = [dual_pairs[0][0], dual_pairs[0][1], dual_pairs[1][0], dual_pairs[1][1]]
            pairs_out = [
                _build_pair_subcluster(probes, pair_edges, a, b, ssh_user or "")
                for a, b in dual_pairs
            ]

    idx_to_rank = {orig_idx: rank for rank, orig_idx in enumerate(order)}
    nodes_out: list[dict[str, Any]] = []
    nccl_lib = ""

    for rank, orig_idx in enumerate(order):
        p = probes[orig_idx]
        if not nccl_lib and p.get("nccl_lib"):
            nccl_lib = p["nccl_lib"]
        peer_hcas: dict[str, list[str]] = {}
        subnets: dict[str, str] = {}
        hcas_ordered: list[str] = []
        rdma_ifs: set[str] = set()
        peer_links: list[dict[str, Any]] = []

        # Order peers canonically: clockwise neighbor (rank+1)%N first, then counter-clockwise (rank-1)%N
        n_world = len(order)
        peer_order = sorted(
            adj[orig_idx],
            key=lambda x: ((idx_to_rank[x] - rank) % n_world),
        )
        for peer_orig in peer_order:
            prank = idx_to_rank[peer_orig]
            el = sorted(pair_edges[(orig_idx, peer_orig)], key=lambda x: x.subnet)
            hlist = []
            for e in el:
                hlist.append(e.src_hca)
                subnets[e.src_hca] = e.subnet
                rdma_ifs.add(e.src_if)
                if e.src_hca not in hcas_ordered:
                    hcas_ordered.append(e.src_hca)
                peer_links.append({
                    "peer_rank": prank,
                    "local_hca": e.src_hca,
                    "local_if": e.src_if,
                    "local_ip": e.src_ip,
                    "peer_hca": e.dst_hca,
                    "peer_ip": e.dst_ip,
                    "subnet": e.subnet,
                    "speed_mbps": e.speed_mbps,
                    "mtu": e.mtu,
                })
            peer_hcas[str(prank)] = hlist

        # For 2-node direct cable, using the first RDMA rail for OOB avoids LAN hops;
        # for 4-node ring, prefer ring0 (or LAN) so all 4 ranks reach MASTER_ADDR directly.
        if topology == "direct":
            first_link = peer_links[0]
            oob_if, host_ip = first_link["local_if"], first_link["local_ip"]
        else:
            oob_if, host_ip = _pick_oob_iface(p, rdma_ifs, prefer_ring0=True)

        nodes_out.append({
            "rank": rank,
            "ssh_host": p["ssh_host"],
            "hostname": p["hostname"],
            "gpu": p["gpu"],
            "mem_gib": p["mem_gib"],
            "host_ip": host_ip,
            "oob_iface": oob_if,
            "nccl_ib_hca": "=" + ",".join(hcas_ordered),
            "sparknet_peer_hcas": peer_hcas,
            "sparknet_subnets": subnets,
            "links": peer_links,
        })

    cluster: dict[str, Any] = {
        "schema_version": 1,
        "topology": topology,
        "world_size": len(nodes_out),
        "ssh_user": ssh_user or "",
        "master_addr": nodes_out[0]["host_ip"],
        "nccl_lib_path": nccl_lib,
        "nodes": nodes_out,
    }
    if pairs_out:
        cluster["pairs"] = pairs_out
    if save:
        save_cluster(cluster, out_path)
    return cluster


def format_topology_summary(cluster: dict[str, Any]) -> str:
    lines = [
        f"Discovered {cluster['world_size']}x DGX Spark cluster "
        f"(topology={cluster['topology']}, master_addr={cluster['master_addr']})",
    ]
    if cluster.get("nccl_lib_path"):
        lines.append(f"  Switchless 4-HCA NCCL: {cluster['nccl_lib_path']}")
    for n in cluster["nodes"]:
        rails = ", ".join(
            f"-> rank {l['peer_rank']} [{l['local_hca']}({l['local_ip']}) <-> {l['peer_hca']}({l['peer_ip']}), "
            f"{l['speed_mbps']//1000 if l['speed_mbps'] > 0 else '?'}GbE MTU {l['mtu']}]"
            for l in n["links"]
        )
        lines.append(
            f"  Rank {n['rank']}: {n['ssh_host']} ({n['hostname']}, {n['gpu']}, {n['mem_gib']} GiB) "
            f"OOB={n['oob_iface']}({n['host_ip']}) | {rails}"
        )
    if cluster.get("pairs"):
        p_desc = ", ".join(
            f"pair{i}=({p['nodes'][0]['ssh_host']}<->{p['nodes'][1]['ssh_host']}, {len(p['nodes'][0]['links'])}x200GbE)"
            for i, p in enumerate(cluster["pairs"])
        )
        lines.append(f"  2x2 Parallelized Pairs Ready: {p_desc}")
    return "\n".join(lines)
