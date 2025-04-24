#!/usr/bin/env python3
import random
from pathlib import Path

def generate_weakscale_edgefile(global_num_nodes, global_num_edges, out_path):
    """
    Generates a single edge file for weak scaling.
    - global_num_nodes: total nodes (e.g., P * nodes_per_rank)
    - global_num_edges: total edges (e.g., P * edges_per_rank)
    - out_path: path to write the combined edges file
    """
    nodes = list(range(global_num_nodes))
    degree_list = nodes.copy()
    edges = set()

    while len(edges) < global_num_edges:
        src = random.choice(degree_list)
        dst = random.choice(degree_list)
        if src != dst:
            e = (src, dst)
            if e not in edges:
                edges.add(e)
                degree_list.extend([src, dst])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for src, dst in edges:
            weight = round(random.uniform(0.1, 10.0), 2)
            f.write(f"{src}\t{dst}\t{weight}\n")

    print(f"Wrote {len(edges)} edges → {out_path}")

def main():
    # Per-rank workload
    nodes_per_rank = 20000
    edges_per_rank = 250000

    # MPI sizes to generate
    mpi_sizes = [1, 2, 4, 6, 8, 12]

    for P in mpi_sizes:
        total_nodes = P * nodes_per_rank
        total_edges = P * edges_per_rank
        out_file = f"./edges_P{P}.txt"
        print(f"Generating combined file for P={P}: nodes={total_nodes}, edges={total_edges}")
        generate_weakscale_edgefile(total_nodes, total_edges, out_file)

if __name__ == "__main__":
    main()
