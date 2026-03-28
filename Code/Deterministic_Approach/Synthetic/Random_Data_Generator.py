import json
import math


def generate_graph():
    print("===== Deterministic SOTA Dataset Generator =====")

    # =========================
    # USER INPUT
    # =========================
    num_nodes = int(input("Enter number of nodes to generate: "))
    filename = input("Enter output JSON filename (e.g., graph.json): ")

    if num_nodes <= 1:
        print("Number of nodes must be > 1")
        return

    edges = []

    # =========================
    # Automatically determine grid shape
    # =========================
    width = int(math.sqrt(num_nodes))
    height = math.ceil(num_nodes / width)

    print(f"\nGrid structure: approx {width} x {height}")

    # =========================
    # Build structured deterministic graph
    # =========================
    for i in range(height):
        for j in range(width):

            node_id = i * width + j
            if node_id >= num_nodes:
                continue

            # ---- Right neighbor ----
            if j + 1 < width:
                right_id = i * width + (j + 1)
                if right_id < num_nodes:
                    edges.append({
                        "from": node_id,
                        "to": right_id,
                        "pmf": [0, 0.7, 0.3]   # travel time = 1 or 2
                    })

            # ---- Down neighbor ----
            if i + 1 < height:
                down_id = (i + 1) * width + j
                if down_id < num_nodes:
                    edges.append({
                        "from": node_id,
                        "to": down_id,
                        "pmf": [0, 0.6, 0.4]
                    })

            # ---- Diagonal neighbor (adds path diversity) ----
            if i + 1 < height and j + 1 < width:
                diag_id = (i + 1) * width + (j + 1)
                if diag_id < num_nodes:
                    edges.append({
                        "from": node_id,
                        "to": diag_id,
                        "pmf": [0, 0.5, 0.5]
                    })

    # =========================
    # Ensure graph connectivity (chain backbone)
    # =========================
    for i in range(num_nodes - 1):
        edges.append({
            "from": i,
            "to": i + 1,
            "pmf": [0, 0.8, 0.2]
        })

    graph_data = {
        "num_nodes": num_nodes,
        "edges": edges
    }

    with open(filename, "w") as f:
        json.dump(graph_data, f, indent=2)

    print("\n===== Dataset Generated Successfully =====")
    print(f"Nodes: {num_nodes}")
    print(f"Edges: {len(edges)}")
    print(f"Saved to: {filename}")


if __name__ == "__main__":
    generate_graph()