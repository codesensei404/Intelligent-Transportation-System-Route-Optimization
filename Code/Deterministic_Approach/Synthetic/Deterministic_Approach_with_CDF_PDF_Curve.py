import numpy as np
import heapq
import json
import sys
import matplotlib.pyplot as plt

class Edge:
    def __init__(self, to_node, pmf):
        self.to_node = to_node
        self.pmf = np.array(pmf)
        # Expected cost: sum(time * probability)
        times = np.arange(len(self.pmf))
        self.expected_time = np.sum(times * self.pmf)

class Graph:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.adj_list = {i: [] for i in range(num_nodes)}
        
    def add_edge(self, from_node, to_node, pmf):
        self.adj_list[from_node].append(Edge(to_node, pmf))

def load_graph_from_json(filepath):
    """Loads graph topology and edge PMFs from a JSON dataset."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        g = Graph(data['num_nodes'])
        for edge_data in data['edges']:
            g.add_edge(edge_data['from'], edge_data['to'], edge_data['pmf'])
        return g
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)

def dijkstra_let(graph, source, destination):
    """Computes the Least Expected Travel Time (LET) path for a single segment."""
    pq = [(0.0, source)]
    distances = {i: float('inf') for i in range(graph.num_nodes)}
    distances[source] = 0.0
    predecessors = {i: None for i in range(graph.num_nodes)}
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        
        if current_dist > distances[u]:
            continue
            
        if u == destination:
            break
            
        for edge in graph.adj_list[u]:
            v = edge.to_node
            new_dist = current_dist + edge.expected_time
            
            if new_dist < distances[v]:
                distances[v] = new_dist
                predecessors[v] = (u, edge)
                heapq.heappush(pq, (new_dist, v))
                
    if distances[destination] == float('inf'):
        return None, None, float('inf') # No path found
        
    path_nodes, path_edges = [], []
    curr = destination
    
    while curr is not None:
        path_nodes.append(curr)
        if predecessors[curr] is not None:
            pred_node, pred_edge = predecessors[curr]
            path_edges.append(pred_edge)
            curr = pred_node
        else:
            curr = None
            
    path_nodes.reverse()
    path_edges.reverse()
    return path_nodes, path_edges, distances[destination]

def compute_path_reliability(path_edges, time_budget):
    """Convolves PMFs along the combined deterministic path."""
    if not path_edges:
        return np.array([1.0]), (1.0 if time_budget >= 0 else 0.0)
        
    total_pmf = path_edges[0].pmf
    for edge in path_edges[1:]:
        total_pmf = np.convolve(total_pmf, edge.pmf)
        
    max_idx = min(time_budget, len(total_pmf) - 1)
    reliability = np.sum(total_pmf[:max_idx + 1])
    return total_pmf, reliability

def route_with_waypoints(graph, route_sequence):
    """Runs Dijkstra sequentially through all waypoints."""
    full_path_nodes = []
    full_path_edges = []
    total_expected_cost = 0.0
    
    for i in range(len(route_sequence) - 1):
        start = route_sequence[i]
        end = route_sequence[i+1]
        
        nodes, edges, cost = dijkstra_let(graph, start, end)
        
        if nodes is None:
            print(f"Error: No path exists between node {start} and node {end}.")
            sys.exit(1)
  
        if i == 0:
            full_path_nodes.extend(nodes)
        else:
            full_path_nodes.extend(nodes[1:])
            
        full_path_edges.extend(edges)
        total_expected_cost += cost
        
    return full_path_nodes, full_path_edges, total_expected_cost

def plot_pdf_cdf(pmf):
    """
    Plots PDF and CDF of the final arrival time distribution.
    """
    times = np.arange(len(pmf))
    cdf = np.cumsum(pmf)

    # ---- PDF Plot ----
    plt.figure()
    plt.stem(times, pmf)
    plt.xlabel("Arrival Time")
    plt.ylabel("Probability")
    plt.title("PDF of Total Arrival Time")
    plt.grid(True)
    plt.show()

    # ---- CDF Plot ----
    plt.figure()
    plt.step(times, cdf, where='post')
    plt.xlabel("Arrival Time")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF of Total Arrival Time")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("=== SOTA Deterministic Route Planner ===")

    json_file = input("Enter path to graph JSON file (e.g., graph_data.json): ").strip()
    g = load_graph_from_json(json_file)
    print(f"Graph loaded successfully! ({g.num_nodes} nodes)")

    try:
        source_node = int(input(f"Enter starting Point A (0 to {g.num_nodes-1}): "))
        dest_node = int(input(f"Enter destination Point B (0 to {g.num_nodes-1}): "))
        budget_T = int(input("Enter maximum Time Budget (T): "))
        
        num_stops = int(input("How many intermediate stops (waypoints)? "))
        waypoints = []
        for i in range(num_stops):
            stop = int(input(f"  Enter node ID for Stop {i+1}: "))
            waypoints.append(stop)
            
    except ValueError:
        print("Invalid input. Please enter integers only.")
        sys.exit(1)


    route_sequence = [source_node] + waypoints + [dest_node]
    print(f"\n--- Calculating LET Route for Sequence: {' -> '.join(map(str, route_sequence))} ---")


    let_nodes, let_edges, let_expected_cost = route_with_waypoints(g, route_sequence)
    

    let_pmf, let_reliability = compute_path_reliability(let_edges, budget_T)


    print("\n[ RESULTS ]")
    print(f"Full LET Path Taken: {' -> '.join(map(str, let_nodes))}")
    print(f"Total Expected Cost: {let_expected_cost:.2f}")
    
    print("\n[ STOCHASTIC EVALUATION ]")
    print(f"Probability of arriving within budget (T <= {budget_T}): {let_reliability * 100:.2f}%")
    
 
    print("Predicted Arrival Time Probabilities (top 5 possible times):")
    non_zero_times = np.nonzero(let_pmf)[0]
    sorted_times = sorted(non_zero_times, key=lambda t: let_pmf[t], reverse=True)[:5]
    for t in sorted_times:
        print(f"  - Arriving at t={t}: {let_pmf[t]*100:.2f}%")
    
    plot_pdf_cdf(let_pmf)