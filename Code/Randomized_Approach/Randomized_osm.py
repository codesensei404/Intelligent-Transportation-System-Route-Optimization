import osmnx as ox
import networkx as nx
import folium
import numpy as np
import heapq
import math
from scipy.stats import norm

print("======================================================")
print(" SOTA (Stochastic On-Time Arrival) Route Optimizer")
print("======================================================")

# 1. GET DYNAMIC USER INPUT
source_query = input("\nEnter Source Landmark/Address (e.g., 'IIM Ahmedabad'): ")
dest_query = input("Enter Destination Landmark/Address (e.g., 'Kankaria Lake, Ahmedabad'): ")
time_budget = int(input("Enter Time Budget in seconds (e.g., 600 for 10 mins): "))

# 2. GEOCODE LOCATIONS (Convert text to Lat/Lon)
print(f"\n[1/5] Geocoding '{source_query}'...")
try:
    src_lat, src_lon = ox.geocode(source_query)
except Exception as e:
    print(f"Error: Could not find source location. Try adding ', Ahmedabad'. Details: {e}")
    exit()

print(f"[2/5] Geocoding '{dest_query}'...")
try:
    dst_lat, dst_lon = ox.geocode(dest_query)
except Exception as e:
    print(f"Error: Could not find destination location. Try adding ', Ahmedabad'. Details: {e}")
    exit()

# 3. DOWNLOAD DYNAMIC MAP AREA
# Calculate the midpoint between source and destination
mid_lat = (src_lat + dst_lat) / 2
mid_lon = (src_lon + dst_lon) / 2

# Calculate the straight-line distance between the points (Haversine formula)
R = 6371e3 # Earth radius in meters
phi1, phi2 = math.radians(src_lat), math.radians(dst_lat)
dphi = math.radians(dst_lat - src_lat)
dlambda = math.radians(dst_lon - src_lon)
a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
dist_meters = R * c

# Set download radius: half the distance + 1.5km padding for alternate routes
radius = (dist_meters / 2) + 1500 

print(f"[3/5] Downloading map data (Radius: {radius:.0f} meters around route)...")
try:
    G = ox.graph_from_point((mid_lat, mid_lon), dist=radius, network_type='drive')
except Exception as e:
    print(f"Error downloading map data: {e}")
    exit()

# Snap coordinates to the nearest actual road nodes on the graph
source_node = ox.distance.nearest_nodes(G, src_lon, src_lat)
dest_node = ox.distance.nearest_nodes(G, dst_lon, dst_lat)

if source_node == dest_node:
    print("Error: Source and Destination mapped to the exact same node! Try locations further apart.")
    exit()

# 4. SIMULATE STOCHASTIC TRAFFIC & COMPUTE HEURISTIC
print("[4/5] Assigning stochastic traffic distributions & computing DP Heuristic...")
for u, v, key, data in G.edges(keys=True, data=True):
    length = data.get('length', 10) # meters
    # Base speed ~ 40 km/h (11.1 m/s)
    mean_time = length / 11.1 
    # Traffic uncertainty variance
    variance_time = (mean_time * 0.3) ** 2 
    data['mean_t'] = mean_time
    data['var_t'] = variance_time

G_rev = G.reverse(copy=True)
heuristic_means = nx.single_source_dijkstra_path_length(G_rev, dest_node, weight='mean_t')

def get_heuristic_prob(node, current_time, current_var, budget):
    """Estimates the probability of arriving on time from 'node'."""
    if node not in heuristic_means:
        return 0.0
    rem_mean = heuristic_means[node]
    rem_var = rem_mean * 0.3 
    total_mean = current_time + rem_mean
    total_var = current_var + rem_var
    
    if total_var == 0:
        return 1.0 if total_mean <= budget else 0.0
    return norm.cdf(budget, loc=total_mean, scale=np.sqrt(total_var))

# 5. SOTA A* SEARCH ALGORITHM
print("[5/5] Running SOTA A* Search...")
def sota_a_star(graph, source, target, budget):
    initial_heuristic = get_heuristic_prob(source, 0, 0, budget)
    pq = [(-initial_heuristic, source, [source], 0.0, 0.0)]
    visited = set()
    
    while pq:
        neg_prob, current, path, t_mean, t_var = heapq.heappop(pq)
        prob = -neg_prob
        
        if current == target:
            return path, prob, t_mean
            
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor in graph.neighbors(current):
            if neighbor in visited: continue
            edge_data = graph.get_edge_data(current, neighbor)[0]
            new_t_mean = t_mean + edge_data['mean_t']
            new_t_var = t_var + edge_data['var_t']
            new_path = path + [neighbor]
            
            reliability = get_heuristic_prob(neighbor, new_t_mean, new_t_var, budget)
            
            if reliability > 0.01:
                heapq.heappush(pq, (-reliability, neighbor, new_path, new_t_mean, new_t_var))
                
    return None, 0, 0

optimal_path, success_prob, exp_time = sota_a_star(G, source_node, dest_node, time_budget)

if not optimal_path:
    print(f"\n❌ No path found that meets the {time_budget}s time budget!")
else:
    print("\n✅ OPTIMAL SOTA PATH FOUND!")
    print(f"   Expected Travel Time: {exp_time:.2f} seconds")
    print(f"   Probability of On-Time Arrival: {success_prob*100:.2f}%")

    print("\nGenerating interactive HTML map...")
    route_coords = []
    for node in optimal_path:
        lat = G.nodes[node]['y']
        lon = G.nodes[node]['x']
        route_coords.append((lat, lon))
        
    center_lat = sum(p[0] for p in route_coords) / len(route_coords)
    center_lon = sum(p[1] for p in route_coords) / len(route_coords)
    
    route_map = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
    folium.PolyLine(
        route_coords, color='red', weight=5, opacity=0.8, tooltip="Optimal SOTA Path"
    ).add_to(route_map)
    
    folium.Marker(
        route_coords[0], popup=f'Source: {source_query}', icon=folium.Icon(color='green', icon='play')
    ).add_to(route_map)
    
    folium.Marker(
        route_coords[-1], popup=f'Destination: {dest_query}', icon=folium.Icon(color='red', icon='stop')
    ).add_to(route_map)
    
    filename = "Randomized_sota_map.html"
    route_map.save(filename)
    print(f"Done! Open '{filename}' in your web browser.")