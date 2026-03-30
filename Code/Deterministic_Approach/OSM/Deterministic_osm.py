import osmnx as ox
import networkx as nx
import folium
import math

print("======================================================")
print(" BASELINE: Deterministic Dijkstra Route Optimizer")
print("======================================================")

# 1. GET DYNAMIC USER INPUT
source_query = input("\nEnter Source Landmark/Address (e.g., 'IIM Ahmedabad'): ")
dest_query = input("Enter Destination Landmark/Address (e.g., 'Kankaria Lake, Ahmedabad'): ")

# 2. GEOCODE LOCATIONS (Convert text to Lat/Lon)
print(f"\n[1/4] Geocoding '{source_query}'...")
try:
    src_lat, src_lon = ox.geocode(source_query)
except Exception as e:
    print(f"Error: Could not find source location. Try adding ', Ahmedabad'. Details: {e}")
    exit()

print(f"[2/4] Geocoding '{dest_query}'...")
try:
    dst_lat, dst_lon = ox.geocode(dest_query)
except Exception as e:
    print(f"Error: Could not find destination location. Try adding ', Ahmedabad'. Details: {e}")
    exit()

# 3. DOWNLOAD DYNAMIC MAP AREA
# Calculate the midpoint and distance for the download radius
mid_lat = (src_lat + dst_lat) / 2
mid_lon = (src_lon + dst_lon) / 2

R = 6371e3 # Earth radius in meters
phi1, phi2 = math.radians(src_lat), math.radians(dst_lat)
dphi = math.radians(dst_lat - src_lat)
dlambda = math.radians(dst_lon - src_lon)
a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
dist_meters = R * c

# Set download radius: half the distance + 1.5km padding
radius = (dist_meters / 2) + 1500 

print(f"[3/4] Downloading map data (Radius: {radius:.0f} meters around route)...")
try:
    G = ox.graph_from_point((mid_lat, mid_lon), dist=radius, network_type='drive')
except Exception as e:
    print(f"Error downloading map data: {e}")
    exit()

source_node = ox.distance.nearest_nodes(G, src_lon, src_lat)
dest_node = ox.distance.nearest_nodes(G, dst_lon, dst_lat)

if source_node == dest_node:
    print("Error: Source and Destination mapped to the exact same node! Try locations further apart.")
    exit()

# 4. COMPUTE DETERMINISTIC WEIGHTS & RUN DIJKSTRA
print("[4/4] Assigning deterministic weights and running Dijkstra...")
# In a deterministic model, we ONLY care about the expected travel time, ignoring variance.
for u, v, key, data in G.edges(keys=True, data=True):
    length = data.get('length', 10) # meters
    # Base speed ~ 40 km/h (11.1 m/s)
    expected_time = length / 11.1 
    data['travel_time'] = expected_time

# Run standard Dijkstra's algorithm minimizing 'travel_time'
try:
    # nx.shortest_path uses Dijkstra under the hood when weights are strictly positive
    optimal_path = nx.shortest_path(G, source_node, dest_node, weight='travel_time')
    exp_time = nx.shortest_path_length(G, source_node, dest_node, weight='travel_time')
except nx.NetworkXNoPath:
    optimal_path = None

if not optimal_path:
    print(f"\n❌ No path found between the locations!")
else:
    print("\n✅ DETERMINISTIC PATH FOUND!")
    print(f"   Expected Travel Time: {exp_time:.2f} seconds")
    print("   (Note: This assumes 0 traffic variance and no uncertainty)")

    print("\nGenerating interactive HTML map...")
    route_coords = []
    for node in optimal_path:
        lat = G.nodes[node]['y']
        lon = G.nodes[node]['x']
        route_coords.append((lat, lon))
        
    center_lat = sum(p[0] for p in route_coords) / len(route_coords)
    center_lon = sum(p[1] for p in route_coords) / len(route_coords)
    
    route_map = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
    # Using BLUE for the deterministic path to contrast with SOTA's RED
    folium.PolyLine(
        route_coords, color='blue', weight=5, opacity=0.8, tooltip="Deterministic Dijkstra Path"
    ).add_to(route_map)
    
    folium.Marker(
        route_coords[0], popup=f'Source: {source_query}', icon=folium.Icon(color='green', icon='play')
    ).add_to(route_map)
    
    folium.Marker(
        route_coords[-1], popup=f'Destination: {dest_query}', icon=folium.Icon(color='red', icon='stop')
    ).add_to(route_map)
    
    filename = "Deterministic_dijkstra_map.html"
    route_map.save(filename)
    print(f"Done! Open '{filename}' in your web browser.")