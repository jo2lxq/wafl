#!/usr/bin/env python3
"""
Unified WAFL-Testbed Mobility Preprocessing Pipeline.

This script consolidates the mobility preprocessing workflow:
1. Generate SUMO network and routes from execution_config.json
2. Run SUMO simulation to generate mobility trace
3. Map trace to contact pattern and network conditions

Usage:
  python prepare_mobility.py --config ctrl/execution_config.json --output-dir data/sumo/
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

# ==============================================================================
# SUMO Network and Route Generation
# ==============================================================================


def load_execution_config(config_path: str) -> dict:
    """Load execution configuration file."""
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {config_path}: {e}", file=sys.stderr)
        sys.exit(1)


def generate_edge_xml(edge_id: str, from_jid: int, to_jid: int, x1: float, y1: float, x2: float, y2: float) -> str:
    """Generate XML for a single edge with proper geometry."""
    # Calculate edge length
    edge_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Calculate lane shape
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)

    if length > 0:
        perp_x = -dy / length * 1.6
        perp_y = dx / length * 1.6

        lane_x1 = x1 + perp_x
        lane_y1 = y1 + perp_y
        lane_x2 = x2 + perp_x
        lane_y2 = y2 + perp_y

        shape = f"{lane_x1:.2f},{lane_y1:.2f} {lane_x2:.2f},{lane_y2:.2f}"
    else:
        shape = f"{x1:.2f},{y1:.2f} {x2:.2f},{y2:.2f}"

    xml = f'    <edge id="{edge_id}" from="J{from_jid}" to="J{to_jid}" priority="1" numLanes="1" speed="13.89">\n'
    xml += f'        <lane id="{edge_id}_0" index="0" speed="13.89" length="{edge_length:.2f}" shape="{shape}"/>\n'
    xml += "    </edge>\n"

    return xml


def generate_manhattan_network(output_path: str, num_nodes: int):
    """Generate Manhattan grid SUMO network file."""
    grid_size = max(3, int(math.ceil(math.sqrt(num_nodes * 2))))
    block_length = 200
    offset_x = 100
    offset_y = 100

    junctions = []
    junction_map = {}
    junction_id = 0

    for row in range(grid_size):
        for col in range(grid_size):
            x = offset_x + col * block_length
            y = offset_y + row * block_length
            junctions.append((junction_id, x, y))
            junction_map[(row, col)] = junction_id
            junction_id += 1

    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<net version="1.20" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNameSpaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">

    <location netOffset="0.00,0.00" convBoundary="0.00,0.00,2000.00,2000.00" origBoundary="-10000000000.00,-10000000000.00,10000000000.00,10000000000.00" projParameter="!"/>

"""

    # Generate junctions
    xml_content += "    <!-- Junctions (Intersections) -->\n"
    for jid, x, y in junctions:
        shape_points = [
            f"{x - 3:.2f},{y + 3:.2f}",
            f"{x + 3:.2f},{y + 3:.2f}",
            f"{x + 3:.2f},{y - 3:.2f}",
            f"{x - 3:.2f},{y - 3:.2f}",
        ]
        shape = " ".join(shape_points)
        xml_content += f'    <junction id="J{jid}" type="priority" x="{x:.2f}" y="{y:.2f}" incLanes="" intLanes="" shape="{shape}"/>\n'

    # Generate edges
    xml_content += "\n    <!-- Edges (Roads) -->\n"
    edge_list = []
    edge_id = 0

    # Horizontal roads
    for row in range(grid_size):
        for col in range(grid_size - 1):
            from_jid = junction_map[(row, col)]
            to_jid = junction_map[(row, col + 1)]

            from_x, from_y = junctions[from_jid][1], junctions[from_jid][2]
            to_x, to_y = junctions[to_jid][1], junctions[to_jid][2]

            edge_name = f"E{edge_id}"
            edge_list.append((edge_name, from_jid, to_jid))
            xml_content += generate_edge_xml(edge_name, from_jid, to_jid, from_x, from_y, to_x, to_y)
            edge_id += 1

            edge_name = f"E{edge_id}"
            edge_list.append((edge_name, to_jid, from_jid))
            xml_content += generate_edge_xml(edge_name, to_jid, from_jid, to_x, to_y, from_x, from_y)
            edge_id += 1

    # Vertical roads
    for col in range(grid_size):
        for row in range(grid_size - 1):
            from_jid = junction_map[(row, col)]
            to_jid = junction_map[(row + 1, col)]

            from_x, from_y = junctions[from_jid][1], junctions[from_jid][2]
            to_x, to_y = junctions[to_jid][1], junctions[to_jid][2]

            edge_name = f"E{edge_id}"
            edge_list.append((edge_name, from_jid, to_jid))
            xml_content += generate_edge_xml(edge_name, from_jid, to_jid, from_x, from_y, to_x, to_y)
            edge_id += 1

            edge_name = f"E{edge_id}"
            edge_list.append((edge_name, to_jid, from_jid))
            xml_content += generate_edge_xml(edge_name, to_jid, from_jid, to_x, to_y, from_x, from_y)
            edge_id += 1

    # Generate connections
    xml_content += "\n    <!-- Connections at Intersections -->\n"
    for row in range(grid_size):
        for col in range(grid_size):
            jid = junction_map[(row, col)]

            incoming = [e for e in edge_list if e[2] == jid]
            outgoing = [e for e in edge_list if e[1] == jid]

            for in_edge, in_from, _ in incoming:
                for out_edge, _, out_to in outgoing:
                    if in_from != out_to:
                        xml_content += f'    <connection from="{in_edge}" to="{out_edge}" fromLane="0" toLane="0" dir="s" state="M"/>\n'

    xml_content += "\n</net>\n"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(xml_content)

    print(f"✅ Generated Manhattan grid network: {grid_size}x{grid_size}, {len(junctions)} junctions, {edge_id} edges")

    return grid_size, edge_id


def generate_routes(
    config: dict,
    output_path: str,
    grid_size: int,
    num_h_edges: int,
    target_epochs: int = None,
):
    """Generate SUMO route file.

    Args:
        config: Execution configuration with nodes
        output_path: Path to write routes file
        grid_size: Size of the Manhattan grid
        num_h_edges: Number of horizontal edges
        target_epochs: Target number of epochs for simulation (determines route length)
    """
    nodes = config.get("nodes", [])
    num_vehicles = len(nodes)

    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNameSpaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    
    <!-- Vehicle type definition -->
    <vType id="type1" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="13.89" guiShape="passenger"/>
    
    <!-- Vehicle definitions (mapped from execution_config.json nodes) -->
"""

    for idx, node in enumerate(nodes):
        veh_id = str(node["name"])

        start_row = random.randint(0, grid_size - 2)
        start_col = random.randint(0, grid_size - 2)

        route_edges = []

        def get_h_edge(row, col, direction):
            edge_num = row * (grid_size - 1) * 2 + col * 2 + direction
            return f"E{edge_num}"

        def get_v_edge(row, col, direction):
            edge_num = num_h_edges + col * (grid_size - 1) * 2 + row * 2 + direction
            return f"E{edge_num}"

        # Calculate loops needed based on target epochs
        # Each loop takes roughly 8-16 edges at ~14 m/s speed with 200m blocks
        # Estimate: each loop takes about 30-50 simulation steps
        # Use generous multiplier to ensure simulation runs long enough
        if target_epochs:
            # Each loop generates ~16 edges, each edge ~200m at ~14m/s takes ~14 steps
            # But with traffic and stops, estimate ~40-60 steps per loop
            # Use conservative estimate: 30 steps per loop
            estimated_steps_per_loop = 30
            loops = max(2, (target_epochs // estimated_steps_per_loop) + 5)  # Add buffer
        else:
            loops = 2

        for _ in range(loops):
            for c in range(start_col, min(start_col + 2, grid_size - 1)):
                route_edges.append(get_h_edge(start_row, c, 0))

            for r in range(start_row, min(start_row + 2, grid_size - 1)):
                route_edges.append(get_v_edge(r, min(start_col + 2, grid_size - 1), 0))

            for c in range(min(start_col + 2, grid_size - 1) - 1, start_col - 1, -1):
                route_edges.append(get_h_edge(min(start_row + 2, grid_size - 1), c, 1))

            for r in range(min(start_row + 2, grid_size - 1) - 1, start_row - 1, -1):
                route_edges.append(get_v_edge(r, start_col, 1))

        edges_str = " ".join(route_edges)
        depart_time = idx * 5

        xml_content += f'    <vehicle id="{veh_id}" type="type1" depart="{depart_time}">\n'
        xml_content += f'        <route edges="{edges_str}"/>\n'
        xml_content += "    </vehicle>\n"

    xml_content += "\n</routes>\n"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(xml_content)

    print(f"✅ Generated SUMO routes: {num_vehicles} vehicles")


def generate_sumocfg(output_path: str, network_file: str, route_file: str, duration: int = 1000):
    """Generate SUMO configuration file."""
    output_dir = os.path.dirname(output_path)
    net_rel = os.path.relpath(network_file, output_dir)
    rou_rel = os.path.relpath(route_file, output_dir)

    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNameSpaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="{net_rel}"/>
        <route-files value="{rou_rel}"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="{duration}"/>
        <step-length value="1"/>
    </time>

    <processing>
        <collision.action value="none"/>
    </processing>

    <report>
        <verbose value="true"/>
        <no-step-log value="false"/>
    </report>

</configuration>
"""

    with open(output_path, "w") as f:
        f.write(xml_content)

    print(f"✅ Generated SUMO configuration: {output_path}")


# ==============================================================================
# SUMO Simulation
# ==============================================================================


def run_sumo_simulation(
    sumo_config: str,
    output_file: str,
    step_duration: float = 1.0,
    target_epochs: int = None,
):
    """Run SUMO simulation and generate mobility trace."""
    try:
        import traci
    except ImportError:
        print("❌ Error: SUMO/traci (Python library) not found.")
        print("")
        print("Please install SUMO and its Python bindings:")
        print("  Ubuntu/Debian:")
        print("    sudo add-apt-repository ppa:sumo/stable")
        print("    sudo apt-get update")
        print("    sudo apt-get install sumo sumo-tools sumo-doc")
        print("")
        print("  Then install Python bindings:")
        print("    pip install traci")
        print("")
        print("  Official documentation: https://sumo.dlr.de/docs/Installing/index.html")
        sys.exit(1)

    print(f"🚀 Running SUMO simulation: {sumo_config}")

    # Try to find SUMO binary in common installation paths
    sumo_paths = [
        "/usr/bin/sumo",
        "/usr/local/bin/sumo",
        "/usr/share/sumo/bin/sumo",
        "/opt/sumo/bin/sumo",
    ]

    # Check SUMO_HOME environment variable as optional fallback
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        sumo_paths.insert(0, os.path.join(sumo_home, "bin", "sumo"))

    sumo_binary = None
    for path in sumo_paths:
        if os.path.exists(path):
            sumo_binary = path
            break

    if not sumo_binary:
        print("❌ Error: SUMO binary not found.")
        print("")
        print("Please install SUMO:")
        print("  Ubuntu/Debian:")
        print("    sudo add-apt-repository ppa:sumo/stable")
        print("    sudo apt-get update")
        print("    sudo apt-get install sumo sumo-tools sumo-doc")
        print("")
        print("  Alternatively, set SUMO_HOME environment variable:")
        print("    export SUMO_HOME=/path/to/sumo")
        print("")
        print(f"  Searched paths: {', '.join(sumo_paths)}")
        print("  Official documentation: https://sumo.dlr.de/docs/Installing/index.html")
        sys.exit(1)

    sumo_cmd = [sumo_binary, "-c", sumo_config, "--step-length", str(step_duration)]

    traci.start(sumo_cmd)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "node_id", "x", "y"])

        epoch = 0
        last_positions = {}  # Store last known positions for all vehicles
        all_vehicle_ids = set()  # Track all vehicle IDs that have appeared

        # Phase 1: Run simulation while vehicles are still active
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            vehicle_ids = traci.vehicle.getIDList()
            for veh_id in vehicle_ids:
                x, y = traci.vehicle.getPosition(veh_id)
                writer.writerow([epoch, veh_id, f"{x:.2f}", f"{y:.2f}"])
                last_positions[veh_id] = (x, y)
                all_vehicle_ids.add(veh_id)

            if epoch % 100 == 0 and epoch > 0:
                print(f"  Processed epoch {epoch}...")

            epoch += 1

            # Check if we've reached target epochs
            if target_epochs is not None and epoch >= target_epochs:
                break

        # Phase 2: If target_epochs is set and not yet reached, continue with last positions
        if target_epochs is not None and epoch < target_epochs:
            print(f"  Simulation ended at epoch {epoch}, continuing to {target_epochs} using last positions...")
            while epoch < target_epochs:
                for veh_id in sorted(all_vehicle_ids, key=lambda x: int(x)):
                    if veh_id in last_positions:
                        x, y = last_positions[veh_id]
                        writer.writerow([epoch, veh_id, f"{x:.2f}", f"{y:.2f}"])

                if epoch % 100 == 0:
                    print(f"  Processed epoch {epoch}...")

                epoch += 1

    traci.close()

    print(f"✅ Mobility trace saved: {output_file} ({epoch} epochs)")
    return epoch


# ==============================================================================
# Trace to Contact Pattern and Network Conditions
# ==============================================================================


def load_mobility_trace(trace_file: str) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """Load mobility trace from CSV."""
    trace = {}
    with open(trace_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            node_id = row["node_id"]
            x = float(row["x"])
            y = float(row["y"])

            if epoch not in trace:
                trace[epoch] = {}
            trace[epoch][node_id] = (x, y)

    return trace


def load_path_loss_model(model_file: str) -> dict:
    """Load path loss model configuration."""
    with open(model_file) as f:
        return json.load(f)


def calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two positions."""
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def get_rank_for_distance(distance: float, model: dict) -> dict:
    """Determine network quality rank based on distance."""
    if distance > model["radio_range"]:
        return None

    for rank in model["ranks"]:
        if rank["distance_min"] <= distance <= rank["distance_max"]:
            return rank

    return None


def generate_contact_and_conditions(trace: Dict, model: dict, execution_config: dict):
    """Generate contact pattern and network conditions from trace and model."""
    contact_pattern = []
    network_conditions = []

    epochs = sorted(trace.keys())

    for epoch in epochs:
        positions = trace[epoch]
        node_ids = sorted(positions.keys(), key=lambda x: int(x))

        contacts = defaultdict(list)
        epoch_conditions = {}

        for i, node_i in enumerate(node_ids):
            pos_i = positions[node_i]
            peers_list = []

            for j, node_j in enumerate(node_ids):
                if i >= j:
                    continue

                pos_j = positions[node_j]
                distance = calculate_distance(pos_i, pos_j)

                rank = get_rank_for_distance(distance, model)

                if rank:
                    # node_i and node_j are strings from CSV, convert to int for consistency
                    contacts[node_i].append(int(node_j))
                    contacts[node_j].append(int(node_i))

                    peers_list.append(
                        {
                            "peer": int(node_j),
                            "rank": rank["name"],
                            "rate": rank["rate"],
                            "delay": rank["delay"],
                            "loss": rank["loss"],
                        }
                    )

            epoch_conditions[f"node_{node_i}"] = peers_list

        # Add reverse direction peers
        for node_id in node_ids:
            if f"node_{node_id}" not in epoch_conditions:
                epoch_conditions[f"node_{node_id}"] = []
            else:
                existing_peers = {p["peer"] for p in epoch_conditions[f"node_{node_id}"]}
                for other_id in contacts[node_id]:
                    other_int = int(other_id)
                    if other_int not in existing_peers:
                        pos_node = positions[node_id]
                        pos_other = positions[str(other_id)]
                        distance = calculate_distance(pos_node, pos_other)
                        rank = get_rank_for_distance(distance, model)

                        if rank:
                            epoch_conditions[f"node_{node_id}"].append(
                                {
                                    "peer": other_int,
                                    "rank": rank["name"],
                                    "rate": rank["rate"],
                                    "delay": rank["delay"],
                                    "loss": rank["loss"],
                                }
                            )

        # Convert to required format
        contact_dict = {int(node_id): sorted(contacts[node_id]) for node_id in node_ids}
        contact_pattern.append(contact_dict)

        epoch_cond_dict = {}
        for node_id in node_ids:
            node_key = int(node_id)
            if f"node_{node_id}" in epoch_conditions:
                epoch_cond_dict[node_key] = epoch_conditions[f"node_{node_id}"]
            else:
                epoch_cond_dict[node_key] = []

        network_conditions.append(epoch_cond_dict)

    return contact_pattern, network_conditions


# ==============================================================================
# Main Pipeline
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Unified WAFL-Testbed Mobility Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script combines:
  1. SUMO network and route generation from execution_config.json
  2. SUMO simulation to generate mobility trace
  3. Trace mapping to contact pattern and network conditions

Example:
  python prepare_mobility.py \\
    --config ctrl/execution_config.json \\
    --pathloss data/path_loss_model.json \\
    --output-dir data/sumo/
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="ctrl/execution_config.json",
        help="Path to execution_config.json (default: ctrl/execution_config.json)",
    )
    parser.add_argument(
        "--pathloss",
        type=str,
        default="data/sumo/path_loss_model.json",
        help="Path to path loss model JSON (default: data/sumo/path_loss_model.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sumo/",
        help="Output directory (default: data/sumo/)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10000,
        help="SUMO simulation duration in seconds (default: 10000, only used if --epochs not specified)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Target number of epochs to generate (overrides --duration, continues until target is reached)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.config).exists():
        print(f"❌ Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    if not Path(args.pathloss).exists():
        print(f"❌ Error: Path loss model not found: {args.pathloss}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("WAFL-Testbed Mobility Preprocessing Pipeline")
    print("=" * 80)

    # Step 1: Load configuration
    print("\n📋 Step 1: Loading configuration...")
    exec_config = load_execution_config(args.config)
    path_loss_model = load_path_loss_model(args.pathloss)
    num_nodes = len(exec_config.get("nodes", []))
    print(f"✅ Loaded config: {num_nodes} nodes, radio range: {path_loss_model['radio_range']}m")

    # Step 2: Generate SUMO network and routes
    print("\n🏗️  Step 2: Generating SUMO network and routes...")
    network_file = str(output_dir / "network.net.xml")
    route_file = str(output_dir / "routes.rou.xml")
    sumocfg_file = str(output_dir / "routes.sumocfg")

    # Determine target epochs and duration
    target_epochs = args.epochs
    if target_epochs:
        # Set duration to be larger than target epochs to ensure simulation runs long enough
        sim_duration = target_epochs + 1000  # Add buffer
    else:
        sim_duration = args.duration
        target_epochs = None

    grid_size, total_edges = generate_manhattan_network(network_file, num_nodes)
    num_h_edges = grid_size * (grid_size - 1) * 2
    generate_routes(exec_config, route_file, grid_size, num_h_edges, target_epochs=target_epochs)
    generate_sumocfg(sumocfg_file, network_file, route_file, sim_duration)

    # Step 3: Run SUMO simulation
    print("\n🚗 Step 3: Running SUMO simulation...")
    trace_file = str(output_dir / "mobility_trace.csv")
    if target_epochs:
        print(f"  Target epochs: {target_epochs}")
    num_epochs = run_sumo_simulation(sumocfg_file, trace_file, target_epochs=target_epochs)

    # Step 4: Generate contact pattern and network conditions
    print("\n🔄 Step 4: Generating contact pattern and network conditions...")
    trace = load_mobility_trace(trace_file)
    contact_pattern, network_conditions = generate_contact_and_conditions(trace, path_loss_model, exec_config)

    # Step 5: Save outputs
    print("\n💾 Step 5: Saving outputs...")
    contact_file = str(output_dir / "contact_pattern_mobility.json")
    conditions_file = str(output_dir / "network_conditions_mobility.json")

    with open(contact_file, "w") as f:
        json.dump(contact_pattern, f, indent=2)
    print(f"✅ Contact pattern saved: {contact_file}")

    with open(conditions_file, "w") as f:
        json.dump(network_conditions, f, indent=2)
    print(f"✅ Network conditions saved: {conditions_file}")

    # Summary
    print("\n" + "=" * 80)
    print("📊 Pipeline Summary:")
    print("=" * 80)
    print(f"  Nodes: {num_nodes}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Network: {grid_size}x{grid_size} Manhattan grid")
    print(f"  Output directory: {output_dir}")
    print("\n  Generated files:")
    print(f"    - {network_file}")
    print(f"    - {route_file}")
    print(f"    - {sumocfg_file}")
    print(f"    - {trace_file}")
    print(f"    - {contact_file}")
    print(f"    - {conditions_file}")
    print("\n✅ Mobility preprocessing pipeline complete!")


if __name__ == "__main__":
    main()
