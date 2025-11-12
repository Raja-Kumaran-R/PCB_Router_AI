import re
import json
import sys
import math


class KiCadParser:
    def __init__(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            self.data = f.read()

    # -------------------------
    # Board & layers
    # -------------------------
    def parse_boundary(self):
        """Extract board boundary from gr_rect in Edge.Cuts."""
        boundary = []
        rect_match = re.search(
            r'\(gr_rect\s*'
            r'\(start\s+([\d\.\-]+)\s+([\d\.\-]+)\)\s*'
            r'\(end\s+([\d\.\-]+)\s+([\d\.\-]+)\)[\s\S]*?'
            r'\(layer\s+"Edge\.Cuts"\)',
            self.data,
            re.I
        )
        if rect_match:
            x1, y1, x2, y2 = map(float, rect_match.groups())
            boundary = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return boundary

    def parse_layers(self):
        """Extract only signal layers."""
        layers = []
        for m in re.finditer(r'\(\s*(\d+)\s+"([^"]+)"\s+signal', self.data):
            layers.append(m.group(2))
        return layers

    # -------------------------
    # Nets & netclasses
    # -------------------------
    def parse_nets(self):
        """Extract nets (id -> name)."""
        nets = {}
        for m in re.finditer(r'\(net (\d+) "([^"]+)"\)', self.data):
            nets[int(m.group(1))] = m.group(2)
        return nets

    def classify_net(self, net_name: str) -> str:
        """
        Heuristic net-to-class mapping based on name.
        POWER: supplies & grounds
        SWITCH: switch/inductor node (e.g., OUT/SW)
        FEEDBACK: FB/feedback nets
        Default: everything else
        """
        up = (net_name or "").upper()

        # Grounds
        if up in {"GND", "PGND", "AGND", "DGND"}:
            return "POWER"
        # Supplies
        if up in {"VCC", "VDD", "VBAT", "VIN", "5V", "3V3", "1V8", "2V5"}:
            return "POWER"
        # Common naming for switch/switching node
        if up.startswith("SW") or "SWITCH" in up or "OUT" in up:
            return "SWITCH"
        # Feedback nets
        if "FB" in up or "FEEDBACK" in up:
            return "FEEDBACK"

        return "Default"

    def build_netclasses(self, board_layers, nets_dict):
        """
        Create:
          - netclasses: class templates
          - net_to_class: mapping for both names and ids (as strings)
        Ensures allowed_layers exist on this board.
        """
        # Class templates (you can tune these defaults)
        netclasses = {
            "Default": { "trace_width": 0.2, "clearance": 0.20, "via_size": 0.6, "via_drill": 0.3, "via_cost": 10, "min_via_from_pads": 0.2, "reuse_same_net_via": False },
            "POWER":   { "trace_width": 0.2, "clearance": 0.20, "via_size": 0.6, "via_drill": 0.3, "via_cost":  10, "min_via_from_pads": 0.2, "reuse_same_net_via": False },
            "SWITCH":  { "trace_width": 0.2, "clearance": 0.20, "via_size": 0.6, "via_drill": 0.3,
                         "via_cost": 10, "min_via_from_pads": 0.2, "reuse_same_net_via": False},
            "FEEDBACK":{ "trace_width": 0.2, "clearance": 0.20, "via_cost": 10, "min_via_from_pads": 0.2, "reuse_same_net_via": False }
        }

        # Make sure any allowed_layers listed actually exist on this board
        layer_set = set(board_layers or [])
        for cls_name, cls_rules in netclasses.items():
            if "allowed_layers" in cls_rules:
                cls_rules["allowed_layers"] = [L for L in cls_rules["allowed_layers"] if L in layer_set]
                if not cls_rules["allowed_layers"]:
                    # If none of the requested layers exist, drop the restriction
                    cls_rules.pop("allowed_layers", None)

        # Build mapping (by name and by id string for convenience)
        net_to_class = {}
        for nid, nname in (nets_dict or {}).items():
            cls = self.classify_net(nname)
            net_to_class[nname] = cls          # map by name
            net_to_class[str(nid)] = cls       # and by id (string)

        return netclasses, net_to_class

    # -------------------------
    # Footprints / pads / segments
    # -------------------------
    def extract_footprint_blocks(self):
        """
        Extract full footprint blocks by manually parsing parentheses.
        Returns list of tuples (footprint_name, fx, fy, rot, footprint_block_text).
        """
        data = self.data
        footprints = []
        pattern = re.compile(r'\(footprint\s+"([^"]+)"')
        for m in pattern.finditer(data):
            start = m.start()
            count = 0
            end = None
            for i in range(start, len(data)):
                if data[i] == '(':
                    count += 1
                elif data[i] == ')':
                    count -= 1
                    if count == 0:
                        end = i + 1
                        break
            if end is None:
                continue
            block_text = data[start:end]
            # Position & rotation from footprint header
            at_match = re.search(r'\(at\s+([\d\.\-]+)\s+([\d\.\-]+)(?:\s+([\d\.\-]+))?\)', block_text)
            if at_match:
                fx = float(at_match.group(1))
                fy = float(at_match.group(2))
                frot = float(at_match.group(3)) if at_match.group(3) else 0.0
                footprints.append((m.group(1), fx, fy, frot, block_text))
        return footprints

    def parse_footprints_and_pads(self):
        pads = []
        footprints = self.extract_footprint_blocks()

        for fp_name, fx, fy, frot, fp_block in footprints:
            theta = -math.radians(frot)
            ct, st = math.cos(theta), math.sin(theta)
            # Scan footprint block for each (pad … ) block
            pos = 0
            while True:
                i = fp_block.find('(pad', pos)
                if i == -1:
                    break
                count = 0
                end = None
                for j in range(i, len(fp_block)):
                    if fp_block[j] == '(':
                        count += 1
                    elif fp_block[j] == ')':
                        count -= 1
                        if count == 0:
                            end = j + 1
                            break
                if end is None:
                    break
                pad_block = fp_block[i:end]
                pos = end

                # Pad name, type, shape
                h = re.search(r'\(pad\s+"?([^"\s]+)"?\s+([^\s]+)\s+([^\s]+)', pad_block)
                if not h:
                    continue
                pad_name = h.group(1)
                pad_type = h.group(2).lower()
                pad_shape = h.group(3).lower()

                # at (x y rot?)
                at_m = re.search(r'\(at\s+([^\)]+)\)', pad_block)
                if at_m:
                    parts = at_m.group(1).split()
                    px_local = float(parts[0])
                    py_local = float(parts[1]) if len(parts) > 1 else 0.0
                    pad_rot = float(parts[2]) if len(parts) > 2 else 0.0
                else:
                    px_local, py_local, pad_rot = 0.0, 0.0, 0.0

                # size
                size_m = re.search(r'\(size\s+([\d\.\-]+)\s+([\d\.\-]+)\)', pad_block)
                size_x = float(size_m.group(1)) if size_m else 0.0
                size_y = float(size_m.group(2)) if size_m else 0.0

                # drill (optional)
                drill_m = re.search(r'\(drill\s+([\d\.\-]+)\)', pad_block)
                drill = float(drill_m.group(1)) if drill_m else None

                # layers
                layers_m = re.search(r'\(layers\s+([^\)]+)\)', pad_block)
                layer = None
                if layers_m:
                    layer = layers_m.group(1).split()[0].replace('"', '')

                # net (optional)
                net_m = re.search(r'\(net\s+(\d+)\s+"([^"]+)"\)', pad_block)
                net_id = int(net_m.group(1)) if net_m else None
                net_name = net_m.group(2) if net_m else None

                # Apply footprint rotation to local pad coords
                px_rot = px_local * ct - py_local * st
                py_rot = px_local * st + py_local * ct
                abs_x = fx + px_rot
                abs_y = fy + py_rot

                # Absolute pad rotation (for router blocking/clearing)
                is_tht = bool(layers_m and '*.Cu' in layers_m.group(1))
                if is_tht:
                    abs_rot = float(pad_rot if pad_rot is not None else frot) % 360.0
                else:
                    abs_rot = float((frot + (pad_rot or 0.0)) % 360.0)
                pads.append({
                    "type": "pad",
                    "pad_name": pad_name,
                    "x": abs_x,
                    "y": abs_y,
                    "layer": layer,          # or omit if pad is on all copper layers
                    "net_id": net_id,
                    "net": net_name,
                    "size_x": size_x,
                    "size_y": size_y,
                    "shape": pad_shape,      # "rect" / "roundrect" / "circle"
                    "pad_rotation": pad_rot,
                    "abs_rotation": abs_rot,
                    "clearance": 0.20,
                    "drill": drill
                })

        return pads

    def parse_segments(self):
        """Extract copper segments as obstacles (optional; router may ignore)."""
        segments = []
        for m in re.finditer(
            r'\(segment\s+\(start ([\d\.\-]+) ([\d\.\-]+)\)\s+\(end ([\d\.\-]+) ([\d\.\-]+)\)\s+\(width ([\d\.]+)\)\s+\(layer "([^"]+)"\)',
            self.data
        ):
            segments.append({
                "type": "segment",
                "layer": m.group(7),
                "start": [float(m.group(1)), float(m.group(2))],
                "end":   [float(m.group(3)), float(m.group(4))],
                "width": float(m.group(5)),
                "clearance": 0.2
            })
        return segments

    # -------------------------
    # Tasks
    # -------------------------
    def parse_tasks(self):
        """Generate routing tasks using absolute pad coordinates from footprints."""
        tasks = []
        net_to_pads = {}

        pads = self.parse_footprints_and_pads()
        for pad in pads:
            net_id = pad["net_id"]
            if net_id is None:
                continue
            net_to_pads.setdefault(net_id, []).append(pad)

        # Connect in the order found (simple 2-pin/chain assumption)
        for net_id, pads in net_to_pads.items():
            if len(pads) >= 2:
                for i in range(len(pads) - 1):
                    tasks.append({
                        "net": pads[i]["net"],
                        "net_id": net_id,
                        "start": {"x": pads[i]["x"], "y": pads[i]["y"], "layer": pads[i]["layer"]},
                        "goal":  {"x": pads[i+1]["x"], "y": pads[i+1]["y"], "layer": pads[i+1]["layer"]}
                    })
        return tasks

    # -------------------------
    # Aggregate
    # -------------------------
    def parse_all(self):
        layers = self.parse_layers()
        nets_dict = self.parse_nets()
        netclasses, net_to_class = self.build_netclasses(layers, nets_dict)

        return {
            "board": {
                "boundary": self.parse_boundary(),
                "layers": layers
            },
            "rules": {
                # Base/defaults (router can be overridden per netclass)
                "grid_step": 0.1,
                "clearance": 0.20,
                "trace_width": 0.20,
                "via_size": 0.6,
                "via_drill": 0.3,
                "via_cost": 20,

                # Global halos (independent of class)
                "board_clearance": 0.4,
                "via_clearance_extra": 0.1,
                "pad_clearance_extra": 0.1,
                "pad_opener_safety": 0.05,
                "trace_clearance_extra": 0.1,
                

                # Netclass system
                "default_netclass": "Default",
                "netclasses": netclasses,
                "net_to_class": net_to_class
            },
            "obstacles": self.parse_footprints_and_pads() + self.parse_segments(),
            "tasks": self.parse_tasks()
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python parser.py <kicad_pcb_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    parser = KiCadParser(filepath)
    parsed_data = parser.parse_all()
    out_path = "input.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, indent=2)

    print(f"✅ Parsed data written to {out_path}")

    # -------------------------
    # Summary counts
    # -------------------------
    obstacles = parsed_data["obstacles"]
    tasks = parsed_data["tasks"]

    smd_pads = 0
    tht_pads = 0
    traces = 0
    vias = 0   # placeholder (not parsed yet)
    others = 0

    for obs in obstacles:
        if obs["type"] == "pad":
            if obs.get("drill") is not None:   # drill → THT pad
                tht_pads += 1
            else:                              # no drill → SMD pad
                smd_pads += 1
        elif obs["type"] == "segment":
            traces += 1
        elif obs["type"] == "via":  # if you add via parsing later
            vias += 1
        else:
            others += 1

    print("\n--- Board Summary ---")
    print(f"SMD pads : {smd_pads}")
    print(f"THT pads : {tht_pads}")
    print(f"Traces   : {traces}")
    print(f"Vias     : {vias}")
    print(f"Others   : {others}")
    print("---------------------")
    print(f"Total routing tasks (connections to establish): {len(tasks)}")



if __name__ == "__main__":
    main()
