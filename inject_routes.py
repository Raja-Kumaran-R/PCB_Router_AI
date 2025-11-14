import json
import uuid
import sys
import os

def _pick_seg_points(seg):
    """
    Support both formats:
      {"start":[x,y], "end":[x,y], "layer":"F.Cu", "width":0.15}
      {"from":[x,y],  "to":[x,y],  "layer":"F.Cu", "width":0.15}
    """
    if "start" in seg and "end" in seg:
        return seg["start"], seg["end"]
    if "from" in seg and "to" in seg:
        return seg["from"], seg["to"]
    raise KeyError("segment missing both ('start','end') and ('from','to')")

def _unique_out_path(path):
    """
    If 'path' exists, generate 'name(n).ext' that doesn't exist.
    e.g. sample_routed.kicad_pcb -> sample_routed(1).kicad_pcb, etc.
    """
    base, ext = os.path.splitext(path)
    if not os.path.exists(path):
        return path
    n = 1
    while True:
        cand = f"{base}({n}){ext}"
        if not os.path.exists(cand):
            return cand
        n += 1

def _assert_not_same_file(in_path, out_path):
    # Prevent accidental overwrite if user passes same file twice
    in_abs = os.path.abspath(in_path)
    out_abs = os.path.abspath(out_path)
    if in_abs == out_abs:
        raise RuntimeError("Output file must be different from input .kicad_pcb")

def json_to_kicad(json_file, pcb_file, out_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except UnicodeDecodeError:
        with open(json_file, 'r', encoding='utf-16') as f:
            data = json.load(f)

    # Expect router output with "routes"
    if "routes" not in data or not isinstance(data["routes"], list):
        raise RuntimeError(
            "No 'routes' list found in JSON. "
            "It looks like the router didn't return routable geometry."
        )

    kicad_lines = []
    skipped_failed = 0
    skipped_empty = 0

    for route in data["routes"]:
        # Skip any failed routes entirely
        if route.get("failed"):
            skipped_failed += 1
            continue

        net_name = route.get("net", "<unknown>")
        net_id = route.get("net_id")

        # KiCad needs a numeric net id
        if net_id is None:
            # If this route has nothing to place, just skip it quietly
            if not route.get("segments") and not route.get("vias"):
                skipped_empty += 1
                continue
            raise RuntimeError(f"Route for net '{net_name}' is missing 'net_id'.")

        segs = route.get("segments", []) or []
        vias = route.get("vias", []) or []

        # If no geometry for this net, skip it
        if not segs and not vias:
            skipped_empty += 1
            continue

        # Segments
        for seg in segs:
            try:
                (start_x, start_y), (end_x, end_y) = _pick_seg_points(seg)
            except KeyError as e:
                raise RuntimeError(f"Bad segment format for net '{net_name}' (id {net_id}): {e}")
            width = seg.get("width", 0.25)
            layer = seg.get("layer")
            if not layer:
                raise RuntimeError(f"Segment for net '{net_name}' (id {net_id}) is missing 'layer'.")
            line = f'''  (segment
    (start {start_x} {start_y})
    (end {end_x} {end_y})
    (width {width})
    (layer "{layer}")
    (net {net_id})
    (uuid "{uuid.uuid4()}")
  )'''
            kicad_lines.append(line)

        # Vias
        for via in vias:
            at = via.get("at")
            if not at or len(at) != 2:
                raise RuntimeError(f"Via for net '{net_name}' (id {net_id}) missing 'at' [x,y].")
            at_x, at_y = at
            vsize = via.get("size", 0.8)
            vdrill = via.get("drill", 0.4)
            vfrom = via.get("from", "F.Cu")
            vto   = via.get("to",   "B.Cu")
            line = f'''  (via
    (at {at_x} {at_y})
    (size {vsize})
    (drill {vdrill})
    (layers "{vfrom}" "{vto}")
    (net {net_id})
    (uuid "{uuid.uuid4()}")
  )'''
            kicad_lines.append(line)

    if not kicad_lines:
        raise RuntimeError(
            "Nothing to inject: all routes were failed or empty. "
            "Ensure your router produced at least one successful segment/via."
        )

    # Read the existing PCB file
    with open(pcb_file, "r", encoding="utf-8") as f:
        pcb_data = f.read().rstrip()

    # Insert just before the final ')', else append and add ')'
    insert_ok = pcb_data.endswith(")")
    if insert_ok:
        pcb_body = pcb_data[:-1].rstrip()
        new_pcb  = pcb_body + "\n" + "\n".join(kicad_lines) + "\n)"
    else:
        new_pcb  = pcb_data + "\n" + "\n".join(kicad_lines) + "\n)"

    # Ensure we don't overwrite existing files or the input file
    _assert_not_same_file(pcb_file, out_file)
    safe_out = _unique_out_path(out_file)

    with open(safe_out, "w", encoding="utf-8") as f:
        f.write(new_pcb)

    total_routes = len(data["routes"])
    successful_routes = total_routes - skipped_failed - skipped_empty

    print("\n--- Routing Injection Summary ---")
    print(f"Total routes in JSON : {total_routes}")
    print(f"✔ Successful routes  : {successful_routes}")
    print(f"✖ Failed routes      : {skipped_failed}")
    print(f"⚠ Empty routes       : {skipped_empty}")
    print("-------------------------------")
    print(f"Injected {len(kicad_lines)} new items (segments + vias) into: {safe_out}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python inject_routes.py <output.json> <input.kicad_pcb> <output.kicad_pcb>")
        sys.exit(1)

    json_file = sys.argv[1]
    pcb_file  = sys.argv[2]
    out_file  = sys.argv[3]
    json_to_kicad(json_file, pcb_file, out_file)
