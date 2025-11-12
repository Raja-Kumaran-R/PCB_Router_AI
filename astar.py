import sys
import json
import heapq
import math


# -----------------------
# Grid class
# -----------------------

# Global rules knob (read-only inside helpers)
RULES = {}

class Grid:
    def __init__(self, boundary, layers, step):
        self.boundary = boundary
        self.layers = layers
        self.step = step
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        self.minx = min(xs)
        self.maxx = max(xs)
        self.miny = min(ys)
        self.maxy = max(ys)
        self.nx = int(math.ceil((self.maxx - self.minx) / step)) + 1
        self.ny = int(math.ceil((self.maxy - self.miny) / step)) + 1
        # occupancy: occ[layer_index][ix][iy]
        self.occ = [[[False for _ in range(self.ny)] for _ in range(self.nx)] for _ in layers]
        self.pad_mask = [[[False]*self.ny for _ in range(self.nx)] for _ in range(len(self.layers))]
        self.pad_owner = [[[set() for _ in range(self.ny)] for _ in range(self.nx)] for _ in self.layers]

    def in_bounds(self, ix, iy):
        return 0 <= ix < self.nx and 0 <= iy < self.ny

    def world_to_grid(self, x, y):
        return (int(round((x - self.minx) / self.step)),
                int(round((y - self.miny) / self.step)))

    def grid_to_world(self, ix, iy):
        return (self.minx + ix * self.step,
                self.miny + iy * self.step)

    def is_blocked(self, ix, iy, il):
        return self.occ[il][ix][iy]

    def set_block(self, ix, iy, il):
        self.occ[il][ix][iy] = True

    def clear_block(self, ix, iy, il):
        """Forcefully mark a grid cell as free (used temporarily for start/goal)."""
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            self.occ[il][ix][iy] = False

        # NEW — used by Fix A
    def open_circular(self, layer_index, x_mm, y_mm, radius_mm):
        """Force-clear a disk of radius_mm (mm) around (x_mm,y_mm) on the given layer."""
        cx, cy = self.world_to_grid(x_mm, y_mm)
        r_cells = int(math.ceil(radius_mm / self.step))
        r2 = radius_mm * radius_mm + 1e-12
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                if (dx * self.step) ** 2 + (dy * self.step) ** 2 > r2:
                    continue
                jx, jy = cx + dx, cy + dy
                if 0 <= jx < self.nx and 0 <= jy < self.ny:
                    self.occ[layer_index][jx][jy] = False


def _inside_rotated_rect(cx, cy, px, py, sx, sy, rot_deg):
    """Is (cx,cy) inside a rectangle of half-sizes (sx,sy) centered at (px,py)
       rotated by rot_deg (degrees, CCW)?"""
    theta = math.radians(rot_deg or 0.0)
    ct, st = math.cos(theta), math.sin(theta)
    dx, dy = cx - px, cy - py
    # rotate point by -theta into pad-local frame
    lx =  dx*ct + dy*st
    ly = -dx*st + dy*ct
    return (-sx <= lx <= sx) and (-sy <= ly <= sy)


def clear_pad_region(grid, pad, layer, pad_uid=None):
    shape = pad.get("shape", "circle")
    clearance = float(pad.get("clearance", 0.2))
    rot = float(pad.get("abs_rotation", pad.get("pad_rotation", 0.0)))
    uid = pad_uid if pad_uid is not None else pad.get("uid")

    def can_clear(ix, iy):
        owners = grid.pad_owner[layer][ix][iy]
        # allow clearing only cells owned by THIS pad
        return uid is not None and owners and (owners == {uid})

    if shape in ("roundrect", "rect", "oval"):
        hx = float(pad.get("size_x", 0.0)) / 2.0 + clearance
        hy = float(pad.get("size_y", 0.0)) / 2.0 + clearance
        b = max(hx, hy)
        ix0, iy0 = grid.world_to_grid(pad["x"] - b, pad["y"] - b)
        ix1, iy1 = grid.world_to_grid(pad["x"] + b, pad["y"] + b)
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                if not grid.in_bounds(ix, iy): continue
                cx, cy = grid.grid_to_world(ix, iy)
                if _inside_rotated_rect(cx, cy, pad["x"], pad["y"], hx, hy, rot):
                    if can_clear(ix, iy):
                        grid.clear_block(ix, iy, layer)
    else:
        if "radius" in pad:
            r = float(pad["radius"]) + clearance
        else:
        # use average of size_x and size_y if given, else fallback
            sx = float(pad.get("size_x", 0.0))
            sy = float(pad.get("size_y", 0.0))
            r = (max(sx, sy) / 2.0) + clearance

        ix0, iy0 = grid.world_to_grid(pad["x"] - r, pad["y"] - r)
        ix1, iy1 = grid.world_to_grid(pad["x"] + r, pad["y"] + r)
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                if not grid.in_bounds(ix, iy): continue
                cx, cy = grid.grid_to_world(ix, iy)
                if (cx - pad["x"])**2 + (cy - pad["y"])**2 <= r*r:
                    if can_clear(ix, iy):
                        grid.clear_block(ix, iy, layer)


def set_pad_region(grid, pad, layer, pad_uid=None):
    shape = pad.get("shape", "circle")
    base_clear = float(pad.get("clearance", 0.2))
    pad_extra  = float(RULES.get("pad_clearance_extra", 0.0))
    is_tht = (pad.get("layer") == "*.Cu")
    tht_extra  = float(RULES.get("tht_clearance_extra", 0.0))
    # keep SMD halo; THT halo is controlled by tht_clearance_extra (can be 0 for exact size)
    clearance = base_clear + pad_extra

    rot = float(pad.get("abs_rotation", pad.get("pad_rotation", 0.0)))
    uid = pad_uid if pad_uid is not None else pad.get("uid")

    def mark(ix, iy):
        grid.set_block(ix, iy, layer)
        grid.pad_mask[layer][ix][iy] = True
        if uid is not None:
            grid.pad_owner[layer][ix][iy].add(uid)

    if shape in ("roundrect", "rect", "oval"):
        hx = float(pad.get("size_x", 0.0)) / 2.0 + clearance
        hy = float(pad.get("size_y", 0.0)) / 2.0 + clearance
        b = max(hx, hy)
        ix0, iy0 = grid.world_to_grid(pad["x"] - b, pad["y"] - b)
        ix1, iy1 = grid.world_to_grid(pad["x"] + b, pad["y"] + b)
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                if not grid.in_bounds(ix, iy): continue
                cx, cy = grid.grid_to_world(ix, iy)
                if _inside_rotated_rect(cx, cy, pad["x"], pad["y"], hx, hy, rot):
                    mark(ix, iy)
    else:
        if "radius" in pad:
            r = float(pad["radius"]) + clearance
        else:
            # use average of size_x and size_y if given, else fallback
            sx = float(pad.get("size_x", 0.0))
            sy = float(pad.get("size_y", 0.0))
            r = (max(sx, sy) / 2.0) + clearance

        ix0, iy0 = grid.world_to_grid(pad["x"] - r, pad["y"] - r)
        ix1, iy1 = grid.world_to_grid(pad["x"] + r, pad["y"] + r)
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                if not grid.in_bounds(ix, iy): continue
                cx, cy = grid.grid_to_world(ix, iy)
                if (cx - pad["x"])**2 + (cy - pad["y"])**2 <= r*r:
                    mark(ix, iy)




# -----------------------
# Helper functions
# -----------------------

def _via_clear_every_layer(grid, ix, iy, radius_mm):
    for L in range(len(grid.layers)):
        if grid.is_blocked(ix, iy, L):
            return False
        if not _disk_is_free(grid, ix, iy, L, radius_mm):
            return False
    return True


def _near_any_pad(grid, ix, iy, margin_cells, il):
    nx, ny = grid.nx, grid.ny
    pm = grid.pad_mask[il]                     # layer slice
    for dx in range(-margin_cells, margin_cells + 1):
        for dy in range(-margin_cells, margin_cells + 1):
            jx, jy = ix + dx, iy + dy
            if 0 <= jx < nx and 0 <= jy < ny and pm[jx][jy]:
                return True
    return False


def _is_safe_free_cell(grid, ix, iy, il, pad_margin_cells):
    if not grid.in_bounds(ix, iy):
        return False
    if grid.is_blocked(ix, iy, il):
        return False
    # keep one-cell (or given) clearance from pad_mask so we don't graze a neighboring pad
    if _near_any_pad(grid, ix, iy, pad_margin_cells, il):
        return False
    return True


def _goal_entry_candidates(grid, pad, layer_index):
    """
    Return up to 8 candidate grid cells just *outside* the goal pad,
    one per compass direction, that are free on the given layer.
    """
    if not pad:
        return []

    # ring radius just outside the copper
    shape = pad.get("shape", "circle")
    if shape in ("rect", "roundrect", "oval"):
        hx = float(pad.get("size_x", 0.0)) / 2.0
        hy = float(pad.get("size_y", 0.0)) / 2.0
        r  = max(hx, hy) + grid.step * 1.2
    else:
        sx = float(pad.get("size_x", 0.0))
        sy = float(pad.get("size_y", 0.0))
        r = (max(sx, sy) / 2.0) + grid.step * 1.2

    x0, y0 = float(pad["x"]), float(pad["y"])
    dirs = [

        # ring 1
        (1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),
        # ring 2
        (2,0),(2,2),(0,2),(-2,2),(-2,0),(-2,-2),(0,-2),(2,-2)
    ]

    cand = []
    for dx, dy in dirs:
        wx, wy = x0 + dx * r, y0 + dy * r
        ix, iy = grid.world_to_grid(wx, wy)
        if not grid.in_bounds(ix, iy):
            continue
        if not grid.is_blocked(ix, iy, layer_index):
            cand.append((ix, iy))

    # de-dup cells
    seen, out = set(), []
    for t in cand:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _normalize_layer_name(name, grid_layers, rules_net=None):
    # Treat KiCad "*.Cu" as "pick a practical copper layer"
    if name == "*.Cu":
        # Prefer F.Cu, then B.Cu, else first copper layer
        for pref in ("F.Cu", "B.Cu"):
            if pref in grid_layers:
                return pref
        return grid_layers[0]

    if name in grid_layers:
        return name

    allowed = (rules_net or {}).get("allowed_layers") or grid_layers
    for pref in ("F.Cu", "B.Cu"):
        if pref in allowed:
            return pref
    return allowed[0]


def rules_for_net(global_rules, net_name, net_id):
    """Return (merged_rules, class_name) for this net."""
    classes = global_rules.get("netclasses", {}) or {}
    mapping = global_rules.get("net_to_class", {}) or {}
    cls_name = mapping.get(net_name) or mapping.get(str(net_id)) or global_rules.get("default_netclass", "Default")
    overrides = classes.get(cls_name, {}) or {}

    # start from base rules but drop the big maps to avoid re-injecting them
    base = {k: v for k, v in global_rules.items() if k not in ("netclasses","net_to_class","default_netclass")}
    merged = {**base, **overrides}
    return merged, cls_name


def _block_disk(grid, ix, iy, il, radius_mm):
    """Block a disk of given radius (mm) around (ix,iy,layer)."""
    r_cells = int(math.ceil(radius_mm / grid.step))
    for dx in range(-r_cells, r_cells + 1):
        for dy in range(-r_cells, r_cells + 1):
            jx, jy = ix + dx, iy + dy
            if not grid.in_bounds(jx, jy):
                continue
            # distance in mm using grid.step
            if (dx * grid.step) ** 2 + (dy * grid.step) ** 2 <= radius_mm ** 2 + 1e-12:
                grid.set_block(jx, jy, il)

def _disk_is_free(grid, ix, iy, il, radius_mm):
    """
    True iff every grid cell within 'radius_mm' of (ix,iy) on layer 'il'
    is free of obstacles/pad copper.
    """
    r_cells = int(math.ceil(radius_mm / grid.step))
    r2 = radius_mm * radius_mm + 1e-12
    for dx in range(-r_cells, r_cells + 1):
        for dy in range(-r_cells, r_cells + 1):
            if (dx * grid.step) ** 2 + (dy * grid.step) ** 2 > r2:
                continue
            jx, jy = ix + dx, iy + dy
            if not grid.in_bounds(jx, jy):
                return False
            if grid.occ[il][jx][jy]:
                return False
    return True


def block_path_as_obstacles(grid, rules, path_cells, vias_raw):
    """
    After a net is routed, mark its copper (trace & vias) as obstacles for later nets.
    Uses trace_width/2 + clearance as the blocking radius (plus optional extra).
    """
    trace_w   = float(rules.get("trace_width", 0.25))
    clearance = float(rules.get("clearance", 0.2))
    extra_t   = float(rules.get("trace_clearance_extra", 0.0))
    r_trace   = 0.5 * trace_w + clearance + extra_t

    for ix, iy, il in path_cells:
        _block_disk(grid, ix, iy, il, r_trace)

    via_size  = float(rules.get("via_size", 0.6))
    extra_v   = float(rules.get("via_clearance_extra", 0.0))  # optional knob
    r_via     = 0.5 * via_size + clearance + extra_v
    for v in vias_raw:
        vix, viy = grid.world_to_grid(v["x"], v["y"])
        for il in range(len(grid.layers)):
            _block_disk(grid, vix, viy, il, r_via)



def find_pad_for_point(obstacles, x, y, layer_name, tol=0.8, net_id=None):
    best = None; bestd = 1e9
    for o in obstacles:
        if o.get("type") != "pad":
            continue
        if net_id is not None and o.get("net_id") != net_id:
            continue
        pad_layer = o.get("layer")
        if pad_layer and pad_layer not in (layer_name, "*.Cu"):
            continue
        d = abs(float(o["x"])-float(x)) + abs(float(o["y"])-float(y))
        if d < bestd and d <= tol:
            best, bestd = o, d
    return best



def clear_full_pad_access(grid, pad, layer_index, rules):
    if not pad:
        return
    base_clear   = float(pad.get("clearance", float(rules.get("clearance", 0.2))))
    trace_w      = float(rules.get("trace_width", 0.25))
    access_extra = float(rules.get("pad_clearance_extra", 0.5 * trace_w))
    access_extra *= 1.5

    # Default: SMD behavior
    open_clear = base_clear + access_extra

    # NEW: tiny poke-through ring to guarantee at least one free neighbor cell
    poke = 0.5 * grid.step  # half a cell
    grid.open_circular(layer_index, pad["x"], pad["y"], open_clear + poke)

    # THT pads (*.Cu): match rasterization model
    if pad.get("layer") == "*.Cu":
        pad_extra = float(rules.get("pad_clearance_extra", 0.0))
        tht_extra = float(rules.get("tht_clearance_extra", 0.0))
        open_clear = base_clear + access_extra

    pad_copy = dict(pad)
    pad_copy["clearance"] = open_clear
    uid = pad.get("uid")

    clear_pad_region(grid, pad_copy, layer_index, pad_uid=uid)




def compute_coords_extent(obstacles, tasks):
    xs, ys = [], []
    for obs in obstacles:
        if obs.get("type") == "pad":
            xs.append(obs["x"])
            ys.append(obs["y"])
        else:
            for x, y in obs.get("polygon", []):
                xs.append(x)
                ys.append(y)
    for task in tasks:
        if "start" in task:
            xs.append(task["start"]["x"])
            ys.append(task["start"]["y"])
        if "goal" in task:
            xs.append(task["goal"]["x"])
            ys.append(task["goal"]["y"])
    if not xs:
        return 0, 0, 0, 0, []
    return min(xs), max(xs), min(ys), max(ys), list(zip(xs, ys))


def expand_boundary_to_include(boundary, minx, maxx, miny, maxy, margin):
    if not boundary:
        return [(minx - margin, miny - margin),
                (maxx + margin, miny - margin),
                (maxx + margin, maxy + margin),
                (minx - margin, maxy + margin)]
    return boundary


def rasterize_obstacles(grid, rules, obstacles):
    """
    Pads (SMD + THT) and polygons -> grid blocks.
    Rect/roundrect/oval use a rotated-rectangle test (honors pad rotation).
    THT pads ("*.Cu") get an extra halo: rules["tht_clearance_extra"].
    """
    clearance_default = float(rules.get("clearance", 0.2))
    pad_extra         = float(rules.get("pad_clearance_extra", 0.0))
    tht_extra         = float(rules.get("tht_clearance_extra", 0.0))

    for obs in obstacles:
        if obs.get("type") == "pad":
            # stable UID per pad for this run
            if "uid" not in obs:
                layer_nm = obs.get("layer", "")
                obs["uid"] = f'{obs.get("net_id","-")}:{obs.get("pad_name","?")}@{float(obs["x"]):.4f},{float(obs["y"]):.4f}:{layer_nm or "*"}'
            uid = obs["uid"]

            x = float(obs["x"]); y = float(obs["y"])
            shape = obs.get("shape", "circle")
            rot   = float(obs.get("abs_rotation", obs.get("pad_rotation", 0.0)))
            layer_name = obs.get("layer")

            # Which layers this pad blocks
            if layer_name == "*.Cu":
                layers = grid.layers[:]          # THT: all copper layers
            elif layer_name:
                layers = [layer_name]
            else:
                layers = grid.layers[:]

            base_clear = float(obs.get("clearance", clearance_default))
            clearance  = base_clear + pad_extra
            

            def mark(ix, iy, il):
                grid.set_block(ix, iy, il)
                grid.pad_mask[il][ix][iy] = True
                grid.pad_owner[il][ix][iy].add(uid)

            if shape in ("roundrect", "rect", "oval"):
                hx = float(obs.get("size_x", 0.0)) / 2.0 + clearance
                hy = float(obs.get("size_y", 0.0)) / 2.0 + clearance
                bb = max(hx, hy)
                ix0, iy0 = grid.world_to_grid(x - bb, y - bb)
                ix1, iy1 = grid.world_to_grid(x + bb, y + bb)
                for il, lname in enumerate(grid.layers):
                    if lname not in layers: continue
                    for ix in range(ix0, ix1 + 1):
                        for iy in range(iy0, iy1 + 1):
                            if not grid.in_bounds(ix, iy): continue
                            cx, cy = grid.grid_to_world(ix, iy)
                            if _inside_rotated_rect(cx, cy, x, y, hx, hy, rot):
                                mark(ix, iy, il)
            else:
                if "radius" in obs:
                    r = float(obs["radius"]) + clearance
                else:
                    # use average of size_x and size_y if given, else fallback
                    sx = float(obs.get("size_x", 0.0))
                    sy = float(obs.get("size_y", 0.0))
                    r = (max(sx, sy) / 2.0) + clearance
                ix0, iy0 = grid.world_to_grid(x - r, y - r)
                ix1, iy1 = grid.world_to_grid(x + r, y + r)
                for il, lname in enumerate(grid.layers):
                    if lname not in layers: continue
                    for ix in range(ix0, ix1 + 1):
                        for iy in range(iy0, iy1 + 1):
                            if not grid.in_bounds(ix, iy): continue
                            cx, cy = grid.grid_to_world(ix, iy)
                            if (cx - x)**2 + (cy - y)**2 <= r*r:
                                mark(ix, iy, il)

        elif obs.get("polygon"):
            poly = obs["polygon"]
            xs = [px for px,_ in poly]; ys = [py for _,py in poly]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            clearance = float(obs.get("clearance", clearance_default))
            grow = int(math.ceil(clearance / grid.step))
            ix0, iy0 = grid.world_to_grid(minx, miny)
            ix1, iy1 = grid.world_to_grid(maxx, maxy)
            layers = obs.get("layers", grid.layers)
            for il, lname in enumerate(grid.layers):
                if lname not in layers:
                    continue
                if "Courtyard" in lname or "User." in lname:
                    continue
                for ix in range(ix0 - grow, ix1 + grow + 1):
                    for iy in range(iy0 - grow, iy1 + grow + 1):
                        if grid.in_bounds(ix, iy):
                            grid.set_block(ix, iy, il)
                for ix in range(ix0 - grow, ix1 + grow + 1):
                    for iy in range(iy0 - grow, iy1 + grow + 1):
                        if grid.in_bounds(ix, iy):
                            grid.set_block(ix, iy, il)

def enforce_board_edge_clearance(grid, rules):
    """
    Block all grid cells within 'board_clearance' distance from the board edge.
    """
    board_clearance = float(rules.get("board_clearance", 0.5))  # mm
    if board_clearance <= 0:
        return

    r_cells = int(math.ceil(board_clearance / grid.step))
    for il in range(len(grid.layers)):
        for ix in range(grid.nx):
            for iy in range(grid.ny):
                if (ix < r_cells or iy < r_cells or
                    ix >= grid.nx - r_cells or iy >= grid.ny - r_cells):
                    grid.set_block(ix, iy, il)


def octile(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)


def compress_collinear(path, ang_eps_deg=5.0):
    if not path:
        return []
    out = [path[0]]
    for i in range(1, len(path) - 1):
        x0, y0, l0 = out[-1]
        x1, y1, l1 = path[i]
        x2, y2, l2 = path[i + 1]
        dx1, dy1 = x1 - x0, y1 - y0
        dx2, dy2 = x2 - x1, y2 - y1
        if l0 == l1 == l2 and (dx1 or dy1) and (dx2 or dy2):
            if dx1 * dy2 == dy1 * dx2:
                continue
            a1 = math.atan2(dy1, dx1)
            a2 = math.atan2(dy2, dx2)
            da = (a2 - a1 + math.pi) % (2*math.pi) - math.pi
            if abs(da) < math.radians(ang_eps_deg):
                continue
        out.append((x1, y1, l1))
    out.append(path[-1])
    return out



# -----------------------
# A* search helper (pad-exit sampling + gentle via rules)
# -----------------------

def _ordered_free_exits(grid, s_ix, s_iy, sL, g_ix, g_iy, pad_margin_cells=1):
    # 8 neighbors around the start, sorted by how they face the goal
    neigh = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
    cand = []
    for dx, dy in neigh:
        jx, jy = s_ix + dx, s_iy + dy
        if _is_safe_free_cell(grid, jx, jy, sL, pad_margin_cells):
            # prioritize neighbors that point toward the goal
            cost = abs((jx - g_ix)) + abs((jy - g_iy))
            cand.append((cost, jx, jy))
    cand.sort(key=lambda t: t[0])
    return [(jx, jy) for _, jx, jy in cand]

def _ordered_free_entries(grid, g_ix, g_iy, gL, s_ix, s_iy, pad_margin_cells=1):
    # 8 neighbors around the goal, sorted by how they face the start
    neigh = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
    cand = []
    for dx, dy in neigh:
        jx, jy = g_ix + dx, g_iy + dy
        if _is_safe_free_cell(grid, jx, jy, gL, pad_margin_cells):
            # prioritize neighbors that open toward the incoming direction (from start)
            cost = abs((jx - s_ix)) + abs((jy - s_iy))
            cand.append((cost, jx, jy))
    cand.sort(key=lambda t: t[0])
    return [(jx, jy) for _, jx, jy in cand]



def astar_search(grid, rules, start_state, goal_state, allow_via, start_xy, goal_xy, start_pad_xy=None, goal_pad_xy=None, **_ignored):
    start_ix, start_iy, start_layer = start_state
    goal_ix, goal_iy, goal_layer = goal_state
    pad_margin_cells = int(rules.get("pad_margin_cells", 1))
    via_cost = float(rules.get("via_cost", 10.0))
    min_via_from_pads = float(rules.get("min_via_from_pads", 0.6))

    via_size  = float(rules.get("via_size", 0.6))
    clearance = float(rules.get("clearance", 0.2))
    extra_v   = float(rules.get("via_clearance_extra", 0.0))
    r_via     = 0.5 * via_size + clearance + extra_v

    # Use pad centers for via distance checks if provided; otherwise fall back
    if start_pad_xy is not None:
        sx, sy = float(start_pad_xy[0]), float(start_pad_xy[1])
    else:
        sx, sy = float(start_xy[0]), float(start_xy[1])

    if goal_pad_xy is not None:
        gx, gy = float(goal_pad_xy[0]), float(goal_pad_xy[1])
    else:
        gx, gy = float(goal_xy[0]), float(goal_xy[1])

    neighbors = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
                 (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2))]

    allowed_layers = rules.get("allowed_layers")
    allowed_via_layers = None
    if allowed_layers:
        allowed_via_layers = {i for i, name in enumerate(grid.layers) if name in allowed_layers}

    open_heap = []
    start_state_key = (start_ix, start_iy, start_layer)
    goal_state_key  = (goal_ix,  goal_iy,  goal_layer)
    g_score = {start_state_key: 0.0}
    parent  = {start_state_key: None}

    def heuristic(ix, iy, il):
        h = octile(ix, iy, goal_ix, goal_iy)
        if il != goal_layer:
            h += via_cost
        return h

    def push_state(state, gval):
        f = gval + heuristic(state[0], state[1], state[2])
        heapq.heappush(open_heap, (f, state))

    push_state(start_state_key, 0.0)
    EPS = 1e-9

    while open_heap:
        f, cur = heapq.heappop(open_heap)
        cur_g = g_score.get(cur, float('inf'))
        if cur_g + heuristic(cur[0], cur[1], cur[2]) + EPS < f:
            continue

        if cur == goal_state_key:
            path = []
            vias = []
            node = cur
            while node is not None:
                path.append(node)
                node = parent.get(node)
            path.reverse()
            for i in range(1, len(path)):
                if path[i][2] != path[i-1][2]:
                    x, y = grid.grid_to_world(path[i][0], path[i][1])
                    vias.append({"x": x, "y": y,
                                 "from": grid.layers[path[i-1][2]],
                                 "to":   grid.layers[path[i][2]]})
            return path, vias

        ix, iy, il = cur
        gcur = g_score.get(cur, float('inf'))

        # ---- planar neighbors on same layer ----
        best_neighbor_heuristic = float('inf')
        planar_moves_exist = False
        for dx, dy, dcost in neighbors:
            jx, jy = ix + dx, iy + dy
            if not grid.in_bounds(jx, jy):
                continue
            if grid.is_blocked(jx, jy, il):
                continue
            planar_moves_exist = True
            h_neighbor = octile(jx, jy, goal_ix, goal_iy)
            if h_neighbor < best_neighbor_heuristic:
                best_neighbor_heuristic = h_neighbor
            cand = (jx, jy, il)
            tg = gcur + dcost
            if tg + EPS < g_score.get(cand, float('inf')):
                parent[cand] = cur
                g_score[cand] = tg
                push_state(cand, tg)

        # ---- via policy (stall-or-none + tiny lookahead epsilon) ----
        # Allow a via if planar moves don't improve *meaningfully*.
        # The epsilon is configurable (rules["via_stall_epsilon"], default 0.2 "cells").
        stall_eps = float(rules.get("via_stall_epsilon", 0.2))

        allow_via_here = False
        current_h = octile(ix, iy, goal_ix, goal_iy)

        if not planar_moves_exist:
            allow_via_here = True
        else:
            # Treat tiny heuristic gains as “no real progress”
            if best_neighbor_heuristic >= current_h - stall_eps:
                allow_via_here = True
            else:
                # If we’re not yet on the goal layer, permit an early hop when the planar
                # improvement is trivial; keeps time cost O(1), no neighbor scans.
                if il != goal_layer:
                    if (allowed_via_layers is None) or (goal_layer in allowed_via_layers):
                        if (current_h - best_neighbor_heuristic) < 1.0:
                            allow_via_here = True


        # ---- propose via (distance + disk free on all layers) ----
        if allow_via:
            cx, cy = grid.grid_to_world(ix, iy)
            d_start = math.hypot(cx - sx, cy - sy)
            d_goal  = math.hypot(cx - gx, cy - gy)
            too_close = (min(d_start, d_goal) < min_via_from_pads)

            # Base gating: if we can still make planar progress AND the via is inside the user buffer,
            # skip the via for now. If we are stalled (allow_via_here == True), we may allow it.
            allow_drop = (allow_via_here or not too_close)

            if allow_drop:
                for jl in range(len(grid.layers)):
                    if jl == il:
                        continue
                    if allowed_via_layers is not None and jl not in allowed_via_layers:
                        continue
                    # avoid dropping a via exactly on the goal cell
                    if (ix == goal_ix and iy == goal_iy):
                        continue
                    # Real DRC: via disk must be clear on ALL layers
                    if not _via_clear_every_layer(grid, ix, iy, r_via):
                        continue

                    cand = (ix, iy, jl)

                    # ----------------- COST MODEL -----------------
                    # Start with base via cost (do NOT mutate via_cost)
                    via_penalty = via_cost

                    # (A) If we're very close to a pad AND choked (<=2 safe planar exits), give a small discount
                    # to encourage an early escape; uses pad_margin_cells you defined earlier.
                    near_start = (abs(ix - start_state[0]) <= 1 and abs(iy - start_state[1]) <= 1)
                    near_goal  = (abs(ix - goal_state[0])  <= 1 and abs(iy - goal_state[1])  <= 1)
                    if near_start or near_goal:
                        free_planar = 0
                        for dx, dy, _dc in neighbors:
                            jx, jy = ix + dx, iy + dy
                            if grid.in_bounds(jx, jy) and _is_safe_free_cell(grid, jx, jy, il, pad_margin_cells):
                                free_planar += 1
                        if free_planar <= 2:
                            via_penalty = max(1.0, via_penalty * 0.5)

                    # (B) If the via is inside the user’s min distance, keep it **possible only when stalled**,
                    # but discourage it with a small surcharge so the router prefers a farther legal drop.
                    if too_close and allow_via_here:
                        via_penalty *= 1.6  # tune 1.4–2.0 if needed

                    tg = gcur + via_penalty
                    if tg + EPS < g_score.get(cand, float('inf')):
                        parent[cand] = cur
                        g_score[cand] = tg
                        push_state(cand, tg)

    raise RuntimeError("No route found")


def _path_cost(grid, rules, path):
    """A* cost used for comparison: unit/sqrt2 per grid move + via_cost per layer change."""
    if not path or len(path) < 2:
        return 0.0
    via_cost = float(rules.get("via_cost", 10.0))
    cost = 0.0
    for i in range(1, len(path)):
        (ix0, iy0, il0) = path[i-1]
        (ix1, iy1, il1) = path[i]
        if il1 != il0:
            cost += via_cost
        else:
            dx = abs(ix1 - ix0)
            dy = abs(iy1 - iy0)
            # 4/8-way grid: each step is 1 or sqrt(2)
            if dx == 1 and dy == 1:
                cost += math.sqrt(2)
            elif (dx == 1 and dy == 0) or (dx == 0 and dy == 1):
                cost += 1.0
            else:
                cost += math.hypot(dx, dy)
    return cost


def astar_route(grid, rules, start, goal):
    if start is None or goal is None:
        raise RuntimeError("Missing start or goal in astar_route")

    start_ix, start_iy = grid.world_to_grid(start["x"], start["y"])
    goal_ix,  goal_iy  = grid.world_to_grid(goal["x"],  goal["y"])

    if not grid.in_bounds(start_ix, start_iy) or not grid.in_bounds(goal_ix, goal_iy):
        raise RuntimeError("Start or goal out of grid bounds in astar_route")

    s_name = _normalize_layer_name(start.get("layer", grid.layers[0]), grid.layers, rules)
    g_name = _normalize_layer_name(goal.get("layer",  grid.layers[0]), grid.layers, rules)

    start_layer = grid.layers.index(s_name)
    goal_layer  = grid.layers.index(g_name)
    start_state = (start_ix, start_iy, start_layer)
    goal_state  = (goal_ix,  goal_iy,  goal_layer)

    # ---------- GOAL ENTRY SWEEP ----------
    goal_pad = goal.get("_pad_obj")
    goal_candidates = []
    if goal_pad:
        goal_candidates = _goal_entry_candidates(grid, goal_pad, goal_layer)
        if (goal_ix, goal_iy) not in goal_candidates:
            goal_candidates.append((goal_ix, goal_iy))
    else:
        goal_candidates = [(goal_ix, goal_iy)]
    # --------------------------------------

    def run_attempt(allow_via):
        """
        If we have multiple goal approach cells, do a two-phase route:
          start -> candidate, then candidate -> real goal (planar).
        Otherwise, do the single A* you already had.
        Return best (cost, path, vias) or raise if none.
        """
        best = None

        for (gix_c, giy_c) in goal_candidates:
            cand_goal_state = (gix_c, giy_c, goal_layer)

            try:
                # Phase A: start -> candidate (vias allowed per attempt)
                pathA, viasA = astar_search(
                    grid, rules, start_state, cand_goal_state,
                    allow_via=allow_via,
                    start_xy=(start["x"], start["y"]),
                    goal_xy=grid.grid_to_world(gix_c, giy_c),   # candidate entry
                    start_pad_xy=(start["x"], start["y"]),      # true start pad center
                    goal_pad_xy=(goal["x"], goal["y"])          # true goal pad center
                )

                # If candidate is the actual goal cell, we're done.
                if (gix_c, giy_c) == (goal_ix, goal_iy):
                    cost = _path_cost(grid, rules, pathA)
                    tup  = (cost, pathA, viasA)
                else:
                    # Phase B: candidate -> real goal (planar, short hop)
                    pathB, viasB = astar_search(
                        grid, rules, cand_goal_state, goal_state,
                        allow_via=False,
                        start_xy=grid.grid_to_world(gix_c, giy_c),
                        goal_xy=(goal["x"], goal["y"]),
                        start_pad_xy=(start["x"], start["y"]),
                        goal_pad_xy=(goal["x"], goal["y"])
                    )
                    stitched_path = pathA + pathB[1:]
                    stitched_vias = viasA + viasB
                    cost = _path_cost(grid, rules, stitched_path)
                    tup  = (cost, stitched_path, stitched_vias)

                if (best is None) or (tup[0] + 1e-9 < best[0]):
                    best = tup

            except RuntimeError:
                continue

        if best is None:
            raise RuntimeError("No route found")
        return best

    # Try planar, then vias
    best = None
    try:
        best = run_attempt(allow_via=False)
    except RuntimeError:
        pass

    try:
        cand = run_attempt(allow_via=True)
        if (best is None) or (cand[0] + 1e-9 < best[0]):
            best = cand
    except RuntimeError:
        pass

    if best is None:
        raise RuntimeError("No route found")

    _, path, vias = best
    return path, vias





# -----------------------
# (Old) DRC Engine (kept, but unused)
# -----------------------

def _local_to_world(lx, ly, px, py, rot_deg):
    t = math.radians(rot_deg or 0.0)
    ct, st = math.cos(t), math.sin(t)
    wx = px + lx*ct - ly*st
    wy = py + lx*st + ly*ct
    return wx, wy

def _dot(ax, ay, bx, by):
    return ax*bx + ay*by

def _clamp01(t):
    return 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)

def _seg_seg_distance(p1, p2, q1, q2):
    x1,y1 = p1; x2,y2 = p2
    x3,y3 = q1; x4,y4 = q2
    ux, uy = x2-x1, y2-y1
    vx, vy = x4-x3, y4-y3
    wx, wy = x1-x3, y1-y3
    a = _dot(ux,uy,ux,uy)
    b = _dot(ux,uy,vx,vy)
    c = _dot(vx,vy,vx,vy)
    d = _dot(ux,uy,wx,wy)
    e = _dot(vx,vy,wx,wy)
    D = a*c - b*b
    sc, sN, sD = 0.0, D, D
    tc, tN, tD = 0.0, D, D

    if D < 1e-12:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b*e - c*d)
        tN = (a*e - b*d)
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a

    sc = 0.0 if abs(sN) < 1e-12 else (sN / sD)
    tc = 0.0 if abs(tN) < 1e-12 else (tN / tD)

    cx1 = x1 + sc*ux; cy1 = y1 + sc*uy
    cx2 = x3 + tc*vx; cy2 = y3 + tc*vy
    dx = cx1 - cx2; dy = cy1 - cy2
    dist = math.hypot(dx, dy)

    intersects = (dist < 1e-6) and (0.0 <= sc <= 1.0) and (0.0 <= tc <= 1.0)
    return dist, (cx1, cy1), (cx2, cy2), intersects

def _pt_seg_distance(pt, a, b):
    x,y = pt; x1,y1 = a; x2,y2 = b
    vx,vy = x2-x1,y2-y1
    if abs(vx) < 1e-12 and abs(vy) < 1e-12:
        return math.hypot(x-x1,y-y1), (x1,y1)
    t = _clamp01(((x-x1)*vx + (y-y1)*vy) / (vx*vx + vy*vy))
    cx,cy = x1 + t*vx, y1 + t*vy
    return math.hypot(x-cx, y-cy), (cx,cy)

def _dist_seg_to_circle(seg_a, seg_b, cx, cy, radius):
    d, cp = _pt_seg_distance((cx,cy), seg_a, seg_b)
    return d - radius, cp

def _rotate_to_local(x, y, px, py, rot_deg):
    t = math.radians(rot_deg or 0.0)
    ct, st = math.cos(t), math.sin(t)
    dx, dy = x - px, y - py
    lx =  dx*ct + dy*st
    ly = -dx*st + dy*ct
    return lx, ly

def _seg_rect_distance_local(a, b, hx, hy):
    for px,py in (a,b):
        if (-hx <= px <= hx) and (-hy <= py <= hy):
            d = -min(hx - abs(px), hy - abs(py))
            return d, (px,py)
    edges = [
        ((-hx,-hy), ( hx,-hy)),
        (( hx,-hy), ( hx, hy)),
        (( hx, hy), (-hx, hy)),
        ((-hx, hy), (-hx,-hy)),
    ]
    best_d = 1e9
    best_cp = None
    for e1, e2 in edges:
        d, cp1, cp2, _ = _seg_seg_distance(a, b, e1, e2)
        if d < best_d:
            best_d = d
            best_cp = cp1
    return d, best_cp

def _track_pad_clearance(seg_a, seg_b, width, pad):
    shape = pad.get("shape","rect")
    rot = float(pad.get("abs_rotation", pad.get("pad_rotation", 0.0)))
    px, py = pad["x"], pad["y"]

    if shape in ("rect","roundrect"):
        hx = float(pad.get("size_x",0.0))/2.0
        hy = float(pad.get("size_y",0.0))/2.0
        a_loc = _rotate_to_local(seg_a[0], seg_a[1], px, py, rot)
        b_loc = _rotate_to_local(seg_b[0], seg_b[1], px, py, rot)
        d, cp = _seg_rect_distance_local(a_loc, b_loc, hx, hy)
        cp_world = _local_to_world(cp[0], cp[1], px, py, rot)
        return d - (width/2.0), cp_world
    else:
        sx = float(pad.get("size_x", 0.0))
        sy = float(pad.get("size_y", 0.0))
        r = (max(sx, sy) / 2.0)
        d, cp = _dist_seg_to_circle(seg_a, seg_b, px, py, r)
        cp_world = _local_to_world(cp[0], cp[1], px, py, rot)
        return d - (width/2.0), cp_world

def _via_pad_clearance(cx, cy, via_diam, pad):
    shape = pad.get("shape","rect")
    rot = float(pad.get("abs_rotation", pad.get("pad_rotation", 0.0)))
    px, py = pad["x"], pad["y"]

    if shape in ("rect","roundrect"):
        hx = float(pad.get("size_x",0.0))/2.0
        hy = float(pad.get("size_y",0.0))/2.0
        v_loc = _rotate_to_local(cx, cy, px, py, rot)
        x, y = v_loc
        dx = max(abs(x) - hx, 0.0)
        dy = max(abs(y) - hy, 0.0)
        if dx == 0.0 and dy == 0.0:
            dist = -min(hx - abs(x), hy - abs(y))
        else:
            dist = math.hypot(dx, dy)
        return dist - (via_diam/2.0), (cx,cy)
    else:
        sx = float(pad.get("size_x", 0.0))
        sy = float(pad.get("size_y", 0.0))
        r = (max(sx, sy) / 2.0)
        dist = math.hypot(cx - px, cy - py) - r
        return dist - (via_diam/2.0), (cx,cy)



# -----------------------
# Summary helper (for the web API)
# -----------------------

def _compute_summary(routes):
    segs = 0
    vias = 0
    nets = set()
    length = 0.0

    for r in routes:
        if r.get("failed"):
            continue
        nets.add((r.get("net_id"), r.get("net")))
        segs += len(r.get("segments", []))
        vias += len(r.get("vias", []))
        for s in r.get("segments", []):
            (x0, y0) = s["start"]
            (x1, y1) = s["end"]
            length += math.hypot(x1 - x0, y1 - y0)

    return {
        "segments": segs,
        "vias": vias,
        "nets_routed": len(nets),
        "total_length_mm": round(length, 2)
    }


# -----------------------
# Routing manager
# -----------------------

def route_all(input_data):
    board = input_data.get("board", {})
    rules = input_data.get("rules", {})
    # make rules available to helpers
    global RULES
    RULES = rules
    obstacles = input_data.get("obstacles", [])
    tasks = input_data.get("tasks", [])
    # small preference to route VCC early if desired
    # ---- NEW: user-controlled routing order by netclass ----
    tasks.sort(key=lambda t: 0 if t.get("net", "").upper() == "VCC" else 1)

    boundary = board.get("boundary", []) or []
    layers = board.get("layers", [])
    step = float(rules.get("grid_step", 0.1))

    minx, maxx, miny, maxy, coords = compute_coords_extent(obstacles, tasks)
    margin = max(step * 10, 5.0)
    boundary_expanded = expand_boundary_to_include(boundary, minx, maxx, miny, maxy, margin)

    grid = Grid(boundary_expanded, layers, step)
    rasterize_obstacles(grid, rules, obstacles)
    enforce_board_edge_clearance(grid, rules)

    # small helper: open a circular window (radius in mm) as free space on a given layer index
    def _open_disk(ix, iy, il, radius_mm):
        r_cells = int(math.ceil(radius_mm / grid.step))
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                if (dx * grid.step) ** 2 + (dy * grid.step) ** 2 <= radius_mm ** 2 + 1e-12:
                    jx, jy = ix + dx, iy + dy
                    if grid.in_bounds(jx, jy):
                        grid.clear_block(jx, jy, il)

    routes = []
    for task in tasks:
        net = task.get("net")
        net_id = task.get("net_id")
        start = task.get("start")
        goal  = task.get("goal")
        if not start or not goal:
            routes.append({"net": net, "net_id": net_id, "failed": True, "reason": "Missing start or goal"})
            continue

        rules_net, cls_name = rules_for_net(rules, net, net_id)

        # Candidate layers for start/goal (handle "*.Cu")
        def _cand_layers(layer_token):
            if layer_token == "*.Cu":
                return list(grid.layers)  # try all copper layers
            return [_normalize_layer_name(layer_token, grid.layers, rules_net)]

        start_layer_names = _cand_layers(start.get("layer", grid.layers[0]))
        goal_layer_names  = _cand_layers(goal.get("layer",  grid.layers[0]))

        # --- OPTIONAL: reuse an existing same-net via as an anchor/junction ---
        reuse_same_net_via = bool(rules_net.get("reuse_same_net_via", False))
        chosen_anchor = None
        anchor_layers = None
        if reuse_same_net_via and routes:
            candidates = []
            for rr in routes:
                if rr.get("failed") or rr.get("net_id") != net_id:
                    continue
                for v in rr.get("vias", []):
                    candidates.append(v)
            if candidates:
                sx, sy = start["x"], start["y"]
                def vd2(v):
                    vx, vy = float(v["at"][0]), float(v["at"][1])
                    return (vx - sx) ** 2 + (vy - sy) ** 2
                chosen_anchor = min(candidates, key=vd2)
                anchor_layers = (chosen_anchor.get("from"), chosen_anchor.get("to"))

        route_found = False
        last_error = None

        # Try all (startLayer, goalLayer) pairs until one succeeds
        for sL_name in start_layer_names:
            sL = grid.layers.index(sL_name)
            start_pad = find_pad_for_point(obstacles, start["x"], start["y"], sL_name, net_id=net_id)

            for gL_name in goal_layer_names:
                gL = grid.layers.index(gL_name)
                goal_pad = find_pad_for_point(obstacles, goal["x"], goal["y"], gL_name, net_id=net_id)

                # Open the pad copper (plus extra)
                if start_pad:
                    clear_full_pad_access(grid, start_pad, sL, rules_net)
                if goal_pad:
                    clear_full_pad_access(grid, goal_pad,  gL, rules_net)

                # If we have an anchor via, open a small window at its location and steer goal to it (on sL)
                opened_anchor = False
                goal_try = {"x": goal["x"], "y": goal["y"], "layer": gL_name}
                if chosen_anchor:
                    vx, vy = float(chosen_anchor["at"][0]), float(chosen_anchor["at"][1])
                    via_size = float(chosen_anchor.get("size", rules_net.get("via_size", 0.6)))
                    clearance = float(rules_net.get("clearance", 0.2))
                    open_r = 0.5 * via_size + clearance
                    vix, viy = grid.world_to_grid(vx, vy)

                    for lname in anchor_layers:
                        if lname in grid.layers:
                            il = grid.layers.index(lname)
                            _open_disk(vix, viy, il, open_r)

                    goal_try = {"x": vx, "y": vy, "layer": sL_name}
                    goal_try["_pad_obj"] = goal_pad  # harmless to keep for aesthetics
                    opened_anchor = True
                else:
                    goal_try["_pad_obj"] = goal_pad

                try:
                    # --- EXIT SWEEP (start) + ENTRY SWEEP (goal) ---
                    s_ix, s_iy = grid.world_to_grid(start["x"], start["y"])
                    g0_ix, g0_iy = grid.world_to_grid(goal_try["x"], goal_try["y"])

                    # pad-aware exit candidates (won't hug neighboring pad copper)
                    exit_list  = [(s_ix, s_iy)] + _ordered_free_exits(
                        grid, s_ix, s_iy, sL, g0_ix, g0_iy, pad_margin_cells=1
                    )
                    # pad-aware entry candidates (mirror idea on goal)
                    entry_list = [(g0_ix, g0_iy)] + _ordered_free_entries(
                        grid, g0_ix, g0_iy, gL, s_ix, s_iy, pad_margin_cells=1
                    )

                    last_err = None
                    found_local = False

                    # Try a small cross of exit×entry (cheapest first)
                    pairs = []
                    for ex_ix, ex_iy in exit_list:
                        for en_ix, en_iy in entry_list:
                            pri = (abs(ex_ix - g0_ix) + abs(ex_iy - g0_iy)
                                   + abs(en_ix - s_ix) + abs(en_iy - s_iy))
                            pairs.append((pri, (ex_ix, ex_iy), (en_ix, en_iy)))
                    pairs.sort(key=lambda t: t[0])

                    for _, (ex_ix, ex_iy), (en_ix, en_iy) in pairs:
                        ex_x, ex_y = grid.grid_to_world(ex_ix, ex_iy)
                        en_x, en_y = grid.grid_to_world(en_ix, en_iy)

                        start_try = {"x": ex_x, "y": ex_y, "layer": sL_name}
                        goal_variant = {"x": en_x, "y": en_y, "layer": goal_try["layer"]}
                        if goal_pad:
                            goal_variant["_pad_obj"] = goal_pad

                        # If we offset the goal (not using an anchor), append a tiny tail later
                        snap_tail_to_goal = not (en_ix == g0_ix and en_iy == g0_iy) and not opened_anchor

                        try:
                            path_cells, vias_raw = astar_route(grid, rules_net, start_try, goal_variant)

                            # Restore pad blocking on success (re-attach ownership)
                            def _restore_pad(pad, default_li):
                                if not pad:
                                    return
                                uid = pad.get("uid")
                                if pad.get("layer") == "*.Cu":
                                    for li in range(len(grid.layers)):
                                        set_pad_region(grid, pad, li, pad_uid=uid)
                                else:
                                    set_pad_region(grid, pad, default_li, pad_uid=uid)

                            _restore_pad(start_pad, sL)
                            _restore_pad(goal_pad,  gL)

                            # Freeze this route (tracks + vias) as obstacles
                            block_path_as_obstacles(grid, rules_net, path_cells, vias_raw)

                            # Convert path to world segments
                            world_path = []
                            for ix, iy, il in path_cells:
                                x, y = grid.grid_to_world(ix, iy)
                                world_path.append((round(x, 6), round(y, 6), grid.layers[il]))

                            # Append a one-cell tail into the true goal point if we used an entry offset
                            if snap_tail_to_goal:
                                gx_true, gy_true = goal["x"], goal["y"]
                                last_layer_name = world_path[-1][2]
                                world_path.append((round(gx_true, 6), round(gy_true, 6), last_layer_name))

                            simplified = compress_collinear(world_path)

                            segments = []
                            for i in range(len(simplified) - 1):
                                x0, y0, l0 = simplified[i]
                                x1, y1, l1 = simplified[i + 1]
                                if l0 == l1:
                                    segments.append({
                                        "start": [x0, y0],
                                        "end":   [x1, y1],
                                        "layer": l0,
                                        "width": float(rules_net.get("trace_width", 0.25))
                                    })

                            vias_out = []
                            for via in vias_raw:
                                vias_out.append({
                                    "at":   [round(via["x"], 6), round(via["y"], 6)],
                                    "from": via["from"],
                                    "to":   via["to"],
                                    "size": float(rules_net.get("via_size", 0.6)),
                                    "drill":float(rules_net.get("via_drill", 0.3))
                                })

                            routes.append({
                                "net": net,
                                "net_id": net_id,
                                "netclass": cls_name,
                                "segments": segments,
                                "vias": vias_out
                            })

                            route_found = True
                            found_local = True
                            break  # entry loop
                        except Exception as ee:
                            last_err = ee

                    if not found_local:
                        raise last_err if last_err else RuntimeError("No route from any exit/entry")

                    break  # gL loop on success

                except Exception as e:
                    last_error = str(e)
                    # Restore pad blocking before trying next pair
                    def _restore_pad_fail(pad, default_li):
                        if not pad:
                            return
                        if pad.get("layer") == "*.Cu":
                            for li in range(len(grid.layers)):
                                set_pad_region(grid, pad, li)
                        else:
                            set_pad_region(grid, pad, default_li)

                    _restore_pad_fail(start_pad, sL)
                    _restore_pad_fail(goal_pad,  gL)
                    # continue to next (sL, gL)

            if route_found:
                break  # sL loop

        if not route_found:
            routes.append({
                "net": net,
                "net_id": net_id,
                "failed": True,
                "reason": last_error or "No route found",
                "netclass": cls_name
            })

    # Web summary for UI, but keep file schema identical to your sample
    summary = _compute_summary(routes)
    return {"routes": routes, "summary": summary}



# -----------------------
# CLI
# -----------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python astar.py input.json [output.json]")
        sys.exit(1)
    with open(sys.argv[1], "r") as f:
        data = json.load(f)
    out = route_all(data)

    # If a filename is given, write ONLY the 'routes' (matches your sample output.json exactly)
    if len(sys.argv) >= 3:
        with open(sys.argv[2], "w") as f:
            json.dump({"routes": out["routes"]}, f, indent=2)
        s = out.get("summary", {})
        print(f"Routing finished. Results written to {sys.argv[2]}")
        if s:
            print(f"Summary: segments {s.get('segments', 0)} • vias {s.get('vias', 0)} • "
                  f"nets routed {s.get('nets_routed', 0)} • total length {s.get('total_length_mm', 0):.2f} mm")
    else:
        # stdout includes summary for convenience when used without file
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
