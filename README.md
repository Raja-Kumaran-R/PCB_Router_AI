# KiCad Autorouter with A*

This project implements a simple autorouter for KiCad PCB files using the A* search algorithm.  
It parses `.kicad_pcb` files, generates routes, and injects them back into the PCB design.

## Features
- Parse PCB layout files to extract pads, nets, and obstacles.
- Route nets automatically using A* algorithm.
- Support for via creation between layers.
- Generates new routed KiCad PCB file.

## Current Progress
1. Two sample PCB files are included:
   - `Simple_Demo.kicad_pcb`: Single-layer design (works perfectly).
   - `Simple_Via.kicad_pcb`: Modified design with a component on the back layer for via testing.
   - `tht.kicad_pcb`: It has a THT component to be work on.
2. Vias are created successfully.
3. Currently working on handling Through Hole Components.
3. Next step: extend support for **4-layer PCBs**.

## Usage

### Step 1: Parse the PCB File
```bash
python parser.py <file_name>.kicad_pcb
```
This will generate an `input.json` file containing the parsed layout information.

### Step 2: Run A* Routing
```bash
python astar.py input.json > output.json
```
This creates an `output.json` file containing the generated routing paths.

### Step 3: Inject Routes into PCB
```bash
python inject_routes.py output.json <filename>.kicad_pcb <filename>_routed.kicad_pcb
```
This produces a new routed PCB file `<filename>_routed.kicad_pcb`.

### Step 4: Open in KiCad
Open `<filename>_routed.kicad_pcb` in KiCad to check the generated routes.

## Notes
- Works correctly on single-layer boards.
- Vias are being generated and routing is done through vias.
- Currently tested on `Simple_Demo.kicad_pcb`, `Simple_Via.kicad_pcb`, `via2.kicac_pcb` and `via3.kicad_pcb`.(Working perfectly)
- Currently testing on `tht.kicad_pcb`.

## Next Steps
- Fix THT components routing.
- Extend functionality for multi-layer (4-layer) PCBs.
- Improve routing efficiency and optimization.

