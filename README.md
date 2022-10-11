# Smooth Distances

This repo contains the code used for the SIGGRAPH 2022 paper, "Fast Evaluation
of Smooth Distance Constraints on Co-Dimensional Geometry". It contains the
source code for the core library, as well as some applications used for sphere
traced visualizations and rigid body simulations. Please see the source code for
these applications (in the `app` folder) for examples of how to use the API.
This code has been tested on macOS and Linux.

[Project Page](https://www.dgp.toronto.edu/projects/smooth-distances/)

## Building
```bash
mkdir build
cd build
cmake ..
make
```

To download the data, please see the `v1.0.0` release on GitHub and follow the
instructions there.

## Running

Aside from the static libraries for CPU and GPU code, there are also a few
example applications included.

### Scene Builder
Run `./scene_builder [mesh_path | scene_path]` to open the "scene builder", a
way to play around with a query point, set up sphere traced renders of smooth
distance isosurfaces with a camera at the query point, and write these scenes to
json files which can be read later. Renders can optionally be set up with a
"material capture" (MatCap), which maps normals to positions on a circular
texture. Scene file examples are in the `scenes` folder. If no arguments are
specified, it instead creates a cube mesh in memory to interact with in the
viewer.

`scene_builder` also accepts other arguments. It supports overriding the
default-provided scene alpha, rendering smooth or exact distance surfaces, and a
custom output file path. The full invocation is:
```bash
./scene_builder [mesh_path | scene_path] [override_alpha] [render_smooth_dist] [output_file]
```

### GPU Sphere Tracer

A GPU sphere tracer is also available on systems with CUDA support. It takes
essentially the same arguments as `scene_builder` but only supports scene files
and does not accept meshes directly as input.

### Rigid Body Simulator
Run `./rigid_body_simulator sim_file [num_threads]`, specifying a path to a
json file describing a rigid body simulation scene, and optionally the number of
threads to use when evaluating smooth distances (default 4). For scene file
examples, see the `sims` folder.

## License
This project is under the MIT license. See [the license](LICENSE) for details.
