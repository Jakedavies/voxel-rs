# Voxel RS

Voxel renderer in Rust, mostly to learn about lower level graphics programming, graphics pipelines, and aspects of game engines. This is almost certainly full of mistakes and bad practices when it comes to rendering and graphics :). 

Initial rendering code based on the fantastic [Learn WGPU tutorial series](https://sotrh.github.io/learn-wgpu/).


### Status as of April 15 2024

![April 15, 2024 Progress](/progress/2024-04-15_23-52.png)

- Voxel based rendering works.
- Dynamic chunk loading based on camera position
- Basic procedural chunk generation using open simplex noise and height based block selection.
- Basic player controller
- Block type to texture atlas mapping with shader
- Chunk based meshing
- Something resembling a physics engine
- Block destruction

### TODO
In no particular order, here are some future items to look into.
- Block placement
- Block drops
- Seeded RNG
- Fractional voxel support? < a full block
- Modding API for pluggable block generation + adding new block types, drops etc.


