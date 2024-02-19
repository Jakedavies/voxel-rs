# Voxel RS

Voxel renderer in Rust, mostly to learn about lower level graphics programming, graphics pipelines, and aspects of game engines. This is almost certainly full of mistakes and bad practices when it comes to rendering and graphics :). 

Initial rendering code based on the fantastic [Learn WGPU tutorial series](https://sotrh.github.io/learn-wgpu/).


### Status as of Feb 19 2024

[[progress/2024-02-19_21-40.png|Feb 19, 2024 Progress]]

- Voxel based rendering works.
- Mapping of textures based on block type in shader using texture atlas.
- Dynamic chunk loading based on camera position
- Basic procedural chunk generation using open simplex noise and height based block selection.
- Basic camera controller, with gravity. Arbitrary floor at y = 32.

### TODO
In no particular order, here are some future items to look into.

- Greedy mesh optimization rather than drawing 4096 voxel instances for each chunk.
- Basic physics system
- Block destruction, placement
- Block drops
- Seeded RNG
- Fractional voxel support? < a full block
- "Modding API" for pluggable block generation + adding new block types, drops etc.


