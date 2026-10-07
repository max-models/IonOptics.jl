# IonOptics.jl

Self-consistent electrostatic ion optics and charged-particle extraction in Julia, in the spirit
of IBSimu but modular. Status: **milestone 1** (no space charge yet).

The repository holds two packages:

```text
IonOptics.jl               (this directory)    particles, fields, walls, tracing
lib/EmbeddedBoundaryFD     (path dependency)   grids, level-set geometry, Shortley–Weller Poisson
```

`EmbeddedBoundaryFD` knows nothing about particles or plasmas. `IonOptics` builds on it, on
`ElectromagneticFields.jl` (field interface), `ChargedParticleDynamics.jl` (charged-particle
models) and `GeometricIntegrators.jl` (integrators).

## What works

**EmbeddedBoundaryFD**
- `CartesianGrid(xmin, xmax, nx, ymin, ymax, ny)` and `AxisymmetricGrid(z = …, r = …)`:
  node-centred, `nx` cells, `size(grid) == (nx + 1, ny + 1)`. Axisymmetric is `(z, r)` like IBSimu.
- Level-set geometry, `ψ < 0` domain and `ψ > 0` solid: `Circle`, `Rectangle`, `HalfSpace`,
  `LevelSet(f)`, CSG through `∪`, `∩`, `setdiff`, `Complement`. Labelled solids with `Solids(…)`.
- Node classification (`SOLID`, `INTERIOR`, `BOUNDARY_ADJACENT`, `ON_BOUNDARY`) with cut fractions
  `θ` per direction, from linear interpolation (`LinearCut`) or root finding (`RefinedCut`).
  Cuts closer than `θmin` (default 1e-6) snap the node onto the boundary.
- Shortley–Weller `-∇²φ = f`, Cartesian and axisymmetric (regularity on the axis), with Dirichlet
  data per solid and Dirichlet/Neumann box faces. `assemble` (sparse) is separate from
  `solve_linear_system!`. `ShortleyWellerSolver` caches the LU factorization for repeated solves.
- A boundary-aware gradient (second order up to the boundary), multilinear interpolation, and
  ghost-value extrapolation into the solids.

**IonOptics**
- `IonSpecies`, `ParticleSet` (structure of arrays, always 3D Cartesian positions and velocities),
  `Electrode`/`ElectrodeSet` (`Electrode(geo; voltage) ∪ …` goes straight into `PoissonProblem`).
- Fields: `GridElectrostaticField(problem, phi)`, `UniformElectricField`, `UniformMagneticField`,
  `FunctionField`, sums `f + g`. Each one implements the `ElectromagneticFields` accessors, so it
  can be passed to `ChargedParticleDynamics` models as it is.
- `trace!(particles, field, tspan; integrator)` with
  - `Boris(dt, max_displacement)`: native and allocation-free, with an optional adaptive step;
  - `GeometricIntegratorsPusher(method; dt, formulation = :ode | :sode)`: any GeometricIntegrators
    method on `ChargedParticle3d.Noncanonical`. `Strang()` on `:sode` reproduces `Boris` to rounding.
- Wall impacts on the solids and the box, found by root finding on a cubic Hermite path within the
  step. Impact position, velocity, time and the electrode or face id are recorded. `Absorb()` is
  currently the only `SurfaceModel`.

## Running

```sh
julia --project=lib/EmbeddedBoundaryFD -e 'using Pkg; Pkg.test()'
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. examples/aperture.jl
```

## Verified

- Poisson, manufactured solutions at N = 32…256: second order in L∞ and L² on the square, a
  circular solid, the domain inside a circle, two disconnected solids, and a plate with an aperture
  (both cut methods). Also checked with Neumann faces, and in axisymmetric coordinates (cylinder,
  sphere on the axis, torus, aperture). Covered as well: coaxial cylinders and concentric spheres at
  different potentials, boundaries exactly through nodes or 1e-14 h away, and the gradient
  (second order).
- Field: exact for parallel plates. Second order for the coax field both in the open and in cells
  cut by the electrode.
- Particles: parallel-plate energy gain and time of flight to 1e-8. Exact wall-impact locations.
  Box exits. Energy conservation in a numerical field. Angular momentum in an axisymmetric field.
  Native Boris equals GI Strang. The tracing kernel makes zero allocations.

## Not yet

Space charge (deposition, self-consistent iteration), beam sources, diagnostics, and the
regression set (einzel lens, Child–Langmuir, extraction aperture versus IBSimu). See milestone 2.
