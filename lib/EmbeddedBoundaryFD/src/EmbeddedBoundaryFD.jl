"""
    EmbeddedBoundaryFD

Finite-difference discretisations on structured Cartesian grids with embedded (immersed)
boundaries described by level sets.

Geometry ([`AbstractGeometry`](@ref), CSG), node classification
([`EmbeddedBoundary`](@ref)), and the Shortley–Weller Poisson operator
([`PoissonProblem`](@ref), [`assemble`](@ref), [`ShortleyWellerSolver`](@ref)) in 2D Cartesian
and axisymmetric `(z, r)` coordinates.

This package knows nothing about particles or plasmas.
"""
module EmbeddedBoundaryFD

using LinearAlgebra
using SparseArrays
using StaticArrays

export CoordinateSystem, Cartesian, Axisymmetric
export StructuredGrid, CartesianGrid, AxisymmetricGrid
export spacing, coordinates, cell_centers, node_position, node_volume, node_volumes
export coordinate_system, locate, inside_box, on_axis, has_axis
include("grid.jl")

export AbstractGeometry, value, inside, solid_id, nsolids, evaluate
export Circle, Rectangle, HalfSpace, LevelSet, NoSolid
export CSGUnion, CSGIntersection, CSGDifference, Complement, Solids
include("geometry.jl")

export NodeKind, SOLID, INTERIOR, BOUNDARY_ADJACENT, ON_BOUNDARY, isdomain
export LinearCut, RefinedCut, segment_crossing
export EmbeddedBoundary, theta_xm, theta_xp, theta_ym, theta_yp, domain_mask, count_kinds
export side_index, cut_point
include("classification.jl")

export DirichletBoundary, FaceCondition, Dirichlet, Neumann, Axis, face_conditions
include("boundary.jl")

export PoissonProblem, assemble, assemble_operator, solve_linear_system!, scatter!
export ShortleyWeller, ShortleyWellerSolver, solve!, solve, ndofs, embedded_boundary
include("poisson.jl")

export gradient, gradient!, interpolate, ghost_mask, extrapolate_ghosts!
include("gradient.jl")

end
