"""
    CoordinateSystem

How the coordinates of a [`StructuredGrid`](@ref) are to be interpreted by the operators built on
it. The grid itself is always a tensor product of uniform 1D node sets; only the operator (and
quantities such as control volumes) depend on the coordinate system.
"""
abstract type CoordinateSystem end

"""
    Cartesian()

Plain Cartesian coordinates, in any dimension.
"""
struct Cartesian <: CoordinateSystem end

"""
    Axisymmetric()

Cylindrical coordinates with rotational symmetry, `(z, r)`: dimension 1 is the axial coordinate
`z` and dimension 2 the radius `r ≥ 0`. This follows the IBSimu convention (`x` is the axis, `y`
the radius), so a 2D Cartesian particle code needs no reordering to run axisymmetrically.
"""
struct Axisymmetric <: CoordinateSystem end

"""
    StructuredGrid{N,T,C}

A uniform, node-centred, tensor-product grid in `N` dimensions.

`ncells[d]` cells of width `h[d]` span `[lo[d], hi[d]]`, so there are `ncells[d] + 1` nodes in
dimension `d`. Every array living on the grid (potential, source, charge density, ...) has size
`size(grid) == ncells .+ 1`.
"""
struct StructuredGrid{N, T <: AbstractFloat, C <: CoordinateSystem}
    lo::NTuple{N, T}
    hi::NTuple{N, T}
    ncells::NTuple{N, Int}
    h::NTuple{N, T}
    coordinate_system::C

    function StructuredGrid(lo::NTuple{N, T}, hi::NTuple{N, T}, ncells::NTuple{N, Int},
            cs::C = Cartesian()) where {N, T <: AbstractFloat, C <: CoordinateSystem}
        all(hi .> lo) || throw(ArgumentError("grid requires hi > lo in every dimension"))
        all(ncells .≥ 2) || throw(ArgumentError("grid requires at least two cells per dimension"))
        h = (hi .- lo) ./ ncells
        new{N, T, C}(lo, hi, ncells, h, cs)
    end
end

"""
    CartesianGrid(xmin, xmax, nx, ymin, ymax, ny)

A 2D Cartesian grid with `nx × ny` cells, i.e. `(nx + 1) × (ny + 1)` nodes.
"""
function CartesianGrid(xmin, xmax, nx::Integer, ymin, ymax, ny::Integer)
    T = float(promote_type(typeof(xmin), typeof(xmax), typeof(ymin), typeof(ymax)))
    StructuredGrid((T(xmin), T(ymin)), (T(xmax), T(ymax)), (Int(nx), Int(ny)), Cartesian())
end

"""
    AxisymmetricGrid(; z = (zmin, zmax, nz), r = (rmin, rmax, nr))

An axisymmetric `(z, r)` grid with `nz × nr` cells. Dimension 1 is `z`, dimension 2 is `r`.
`rmin` must be non-negative; if it is zero the lower radial face is the symmetry axis and is
treated as such by the operators (no boundary condition is needed or accepted there).
"""
function AxisymmetricGrid(; z, r)
    zmin, zmax, nz = z
    rmin, rmax, nr = r
    rmin ≥ 0 || throw(ArgumentError("axisymmetric grid requires rmin ≥ 0, got $rmin"))
    T = float(promote_type(typeof(zmin), typeof(zmax), typeof(rmin), typeof(rmax)))
    StructuredGrid((T(zmin), T(rmin)), (T(zmax), T(rmax)), (Int(nz), Int(nr)), Axisymmetric())
end

Base.ndims(::StructuredGrid{N}) where {N} = N
Base.eltype(::StructuredGrid{N, T}) where {N, T} = T
Base.size(grid::StructuredGrid) = grid.ncells .+ 1
Base.CartesianIndices(grid::StructuredGrid) = CartesianIndices(size(grid))
Base.LinearIndices(grid::StructuredGrid) = LinearIndices(size(grid))

coordinate_system(grid::StructuredGrid) = grid.coordinate_system

"""
    spacing(grid)

The node spacing in each dimension, as an `NTuple`.
"""
spacing(grid::StructuredGrid) = grid.h

"""
    coordinates(grid)

The node coordinates in each dimension, as a tuple of ranges.
"""
function coordinates(grid::StructuredGrid{N}) where {N}
    ntuple(d -> range(grid.lo[d], grid.hi[d]; length = grid.ncells[d] + 1), Val(N))
end

"""
    cell_centers(grid)

The cell-centre coordinates in each dimension, as a tuple of ranges.
"""
function cell_centers(grid::StructuredGrid{N}) where {N}
    ntuple(d -> range(grid.lo[d] + grid.h[d] / 2, grid.hi[d] - grid.h[d] / 2;
            length = grid.ncells[d]), Val(N))
end

"""
    node_position(grid, I)

Position of node `I` (a `CartesianIndex` or a tuple of integers) as an `SVector`.
"""
@inline function node_position(grid::StructuredGrid{N, T}, I::CartesianIndex{N}) where {N, T}
    SVector{N, T}(ntuple(d -> grid.lo[d] + (I[d] - 1) * grid.h[d], Val(N)))
end
@inline node_position(grid::StructuredGrid{N}, I::NTuple{N, Integer}) where {N} =
    node_position(grid, CartesianIndex(I))

"""
    on_axis(grid, I)

Whether node `I` lies on the symmetry axis of an axisymmetric grid. Always false otherwise.
"""
@inline on_axis(::StructuredGrid, I) = false
@inline on_axis(grid::StructuredGrid{2, T, Axisymmetric}, I::CartesianIndex{2}) where {T} =
    I[2] == 1 && iszero(grid.lo[2])

has_axis(::StructuredGrid) = false
has_axis(grid::StructuredGrid{2, T, Axisymmetric}) where {T} = iszero(grid.lo[2])

"""
    locate(grid, x)

The cell containing `x` and the local coordinates within it: returns `(I, ξ)` where `I` is the
`CartesianIndex` of the lower-left node and `ξ ∈ [0, 1]^N`. Points outside the grid are clamped
to the boundary cell, so callers that care must test [`inside_box`](@ref) first.
"""
@inline function locate(grid::StructuredGrid{N, T}, x) where {N, T}
    idx = ntuple(Val(N)) do d
        s = (x[d] - grid.lo[d]) / grid.h[d]
        i = clamp(unsafe_trunc(Int, floor(s)), 0, grid.ncells[d] - 1)
        (i + 1, clamp(s - i, zero(T), one(T)))
    end
    CartesianIndex(map(first, idx)), SVector{N, T}(map(last, idx))
end

"""
    inside_box(grid, x)

Whether `x` lies within the bounding box of the grid (boundary included).
"""
@inline function inside_box(grid::StructuredGrid{N}, x) where {N}
    all(ntuple(d -> grid.lo[d] ≤ x[d] ≤ grid.hi[d], Val(N)))
end

"""
    node_volume(grid, I)

The volume of the dual (control) cell of node `I`, clipped to the grid's bounding box.

In Cartesian coordinates this is `∏ h_d`, halved for every face the node sits on. In
axisymmetric coordinates it is the true volume of revolution, `2π r h_z h_r` in the interior and
`π (h_r/2)² h_z` on the axis, so that `sum(node_volume)` equals the volume of the cylinder.

This is purely geometric: it does not know about embedded solids.
"""
function node_volume(grid::StructuredGrid{N, T, Cartesian}, I::CartesianIndex{N}) where {N, T}
    v = one(T)
    for d in 1:N
        f = (I[d] == 1 || I[d] == grid.ncells[d] + 1) ? T(0.5) : one(T)
        v *= f * grid.h[d]
    end
    v
end

function node_volume(grid::StructuredGrid{2, T, Axisymmetric}, I::CartesianIndex{2}) where {T}
    hz, hr = grid.h
    fz = (I[1] == 1 || I[1] == grid.ncells[1] + 1) ? T(0.5) : one(T)
    r = grid.lo[2] + (I[2] - 1) * hr
    # radial extent of the dual cell, clipped to [rmin, rmax]
    r₋ = I[2] == 1 ? r : r - hr / 2
    r₊ = I[2] == grid.ncells[2] + 1 ? r : r + hr / 2
    T(π) * (r₊^2 - r₋^2) * fz * hz
end

"""
    node_volumes(grid)

[`node_volume`](@ref) for every node, as an array of `size(grid)`.
"""
node_volumes(grid::StructuredGrid) = [node_volume(grid, I) for I in CartesianIndices(grid)]

function Base.show(io::IO, grid::StructuredGrid{N, T, C}) where {N, T, C}
    print(io, "StructuredGrid{$N,$T,$(nameof(C))}(", join(grid.ncells, "×"), " cells, ",
        join(("[$(grid.lo[d]), $(grid.hi[d])]" for d in 1:N), " × "), ")")
end
