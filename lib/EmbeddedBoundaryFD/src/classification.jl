#
# Classification of grid nodes against an embedded geometry, and the boundary cuts along grid
# lines.
#

"""
    NodeKind

The role of a grid node with respect to the embedded geometry:

- `SOLID` — inside a solid (`ψ ≥ 0`); not an unknown.
- `INTERIOR` — in the domain, and no grid line to a neighbour crosses the boundary.
- `BOUNDARY_ADJACENT` — in the domain, with at least one grid line to a neighbour cut by the
  boundary. Its stencil is a Shortley–Weller stencil.
- `ON_BOUNDARY` — in the domain but closer to the boundary than `θmin` grid spacings along some
  grid line. Such nodes are treated as lying on the boundary (their value is prescribed), which
  avoids the arbitrarily large stencil coefficients of a vanishing cut distance.
"""
@enum NodeKind::UInt8 SOLID INTERIOR BOUNDARY_ADJACENT ON_BOUNDARY

"""
    isdomain(kind)

Whether a node of this kind lies in the computational domain (anything but `SOLID`).
"""
@inline isdomain(k::NodeKind) = k != SOLID

# ---------------------------------------------------------------------------------------------
# Locating the boundary along a segment
# ---------------------------------------------------------------------------------------------

"""
    LinearCut()

Locate boundary crossings by linear interpolation of the nodal level-set values,
`θ = |ψᵢ| / (|ψᵢ| + |ψⱼ|)`. Exact for planar boundaries, `O(h²)` in position otherwise.
"""
struct LinearCut end

"""
    RefinedCut(; tol = 1e-12, maxiter = 50)

Locate boundary crossings by root finding on the level-set function itself along the grid line,
starting from the linear estimate. `tol` is relative to the segment length.
"""
Base.@kwdef struct RefinedCut
    tol::Float64 = 1e-12
    maxiter::Int = 50
end

"""
    segment_crossing(f, fa, fb; tol, maxiter)

Find `t ∈ [0, 1]` with `f(t) = 0`, given `fa = f(0)` and `fb = f(1)` of opposite sign (or
`fb == 0`). Uses the Illinois variant of regula falsi, which is bracketing (so it never leaves
`[0, 1]`) and superlinearly convergent. Returns `t`.

Used both for boundary cuts along grid lines and for particle–wall intersections.
"""
function segment_crossing(f, fa::T, fb::T; tol = 1e-12, maxiter::Int = 50) where {T}
    iszero(fb) && return one(T)
    iszero(fa) && return zero(T)
    a, b = zero(T), one(T)
    side = 0
    t = a
    for _ in 1:maxiter
        t = (a * fb - b * fa) / (fb - fa)
        ft = f(t)
        if iszero(ft) || (b - a) < tol
            return t
        end
        if signbit(ft) == signbit(fa)
            a, fa = t, ft
            side == -1 && (fb /= 2)
            side = -1
        else
            b, fb = t, ft
            side == +1 && (fa /= 2)
            side = +1
        end
    end
    t
end

@inline cut_fraction(::LinearCut, geometry, xa, xb, ψa, ψb) = ψa / (ψa - ψb)

@inline function cut_fraction(c::RefinedCut, geometry, xa, xb, ψa, ψb)
    f = t -> value(geometry, xa + t * (xb - xa))
    segment_crossing(f, ψa, ψb; tol = c.tol, maxiter = c.maxiter)
end

# ---------------------------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------------------------

"""
    side_index(d, s)

Index of the side of a node in direction `d` (`s = -1` or `+1`) into the leading dimension of
the cut arrays: `1 = x-`, `2 = x+`, `3 = y-`, `4 = y+`, ...
"""
@inline side_index(d::Int, s::Int) = 2(d - 1) + (s > 0 ? 2 : 1)

"""
    EmbeddedBoundary(grid, geometry; cut = LinearCut(), θmin = 1e-6)

The classification of every node of `grid` against `geometry`, together with the boundary cuts
along the grid lines.

For a domain node `I` and a direction/side `k = side_index(d, s)`:

- `theta[k, I] ∈ (0, 1]` is the distance to the boundary along that grid line in units of
  `h[d]` (so `d₊ = θ₊ h`). It is `1` if the grid line to the neighbour is not cut — this
  includes the case where the neighbour lies outside the grid.
- `cut_id[k, I]` is the [`solid_id`](@ref) of the solid that is cut, or `0` if not cut.

Nodes with a cut closer than `θmin` are marked `ON_BOUNDARY`; see [`NodeKind`](@ref).

Only the first crossing along a grid line is detected: a solid thinner than one grid spacing that
lies entirely between two domain nodes is invisible to the discretisation, as in any
node-based embedded-boundary method.
"""
struct EmbeddedBoundary{N, T, NP1, G <: StructuredGrid{N, T}, Geo <: AbstractGeometry}
    grid::G
    geometry::Geo
    psi::Array{T, N}
    kind::Array{NodeKind, N}
    theta::Array{T, NP1}
    cut_id::Array{Int32, NP1}
    θmin::T
end

function EmbeddedBoundary(grid::StructuredGrid{N, T}, geometry::AbstractGeometry;
        cut = LinearCut(), θmin = 1e-6) where {N, T}
    sz = size(grid)
    psi = evaluate(geometry, grid)
    kind = fill(INTERIOR, sz)
    theta = ones(T, 2N, sz...)
    cut_id = zeros(Int32, 2N, sz...)
    θmin = T(θmin)

    for I in CartesianIndices(sz)
        if psi[I] ≥ 0
            kind[I] = SOLID
            continue
        end
        xI = node_position(grid, I)
        θsmallest = one(T)
        for d in 1:N, s in (-1, 1)
            J = I + CartesianIndex(ntuple(e -> e == d ? s : 0, Val(N)))
            checkbounds(Bool, psi, J) || continue
            psi[J] ≥ 0 || continue
            xJ = node_position(grid, J)
            θ = T(cut_fraction(cut, geometry, xI, xJ, psi[I], psi[J]))
            θ = clamp(θ, eps(T), one(T))
            k = side_index(d, s)
            theta[k, I] = θ
            # the solid that owns the crossing; evaluated just past it
            cut_id[k, I] = solid_id(geometry, xI + min(θ + sqrt(eps(T)), one(T)) * (xJ - xI))
            θsmallest = min(θsmallest, θ)
        end
        if θsmallest < θmin
            kind[I] = ON_BOUNDARY
        elseif θsmallest < one(T) || any(k -> cut_id[k, I] != 0, 1:(2N))
            kind[I] = BOUNDARY_ADJACENT
        end
    end

    EmbeddedBoundary(grid, geometry, psi, kind, theta, cut_id, θmin)
end

theta_xm(eb::EmbeddedBoundary) = selectdim(eb.theta, 1, 1)
theta_xp(eb::EmbeddedBoundary) = selectdim(eb.theta, 1, 2)
theta_ym(eb::EmbeddedBoundary) = selectdim(eb.theta, 1, 3)
theta_yp(eb::EmbeddedBoundary) = selectdim(eb.theta, 1, 4)

"""
    domain_mask(eb)

`true` for every node in the computational domain (i.e. not `SOLID`).
"""
domain_mask(eb::EmbeddedBoundary) = isdomain.(eb.kind)

"""
    count_kinds(eb)

How many nodes of each [`NodeKind`](@ref) there are, as a `Dict`.
"""
function count_kinds(eb::EmbeddedBoundary)
    Dict(k => count(==(k), eb.kind) for k in instances(NodeKind))
end

"""
    nearest_cut(eb, I)

For an `ON_BOUNDARY` node, the side index and boundary point of its closest cut.
"""
function nearest_cut(eb::EmbeddedBoundary{N, T}, I::CartesianIndex{N}) where {N, T}
    kbest = 0
    θbest = T(Inf)
    for k in 1:(2N)
        if eb.cut_id[k, I] != 0 && eb.theta[k, I] < θbest
            kbest, θbest = k, eb.theta[k, I]
        end
    end
    kbest
end

"""
    cut_point(eb, I, k)

The boundary point on side `k` of node `I`.
"""
@inline function cut_point(eb::EmbeddedBoundary{N, T}, I::CartesianIndex{N}, k::Int) where {N, T}
    d = (k + 1) ÷ 2
    s = isodd(k) ? -1 : 1
    x = node_position(eb.grid, I)
    x + SVector{N, T}(ntuple(e -> e == d ? s * eb.theta[k, I] * eb.grid.h[d] : zero(T), Val(N)))
end
