#
# Level-set geometry.
#
# Every geometry is a signed function ψ(x) with
#
#     ψ < 0   computational domain (vacuum)
#     ψ = 0   boundary
#     ψ > 0   solid
#
# Primitives describe a *solid* region. ψ is the (negated) signed distance for the primitives;
# CSG combinations use max/min, which keeps the zero set exact but makes ψ only a bound on the
# distance away from it. Nothing downstream assumes ψ is a true distance.
#

"""
    AbstractGeometry

A solid region described by a level-set function. Subtypes implement [`value`](@ref).
"""
abstract type AbstractGeometry end

"""
    value(geometry, x)

The level-set value `ψ(x)`: negative in the domain, positive in the solid.
"""
function value end

"""
    inside(geometry, x)

Whether `x` lies in the computational domain, i.e. `ψ(x) < 0`.
"""
@inline inside(g::AbstractGeometry, x) = value(g, x) < 0

"""
    solid_id(geometry, x)

Which labelled solid the point `x` belongs to (or is closest to). Plain geometries consist of a
single solid, `1`; [`Solids`](@ref) returns the index of the part with the largest `ψ`.
"""
@inline solid_id(::AbstractGeometry, x) = 1

"""
    nsolids(geometry)

The number of labelled solids in `geometry`.
"""
nsolids(::AbstractGeometry) = 1

# ---------------------------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------------------------

"""
    Circle(center, radius)

A solid disk.
"""
struct Circle{T} <: AbstractGeometry
    center::SVector{2, T}
    radius::T
end
function Circle(center, radius)
    T = float(promote_type(eltype(center), typeof(radius)))
    Circle{T}(SVector{2, T}(center...), T(radius))
end

@inline function value(c::Circle, x)
    c.radius - hypot(x[1] - c.center[1], x[2] - c.center[2])
end

"""
    Rectangle(lo, hi)

A solid axis-aligned rectangle `[lo[1], hi[1]] × [lo[2], hi[2]]`.
"""
struct Rectangle{T} <: AbstractGeometry
    lo::SVector{2, T}
    hi::SVector{2, T}
end
function Rectangle(lo, hi)
    T = float(promote_type(eltype(lo), eltype(hi)))
    all(hi .> lo) || throw(ArgumentError("Rectangle requires hi > lo"))
    Rectangle{T}(SVector{2, T}(lo...), SVector{2, T}(hi...))
end

@inline function value(r::Rectangle, x)
    # exact signed distance to a box, negated
    cx = (r.lo[1] + r.hi[1]) / 2
    cy = (r.lo[2] + r.hi[2]) / 2
    qx = abs(x[1] - cx) - (r.hi[1] - r.lo[1]) / 2
    qy = abs(x[2] - cy) - (r.hi[2] - r.lo[2]) / 2
    outside = hypot(max(qx, zero(qx)), max(qy, zero(qy)))
    insidepart = min(max(qx, qy), zero(qx))
    -(outside + insidepart)
end

"""
    HalfSpace(point, normal)

The solid half-space behind the plane through `point` with outward normal `normal`, i.e. the
solid is `{x : (x - point) ⋅ n < 0}` and `normal` points into the domain.
"""
struct HalfSpace{N, T} <: AbstractGeometry
    point::SVector{N, T}
    normal::SVector{N, T}
end
function HalfSpace(point, normal)
    N = length(point)
    T = float(promote_type(eltype(point), eltype(normal)))
    n = SVector{N, T}(normal...)
    HalfSpace{N, T}(SVector{N, T}(point...), n / norm(n))
end

@inline function value(h::HalfSpace{N}, x) where {N}
    s = zero(eltype(h.point))
    for d in 1:N
        s += (x[d] - h.point[d]) * h.normal[d]
    end
    -s
end

"""
    LevelSet(f)

A solid described by an arbitrary function `f(x)` with the sign convention of [`value`](@ref).
"""
struct LevelSet{F} <: AbstractGeometry
    f::F
end
@inline value(l::LevelSet, x) = l.f(x)

# ---------------------------------------------------------------------------------------------
# Constructive solid geometry
# ---------------------------------------------------------------------------------------------

# `Union` is a Julia builtin, so the CSG nodes carry a prefix. The set operators `∪`, `∩` and
# `setdiff` build them.

"""
    CSGUnion(a, b)

The union of two solids, `ψ = max(ψa, ψb)`. Also `a ∪ b`.
"""
struct CSGUnion{A <: AbstractGeometry, B <: AbstractGeometry} <: AbstractGeometry
    a::A
    b::B
end
@inline value(g::CSGUnion, x) = max(value(g.a, x), value(g.b, x))

"""
    CSGIntersection(a, b)

The intersection of two solids, `ψ = min(ψa, ψb)`. Also `a ∩ b`.
"""
struct CSGIntersection{A <: AbstractGeometry, B <: AbstractGeometry} <: AbstractGeometry
    a::A
    b::B
end
@inline value(g::CSGIntersection, x) = min(value(g.a, x), value(g.b, x))

"""
    CSGDifference(a, b)

The solid `a` with `b` removed, `ψ = min(ψa, -ψb)`. Also `setdiff(a, b)`.
"""
struct CSGDifference{A <: AbstractGeometry, B <: AbstractGeometry} <: AbstractGeometry
    a::A
    b::B
end
@inline value(g::CSGDifference, x) = min(value(g.a, x), -value(g.b, x))

"""
    Complement(a)

Everything that is not `a`, `ψ = -ψa`.
"""
struct Complement{A <: AbstractGeometry} <: AbstractGeometry
    a::A
end
@inline value(g::Complement, x) = -value(g.a, x)

Base.union(a::AbstractGeometry, b::AbstractGeometry) = CSGUnion(a, b)
Base.intersect(a::AbstractGeometry, b::AbstractGeometry) = CSGIntersection(a, b)
Base.setdiff(a::AbstractGeometry, b::AbstractGeometry) = CSGDifference(a, b)

# ---------------------------------------------------------------------------------------------
# Labelled collections
# ---------------------------------------------------------------------------------------------

"""
    Solids(parts...)

A collection of labelled solids. Geometrically it is their union; in addition [`solid_id`](@ref)
tells which part a point belongs to, which is what lets boundary conditions differ between
disconnected solids (e.g. electrodes at different potentials).

Parts are numbered from 1 in the order given. The parts are stored as a tuple, so evaluation is
type-stable for heterogeneous parts.
"""
struct Solids{P <: Tuple} <: AbstractGeometry
    parts::P
end
Solids(parts::AbstractGeometry...) = Solids(parts)

@inline value(s::Solids, x) = maximum(map(p -> value(p, x), s.parts))
@inline solid_id(s::Solids, x) = argmax(map(p -> value(p, x), s.parts))
nsolids(s::Solids) = length(s.parts)

"""
    evaluate(geometry, grid)

The level-set value at every node of `grid`, as an array of `size(grid)`.
"""
function evaluate(geometry::AbstractGeometry, grid::StructuredGrid)
    [value(geometry, node_position(grid, I)) for I in CartesianIndices(grid)]
end

"""
    NoSolid()

The empty geometry: everything is domain.
"""
struct NoSolid <: AbstractGeometry end
@inline value(::NoSolid, x) = -floatmax(float(eltype(x)))
