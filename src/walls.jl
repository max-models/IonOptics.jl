#
# Where particles stop: embedded solids and the bounding box of the grid.
#

"""
    AbstractWalls

What particles can hit. Implementations provide [`wall_value`](@ref) (negative in free space)
and [`wall_id`](@ref).
"""
abstract type AbstractWalls end

"""
    NoWalls()

Free space: nothing stops a particle.
"""
struct NoWalls <: AbstractWalls end
@inline wall_value(::NoWalls, x) = -one(eltype(x))
@inline wall_id(::NoWalls, x) = Int32(0)

"""
    Walls(grid, geometry)

The solids of `geometry` together with the bounding box of `grid`, for particles whose 3D
position is mapped to grid coordinates as in [`GridElectrostaticField`](@ref).

A particle is stopped when its level-set value [`wall_value`](@ref) — the larger of the
geometry's `ψ` and the distance outside the box — becomes non-negative. The id of what it hit is
the `solid_id` of the geometry (`≥ 1`) or minus the index of the box face (`< 0`, see
[`ParticleStatus`](@ref)).
"""
struct Walls{G <: StructuredGrid, Geo <: AbstractGeometry} <: AbstractWalls
    grid::G
    geometry::Geo
end

# Signed distance outside the bounding box (negative inside). The symmetry axis of an
# axisymmetric grid is not a wall: r ≥ 0 always, and particles cross the axis freely.
@inline function box_value(grid::StructuredGrid{N}, xg) where {N}
    s = typemin(eltype(xg))
    for d in 1:N
        is_axis(grid, d) || (s = max(s, grid.lo[d] - xg[d]))
        s = max(s, xg[d] - grid.hi[d])
    end
    s
end

@inline is_axis(grid::StructuredGrid, d) = d == 2 && has_axis(grid)

# Which box face is crossed: the one the point is furthest beyond.
@inline function box_face(grid::StructuredGrid{N}, xg) where {N}
    best = typemin(eltype(xg))
    k = 0
    for d in 1:N
        a, b = grid.lo[d] - xg[d], xg[d] - grid.hi[d]
        !is_axis(grid, d) && a > best && (best = a; k = side_index(d, -1))
        b > best && (best = b; k = side_index(d, +1))
    end
    k
end

@inline function wall_value(w::Walls, x)
    xg = grid_coordinates(coordinate_system(w.grid), x)
    max(value(w.geometry, xg), box_value(w.grid, xg))
end

@inline function wall_id(w::Walls, x)
    xg = grid_coordinates(coordinate_system(w.grid), x)
    if value(w.geometry, xg) ≥ box_value(w.grid, xg)
        Int32(solid_id(w.geometry, xg))
    else
        Int32(-box_face(w.grid, xg))
    end
end

"""
    default_walls(field)

The walls implied by a field: those of the Poisson problem for a [`GridElectrostaticField`](@ref)
(the first one found in a [`FieldSum`](@ref)), and [`NoWalls`](@ref) otherwise.
"""
default_walls(::AbstractIonOpticsField) = NoWalls()
default_walls(f::GridElectrostaticField) = Walls(f.grid, geometry(f.problem))
function default_walls(f::FieldSum)
    for g in f.fields
        w = default_walls(g)
        w isa NoWalls || return w
    end
    NoWalls()
end
default_walls(f::ChargeScaledField) = default_walls(f.field)

"""
    Impact{T}

Where, when and with what velocity a particle hit a wall, and which wall (`id`, see
[`Walls`](@ref)).
"""
struct Impact{T}
    x::SVector{3, T}
    v::SVector{3, T}
    t::T
    id::Int32
end

"""
    find_impact(walls, x0, v0, t0, x1, v1, t1; tol)

If the step from `(x0, v0, t0)` to `(x1, v1, t1)` ends in a wall, the [`Impact`](@ref) at the
first crossing of the path; `nothing` otherwise.

The path within the step is the cubic Hermite interpolant of the positions and velocities at its
ends, which is exact for uniform acceleration and fourth order in general. The crossing is
located on it by safeguarded regula falsi on the wall level set, and the velocity is the
interpolant's derivative there.

Only the end point is tested, so a wall thinner than one step can be skipped; choose the step
accordingly.
"""
@inline function find_impact(walls::AbstractWalls, x0::SVector{3, T}, v0, t0, x1, v1, t1;
        tol = 1e-12) where {T}
    ψ1 = wall_value(walls, x1)
    ψ1 < 0 && return nothing
    h = t1 - t0
    ψ0 = wall_value(walls, x0)
    s = ψ0 ≥ 0 ? zero(T) :
        T(segment_crossing(τ -> wall_value(walls, hermite(x0, v0, x1, v1, h, τ)), ψ0, ψ1; tol))
    xh = hermite(x0, v0, x1, v1, h, s)
    vh = hermite_derivative(x0, v0, x1, v1, h, s)
    # the id is decided just past the crossing, where the wall that was hit dominates
    xid = hermite(x0, v0, x1, v1, h, min(s + sqrt(eps(T)), one(T)))
    Impact(xh, vh, t0 + s * h, wall_id(walls, xid))
end

@inline function hermite(x0, v0, x1, v1, h, τ)
    τ² = τ * τ
    τ³ = τ² * τ
    (2τ³ - 3τ² + 1) * x0 + (τ³ - 2τ² + τ) * h * v0 + (3τ² - 2τ³) * x1 + (τ³ - τ²) * h * v1
end

@inline function hermite_derivative(x0, v0, x1, v1, h, τ)
    τ² = τ * τ
    ((6τ² - 6τ) * x0 + (3τ² - 4τ + 1) * h * v0 + (6τ - 6τ²) * x1 + (3τ² - 2τ) * h * v1) / h
end

"""
    SurfaceModel

What happens to a particle that hits a wall. Implement
`interact!(model, particles, i, impact::Impact)`, which must set the particle's status.

Only [`Absorb`](@ref) exists so far; reflection, secondary emission, sputtering or surface
charging are further subtypes.
"""
abstract type SurfaceModel end

"""
    Absorb()

The particle stops at the surface; its impact position, velocity and the wall id are recorded.
"""
struct Absorb <: SurfaceModel end

@inline function interact!(::Absorb, p::ParticleSet, i, imp::Impact)
    p.impact_x[i], p.impact_y[i], p.impact_z[i] = imp.x
    p.impact_vx[i], p.impact_vy[i], p.impact_vz[i] = imp.v
    p.impact_id[i] = imp.id
    p.status[i] = imp.id > 0 ? HIT_SOLID : LEFT_BOX
    set_position!(p, i, imp.x)
    set_velocity!(p, i, imp.v)
    p.t[i] = imp.t
    nothing
end
