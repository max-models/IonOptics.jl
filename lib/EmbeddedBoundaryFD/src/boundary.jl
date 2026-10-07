#
# Boundary conditions: on the embedded solids, and on the faces of the grid's bounding box.
#

"""
    DirichletBoundary(g)

Dirichlet data on the embedded boundary. `g` is called as `g(x, id)` with the boundary point `x`
(an `SVector`) and the [`solid_id`](@ref) of the solid being cut, and returns the prescribed
value. Convenience forms:

- `DirichletBoundary(v::Number)` — the same value on every solid;
- `DirichletBoundary(values)` with a tuple or vector — `values[id]` on solid `id`.

`g` is also evaluated at nodes inside the solids, to give them a value in the returned solution;
the discretisation itself only ever uses it on the boundary.
"""
struct DirichletBoundary{F}
    g::F
end
DirichletBoundary(v::Number) = DirichletBoundary(ConstantValue(float(v)))
DirichletBoundary(values::Union{Tuple, AbstractVector{<:Number}}) =
    DirichletBoundary(PerSolidValue(Tuple(float.(values))))

struct ConstantValue{T}
    v::T
end
@inline (c::ConstantValue)(x, id) = c.v

struct PerSolidValue{V <: Tuple}
    values::V
end
@inline (c::PerSolidValue)(x, id) = c.values[id]

@inline boundary_value(b::DirichletBoundary, x, id) = b.g(x, id)

"""
    FaceCondition

Boundary condition on one face of the grid's bounding box: [`Dirichlet`](@ref) or
[`Neumann`](@ref). The symmetry axis of an axisymmetric grid is handled automatically
([`Axis`](@ref)).
"""
abstract type FaceCondition end

"""
    Dirichlet(g)

Prescribed value on a box face: `g` is a number or a function `g(x)` of the position.
"""
struct Dirichlet{F} <: FaceCondition
    g::F
end
@inline face_value(c::Dirichlet{<:Number}, x) = c.g
@inline face_value(c::Dirichlet, x) = c.g(x)

"""
    Neumann()

Homogeneous Neumann (zero normal derivative, i.e. symmetry) on a box face.
"""
struct Neumann <: FaceCondition end

"""
    Axis()

The symmetry axis `r = 0` of an axisymmetric grid. Set automatically; never given by the user.
"""
struct Axis <: FaceCondition end

ismirror(::FaceCondition) = false
ismirror(::Neumann) = true
ismirror(::Axis) = true

dimension_names(::StructuredGrid{N, T, Cartesian}) where {N, T} = (:x, :y, :z)[1:N]
dimension_names(::StructuredGrid{2, T, Axisymmetric}) where {T} = (:z, :r)

"""
    face_conditions(grid, box; default = Dirichlet(0.0))

Resolve the box boundary conditions into a tuple ordered like [`side_index`](@ref)
(`xmin, xmax, ymin, ymax, ...`). `box` is a `NamedTuple` keyed by face name — `xmin`, `xmax`,
`ymin`, `ymax` for Cartesian grids and `zmin`, `zmax`, `rmax` for axisymmetric ones; faces not
mentioned get `default`.
"""
function face_conditions(grid::StructuredGrid{N}, box::NamedTuple;
        default::FaceCondition = Dirichlet(0.0)) where {N}
    names = dimension_names(grid)
    valid = Tuple(Symbol(n, s) for n in names for s in (:min, :max))
    for k in keys(box)
        k in valid || throw(ArgumentError("unknown box face $k; valid faces are $valid"))
    end
    if has_axis(grid) && haskey(box, :rmin)
        throw(ArgumentError("rmin is the symmetry axis and takes no boundary condition"))
    end
    ntuple(2N) do k
        d = (k + 1) ÷ 2
        if has_axis(grid) && k == 3
            Axis()
        else
            get(box, valid[k], default)
        end
    end
end
