#
# Fields seen by particles.
#
# The particle code only ever calls `electric_field(field, x)`, `magnetic_field(field, x)` and
# `potential(field, x)` with a 3D Cartesian position `x` in SI units. Whether the field is
# analytic, a grid solution of EmbeddedBoundaryFD, or an imported map is invisible to it.
#
# Every field here is also an `ElectromagneticFields.ElectromagneticField` and implements the
# accessors `ChargedParticleDynamics` reads (`φ`, `E♭`, `B♭`, `A♭`, `DA♭`, `g♭`, `Dg♭`) for the
# identity (Cartesian) chart, so it can be handed to the charged-particle models there.
#

"""
    AbstractIonOpticsField

Supertype of the fields in this package. Subtypes implement [`electric_field`](@ref) and, where
non-zero, [`magnetic_field`](@ref), [`potential`](@ref) and [`vector_potential`](@ref).
"""
abstract type AbstractIonOpticsField <: EMF.ElectromagneticField end

"""
    electric_field(field, x) -> SVector{3}

The electric field [V/m] at the Cartesian position `x`.
"""
@inline electric_field(::AbstractIonOpticsField, x) = zero(SVector{3, eltype(x)})

"""
    magnetic_field(field, x) -> SVector{3}

The magnetic field [T] at the Cartesian position `x`.
"""
@inline magnetic_field(::AbstractIonOpticsField, x) = zero(SVector{3, eltype(x)})

"""
    potential(field, x)

The electrostatic potential [V] at the Cartesian position `x`.
"""
@inline potential(::AbstractIonOpticsField, x) = zero(eltype(x))

"""
    vector_potential(field, x) -> SVector{3}
    vector_potential_jacobian(field, x) -> SMatrix{3,3}

The magnetic vector potential `A` and its Jacobian `[i, j] = ∂ⱼAᵢ`.

Only fields that know their vector potential (zero for electrostatic fields, `B × x / 2` for a
uniform field) define these exactly. The default is `A = 0` with the *antisymmetric* Jacobian
`∂ⱼAᵢ = -εᵢⱼₖ Bₖ / 2`, whose curl is `B`: this is all the Lorentz-force vector field of
`ChargedParticleDynamics` (`odeproblem`, `sodeproblem`) needs, but it is not consistent with
`A = 0` and must not be used with the variational (`iodeproblem`, `lodeproblem`) formulations.
"""
@inline vector_potential(::AbstractIonOpticsField, x) = zero(SVector{3, eltype(x)})
@inline function vector_potential_jacobian(f::AbstractIonOpticsField, x)
    B = magnetic_field(f, x)
    antisymmetric_jacobian(B)
end

# The antisymmetric matrix M with Mᵢⱼ = ∂ⱼAᵢ such that curl A = B, i.e. M = -[B]ₓ / 2.
@inline function antisymmetric_jacobian(B::SVector{3, T}) where {T}
    h = T(1 // 2)
    SMatrix{3, 3, T}(0, h * B[3], -h * B[2],
        -h * B[3], 0, h * B[1],
        h * B[2], -h * B[1], 0)
end

# ---------------------------------------------------------------------------------------------
# The ElectromagneticFields accessors, for the identity chart
# ---------------------------------------------------------------------------------------------

@inline _x3(ξ) = SVector{3}(ξ[1], ξ[2], ξ[3])

@inline EMF.φ(f::AbstractIonOpticsField, t, ξ::AbstractVector) = potential(f, _x3(ξ))
@inline EMF.E♭(f::AbstractIonOpticsField, t, ξ::AbstractVector) = electric_field(f, _x3(ξ))
@inline EMF.E♯(f::AbstractIonOpticsField, t, ξ::AbstractVector) = electric_field(f, _x3(ξ))
@inline EMF.B♭(f::AbstractIonOpticsField, t, ξ::AbstractVector) = magnetic_field(f, _x3(ξ))
@inline EMF.B♯(f::AbstractIonOpticsField, t, ξ::AbstractVector) = magnetic_field(f, _x3(ξ))
@inline EMF.B(f::AbstractIonOpticsField, t, ξ::AbstractVector) = norm(magnetic_field(f, _x3(ξ)))
@inline EMF.A♭(f::AbstractIonOpticsField, t, ξ::AbstractVector) = vector_potential(f, _x3(ξ))
@inline EMF.DA♭(f::AbstractIonOpticsField, t, ξ::AbstractVector) =
    vector_potential_jacobian(f, _x3(ξ))
@inline EMF.g♭(::AbstractIonOpticsField, t, ξ::AbstractVector{T}) where {T} =
    SMatrix{3, 3, float(T)}(I)
@inline EMF.g♯(::AbstractIonOpticsField, t, ξ::AbstractVector{T}) where {T} =
    SMatrix{3, 3, float(T)}(I)
@inline EMF.Dg♭(::AbstractIonOpticsField, t, ξ::AbstractVector{T}) where {T} =
    zero(SArray{Tuple{3, 3, 3}, float(T)})
EMF.periodic(::AbstractIonOpticsField) = (false, false, false)
EMF.orientation(::AbstractIonOpticsField) = 1

# ---------------------------------------------------------------------------------------------
# Analytic and composite fields
# ---------------------------------------------------------------------------------------------

"""
    UniformElectricField(E)

A uniform electric field `E` [V/m], with potential `φ = -E ⋅ x`.
"""
struct UniformElectricField{T} <: AbstractIonOpticsField
    E::SVector{3, T}
end
UniformElectricField(E) = UniformElectricField(SVector{3}(float.(E)...))
@inline electric_field(f::UniformElectricField, x) = f.E
@inline potential(f::UniformElectricField, x) = -dot(f.E, x)

"""
    UniformMagneticField(B)

A uniform magnetic field `B` [T], with vector potential `A = B × x / 2`.
"""
struct UniformMagneticField{T} <: AbstractIonOpticsField
    B::SVector{3, T}
end
UniformMagneticField(B) = UniformMagneticField(SVector{3}(float.(B)...))
@inline magnetic_field(f::UniformMagneticField, x) = f.B
@inline vector_potential(f::UniformMagneticField, x) = cross(f.B, x) / 2
@inline vector_potential_jacobian(f::UniformMagneticField, x) = antisymmetric_jacobian(f.B)

"""
    FunctionField(; E = nothing, B = nothing, φ = nothing)

A field given by functions of the Cartesian position. Unspecified parts are zero. `B` only
supports the Lorentz-force formulations; see [`vector_potential`](@ref).
"""
struct FunctionField{FE, FB, FP} <: AbstractIonOpticsField
    E::FE
    B::FB
    φ::FP
end
FunctionField(; E = nothing, B = nothing, φ = nothing) = FunctionField(E, B, φ)
@inline electric_field(f::FunctionField, x) =
    f.E === nothing ? zero(SVector{3, eltype(x)}) : SVector{3}(f.E(x))
@inline magnetic_field(f::FunctionField, x) =
    f.B === nothing ? zero(SVector{3, eltype(x)}) : SVector{3}(f.B(x))
@inline potential(f::FunctionField, x) = f.φ === nothing ? zero(eltype(x)) : f.φ(x)

"""
    FieldSum(fields...)

The superposition of several fields, e.g. a computed electrostatic field and an external
magnetic field: `FieldSum(electrostatic, UniformMagneticField((0, 0, 0.1)))`. Also `f + g`.
"""
struct FieldSum{F <: Tuple} <: AbstractIonOpticsField
    fields::F
end
FieldSum(fields::AbstractIonOpticsField...) = FieldSum(fields)
Base.:+(a::AbstractIonOpticsField, b::AbstractIonOpticsField) = FieldSum(a, b)
Base.:+(a::FieldSum, b::AbstractIonOpticsField) = FieldSum((a.fields..., b))

@inline electric_field(f::FieldSum, x) = sum(g -> electric_field(g, x), f.fields)
@inline magnetic_field(f::FieldSum, x) = sum(g -> magnetic_field(g, x), f.fields)
@inline potential(f::FieldSum, x) = sum(g -> potential(g, x), f.fields)
@inline vector_potential(f::FieldSum, x) = sum(g -> vector_potential(g, x), f.fields)
@inline vector_potential_jacobian(f::FieldSum, x) =
    sum(g -> vector_potential_jacobian(g, x), f.fields)

"""
    ChargeScaledField(field, q_over_m)

`field` multiplied by the charge-to-mass ratio, i.e. the acceleration field `(q/m)(E + v × B)`.
The equations of `ChargedParticleDynamics` are written for unit charge and mass; wrapping a
field in this is how a particle species enters them.
"""
struct ChargeScaledField{F <: AbstractIonOpticsField, T} <: AbstractIonOpticsField
    field::F
    qm::T
end
@inline electric_field(f::ChargeScaledField, x) = f.qm * electric_field(f.field, x)
@inline magnetic_field(f::ChargeScaledField, x) = f.qm * magnetic_field(f.field, x)
@inline potential(f::ChargeScaledField, x) = f.qm * potential(f.field, x)
@inline vector_potential(f::ChargeScaledField, x) = f.qm * vector_potential(f.field, x)
@inline vector_potential_jacobian(f::ChargeScaledField, x) =
    f.qm * vector_potential_jacobian(f.field, x)

# ---------------------------------------------------------------------------------------------
# Grid fields from EmbeddedBoundaryFD
# ---------------------------------------------------------------------------------------------

# Map a 3D Cartesian particle position to grid coordinates, and a grid vector back.
@inline grid_coordinates(::Cartesian, x) = SVector(x[1], x[2])
@inline grid_coordinates(::Axisymmetric, x) = SVector(x[1], hypot(x[2], x[3]))

@inline to_cartesian_vector(::Cartesian, x, G) = SVector(G[1], G[2], zero(G[1]))
@inline function to_cartesian_vector(::Axisymmetric, x, G)
    r = hypot(x[2], x[3])
    iszero(r) && return SVector(G[1], zero(G[1]), zero(G[1]))
    SVector(G[1], G[2] * x[2] / r, G[2] * x[3] / r)
end

"""
    GridElectrostaticField(problem, phi)

The electrostatic field of a solution `phi` of an EmbeddedBoundaryFD `PoissonProblem`.

`E = -∇φ` is computed once at the nodes with the boundary-aware `gradient` of
EmbeddedBoundaryFD, which never differences across a boundary, and interpolated bilinearly to
particle positions. Near electrodes no value from inside a solid is used: the first layer of
solid nodes holds *ghost* values extrapolated linearly from the domain
(`extrapolate_ghosts!`), so bilinear interpolation in cut cells is exact for linear fields and
second order for smooth ones. Corners that are neither domain nor ghost nodes (possible at
re-entrant corners) are dropped and the remaining weights renormalised.

Works for 2D Cartesian grids (the field is invariant in `z`) and axisymmetric `(z, r)` grids
(the particle's `x` is the axial coordinate and `r = √(y² + z²)`).

`field.phi` is the solution as given (electrode potentials inside the solids); the extended
arrays used for interpolation are `field.phi_ext`, `field.Ex`, `field.Ey`.

Use [`update!`](@ref) to replace `phi` in place, e.g. inside a space-charge iteration.
"""
struct GridElectrostaticField{T, G <: StructuredGrid{2, T}, P <: PoissonProblem,
    A <: AbstractMatrix{T}, M <: AbstractMatrix{Bool}} <: AbstractIonOpticsField
    grid::G
    problem::P
    phi::A
    phi_ext::A
    Ex::A
    Ey::A
    mask::M
end

function GridElectrostaticField(problem::PoissonProblem{2, T}, phi::AbstractMatrix) where {T}
    grid = problem.grid
    eb = embedded_boundary(problem)
    z() = zeros(T, size(grid))
    mask = domain_mask(eb) .| ghost_mask(eb)
    f = GridElectrostaticField(grid, problem, z(), z(), z(), z(), mask)
    update!(f, phi)
end

"""
    update!(field, phi)

Replace the potential of a [`GridElectrostaticField`](@ref) and recompute its electric field.
"""
function update!(f::GridElectrostaticField, phi::AbstractMatrix)
    eb = embedded_boundary(f.problem)
    f.phi .= phi
    gradient!((f.Ex, f.Ey), f.phi, f.problem)
    f.Ex .*= -1
    f.Ey .*= -1
    f.phi_ext .= phi
    for A in (f.phi_ext, f.Ex, f.Ey)
        extrapolate_ghosts!(A, eb)
    end
    f
end

@inline function electric_field(f::GridElectrostaticField{T}, x) where {T}
    cs = coordinate_system(f.grid)
    xg = grid_coordinates(cs, x)
    Ex, _ = interpolate(f.grid, f.Ex, f.mask, xg)
    Ey, _ = interpolate(f.grid, f.Ey, f.mask, xg)
    to_cartesian_vector(cs, x, SVector{2, T}(Ex, Ey))
end

@inline function potential(f::GridElectrostaticField, x)
    xg = grid_coordinates(coordinate_system(f.grid), x)
    first(interpolate(f.grid, f.phi_ext, f.mask, xg))
end

"""
    electric_field_grid(field)

The nodal electric field components `(Ex, Ey)` of a grid field, as ordinary arrays (for
diagnostics and plotting). For axisymmetric grids these are `(E_z, E_r)`.
"""
electric_field_grid(f::GridElectrostaticField) = (f.Ex, f.Ey)
