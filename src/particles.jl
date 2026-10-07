"""
    ParticleStatus

Why a particle is (or is no longer) being traced:

- `ALIVE` — still being traced.
- `HIT_SOLID` — stopped on an embedded solid (see `impact_id` for which).
- `LEFT_BOX` — left the bounding box of the grid (see `impact_id`: minus the face index,
  `-1 = xmin, -2 = xmax, -3 = ymin/rmin, -4 = ymax/rmax`).
- `TIME_LIMIT` — reached the end of the time span or the step limit while still in the domain.
"""
@enum ParticleStatus::UInt8 ALIVE HIT_SOLID LEFT_BOX TIME_LIMIT

"""
    ParticleSet{T}

Macroparticles in structure-of-arrays layout.

Positions and velocities are always three-dimensional Cartesian, whatever the geometry of the
field: a 2D Cartesian field is invariant in `z`, and an axisymmetric field is invariant under
rotation about the `x` axis (`x` axial, `r = √(y² + z²)`). Keeping all three velocity
components is what retains `v_z` in 2D (needed with magnetic fields) and the azimuthal velocity
in axisymmetry (angular momentum), without any special cases in the pusher.

`weight` is the number of physical particles a macroparticle represents. For steady-state beam
tracing it is used as a current: see [`deposit!`] (later milestone).

Per-particle impact data (`impact_x` …, `impact_id`, `impact_t`) is filled when a particle is
stopped by a surface; see [`ParticleStatus`](@ref).
"""
struct ParticleSet{T <: AbstractFloat, V <: AbstractVector{T}, VI <: AbstractVector{Int32},
    VS <: AbstractVector{ParticleStatus}, S <: AbstractVector{<:IonSpecies{T}}}
    x::V
    y::V
    z::V
    vx::V
    vy::V
    vz::V
    weight::V
    t::V
    species::VI
    status::VS
    impact_x::V
    impact_y::V
    impact_z::V
    impact_vx::V
    impact_vy::V
    impact_vz::V
    impact_id::VI
    species_list::S
end

"""
    ParticleSet(species_list, n; T = Float64)

`n` particles at rest at the origin, all of species 1, alive.
"""
function ParticleSet(species_list::AbstractVector{<:IonSpecies{T}}, n::Integer) where {T}
    z() = zeros(T, n)
    ParticleSet(z(), z(), z(), z(), z(), z(), ones(T, n), z(), ones(Int32, n),
        fill(ALIVE, n), z(), z(), z(), z(), z(), z(), zeros(Int32, n), collect(species_list))
end
ParticleSet(species::IonSpecies, n::Integer) = ParticleSet([species], n)

"""
    ParticleSet(species, positions, velocities; weight = 1)

Particles from vectors of 3-vectors (or 3-tuples).
"""
function ParticleSet(species::IonSpecies{T}, positions::AbstractVector,
        velocities::AbstractVector; weight = one(T)) where {T}
    n = length(positions)
    length(velocities) == n || throw(DimensionMismatch("positions and velocities differ"))
    p = ParticleSet(species, n)
    for i in 1:n
        set_particle!(p, i, positions[i], velocities[i])
    end
    p.weight .= weight
    p
end

Base.length(p::ParticleSet) = length(p.x)

@inline particle_position(p::ParticleSet{T}, i) where {T} = SVector{3, T}(p.x[i], p.y[i], p.z[i])
@inline particle_velocity(p::ParticleSet{T}, i) where {T} = SVector{3, T}(p.vx[i], p.vy[i], p.vz[i])
@inline isalive(p::ParticleSet, i) = p.status[i] == ALIVE
@inline species(p::ParticleSet, i) = p.species_list[p.species[i]]
@inline impact_position(p::ParticleSet{T}, i) where {T} =
    SVector{3, T}(p.impact_x[i], p.impact_y[i], p.impact_z[i])
@inline impact_velocity(p::ParticleSet{T}, i) where {T} =
    SVector{3, T}(p.impact_vx[i], p.impact_vy[i], p.impact_vz[i])

@inline function set_position!(p::ParticleSet, i, x)
    p.x[i], p.y[i], p.z[i] = x[1], x[2], x[3]
    nothing
end
@inline function set_velocity!(p::ParticleSet, i, v)
    p.vx[i], p.vy[i], p.vz[i] = v[1], v[2], v[3]
    nothing
end

"""
    set_particle!(p, i, x, v; species = 1, weight = 1)

Set particle `i` to position `x`, velocity `v`, alive, with time and impact data cleared.
"""
function set_particle!(p::ParticleSet{T}, i, x, v; species::Integer = 1,
        weight = one(T)) where {T}
    set_position!(p, i, x)
    set_velocity!(p, i, v)
    p.species[i] = species
    p.weight[i] = weight
    p.t[i] = zero(T)
    p.status[i] = ALIVE
    p.impact_id[i] = 0
    p
end

nalive(p::ParticleSet) = count(==(ALIVE), p.status)
