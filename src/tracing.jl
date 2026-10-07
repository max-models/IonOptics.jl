#
# Particle trajectories in static fields.
#
#     dx/dt = v,    m dv/dt = q (E(x) + v × B(x))
#

"""
    ParticleIntegrator

How [`trace!`](@ref) advances a particle by one step. Two families:

- [`Boris`](@ref): a native, allocation-free Boris pusher. This is the kernel meant for many
  particles (and, later, GPUs).
- [`GeometricIntegratorsPusher`](@ref): any integrator of `GeometricIntegrators.jl`, applied to
  the charged-particle models of `ChargedParticleDynamics.jl`.

Both integrate the same equations and are cross-checked in the tests.
"""
abstract type ParticleIntegrator end

"""
    Boris(; dt, max_displacement = Inf)

The Boris pusher in kick–drift–kick form: half a Boris velocity update with the fields at `xₙ`,
a full drift, half a velocity update with the fields at `xₙ₊₁`. Each half update is the implicit
midpoint rule for `dv/dt = (q/m)(E + v × B)` at frozen position, so for `E = 0` it rotates the
velocity exactly and conserves `|v|`. The fields at the end of one step are those at the start of
the next, so there is one field evaluation per step.

This is exactly the Strang composition of the splitting `sodeproblem` of
`ChargedParticleDynamics.ChargedParticle3d.Noncanonical` (kick, then drift), i.e. what
`GeometricIntegratorsPusher(Strang(); formulation = :sode)` computes; the tests check that the two
agree to rounding.

The step is `min(dt, max_displacement / |v|)`, so with a finite `max_displacement` (e.g. a
fraction of the grid spacing) slow particles near an emitter take long steps and fast ones short
ones, as in IBSimu. Time is still tracked exactly.
"""
struct Boris{T} <: ParticleIntegrator
    dt::T
    max_displacement::T
end
function Boris(; dt, max_displacement = Inf)
    T = float(promote_type(typeof(dt), typeof(max_displacement)))
    Boris{T}(T(dt), T(max_displacement))
end

@inline function step_size(b::Boris{T}, v, remaining) where {T}
    s = norm(v)
    h = s > 0 ? min(b.dt, b.max_displacement / s) : b.dt
    min(h, remaining)
end

"""
    boris_kick(v, a, ω, h)

The Boris velocity update over time `h` for acceleration `a = (q/m) E` and `ω = (q/m) B`:
equal to the implicit midpoint rule `(v₁ - v₀)/h = a + (v₀ + v₁)/2 × ω`.
"""
@inline function boris_kick(v::SVector{3, T}, a::SVector{3, T}, ω::SVector{3, T}, h) where {T}
    v⁻ = v + (h / 2) * a
    t = (h / 2) * ω
    s = 2 * t / (1 + dot(t, t))
    v′ = v⁻ + cross(v⁻, t)
    v⁺ = v⁻ + cross(v′, s)
    v⁺ + (h / 2) * a
end

@inline fields_at(field, qm, x) = (qm * electric_field(field, x), qm * magnetic_field(field, x))

# One kick–drift–kick step from `x, v` with the scaled fields `a, ω` at `x`. Returns the new
# state and the fields there.
@inline function advance(::Boris, field, qm, x, v, a, ω, h)
    v½ = boris_kick(v, a, ω, h / 2)
    x1 = x + h * v½
    a1, ω1 = fields_at(field, qm, x1)
    v1 = boris_kick(v½, a1, ω1, h / 2)
    x1, v1, a1, ω1
end

"""
    GeometricIntegratorsPusher(method = RK4(); dt, formulation = :ode)

Advance particles with `method` from `GeometricIntegrators.jl`, applied to the noncanonical
charged-particle model of `ChargedParticleDynamics.ChargedParticle3d` in the field scaled by the
species' `q/m` (see [`ChargeScaledField`](@ref)):

- `formulation = :ode` — the vector field `ODEProblem`; use any ODE method (`RK4()`,
  `Gauss(2)`, ...).
- `formulation = :sode` — the Boris splitting `SODEProblem`; use a splitting method
  (`Strang()`, `McLachlan4()`, ...). `Strang()` is the Boris pusher.

The step is fixed to `dt`; the last step is not shortened, so a particle may overshoot the end of
the time span by less than `dt`.

This path builds a problem and an integrator per particle, so it is slower than [`Boris`](@ref);
it gives access to the whole GeometricIntegrators toolbox (high order, symplectic, implicit) and
is the reference the native pusher is tested against.
"""
struct GeometricIntegratorsPusher{M, T} <: ParticleIntegrator
    method::M
    dt::T
    formulation::Symbol
end
function GeometricIntegratorsPusher(method = RK4(); dt, formulation::Symbol = :ode)
    formulation in (:ode, :sode) ||
        throw(ArgumentError("formulation must be :ode or :sode, got :$formulation"))
    GeometricIntegratorsPusher(method, float(dt), formulation)
end

"""
    trace!(particles, field, tspan; integrator, walls = default_walls(field),
           surface = Absorb(), maxsteps = 10^6, record = false)

Trace every alive particle from time `tspan[1]` until `tspan[2]`, until it hits a wall, or for at
most `maxsteps` steps.

After every step the segment from the old to the new position is tested against `walls`
(see [`find_impact`](@ref)); a particle that hits one is handed to the `surface` model, which for
[`Absorb`](@ref) stops it and records the impact. Particles still in flight at `tspan[2]` stay
`ALIVE`; those that run out of steps become `TIME_LIMIT`.

With `record = true` the positions of every step are returned, one vector of `SVector{3}` per
particle (including the start and the impact point); otherwise `nothing` is returned.
"""
function trace!(p::ParticleSet{T}, field::AbstractIonOpticsField, tspan;
        integrator::ParticleIntegrator, walls::AbstractWalls = default_walls(field),
        surface::SurfaceModel = Absorb(), maxsteps::Integer = 1_000_000,
        record::Bool = false) where {T}
    t0, t1 = T(tspan[1]), T(tspan[2])
    trajectories = record ? [SVector{3, T}[] for _ in 1:length(p)] : nothing
    for i in 1:length(p)
        isalive(p, i) || continue
        p.t[i] = t0
        traj = record ? trajectories[i] : nothing
        trace_particle!(p, i, field, walls, surface, integrator, t1, maxsteps, traj)
    end
    trajectories
end

@inline record!(::Nothing, x) = nothing
@inline record!(traj::Vector, x) = (push!(traj, x); nothing)

# Common start: a particle starting inside a wall is stopped where it is.
@inline function start_inside!(p, i, walls, surface, x, v, t, traj)
    wall_value(walls, x) < 0 && return false
    interact!(surface, p, i, Impact(x, v, t, wall_id(walls, x)))
    record!(traj, x)
    true
end

function trace_particle!(p::ParticleSet{T}, i, field, walls, surface, integ::Boris, t1,
        maxsteps, traj) where {T}
    qm = charge_to_mass(species(p, i))
    x, v, t = particle_position(p, i), particle_velocity(p, i), p.t[i]
    record!(traj, x)
    start_inside!(p, i, walls, surface, x, v, t, traj) && return
    a, ω = fields_at(field, qm, x)
    ttol = sqrt(eps(T)) * integ.dt
    for _ in 1:maxsteps
        t1 - t > ttol || break
        h = step_size(integ, v, t1 - t)
        x1, v1, a, ω = advance(integ, field, qm, x, v, a, ω, h)
        imp = find_impact(walls, x, v, t, x1, v1, t + h)
        if imp !== nothing
            record!(traj, imp.x)
            interact!(surface, p, i, imp)
            return
        end
        x, v, t = x1, v1, t + h
        record!(traj, x)
    end
    t1 - t > ttol && (p.status[i] = TIME_LIMIT)
    set_position!(p, i, x)
    set_velocity!(p, i, v)
    p.t[i] = t
    nothing
end

function charged_particle_problem(g::GeometricIntegratorsPusher, q0, t0, t1, params)
    if g.formulation == :ode
        Noncanonical.odeproblem(q0; timespan = (t0, t1), timestep = g.dt,
            parameters = params, periodic = false)
    else
        Noncanonical.sodeproblem(q0; timespan = (t0, t1), timestep = g.dt,
            parameters = params, periodic = false)
    end
end

function trace_particle!(p::ParticleSet{T}, i, field, walls, surface,
        integ::GeometricIntegratorsPusher, t1, maxsteps, traj) where {T}
    qm = charge_to_mass(species(p, i))
    x, v, t = particle_position(p, i), particle_velocity(p, i), p.t[i]
    record!(traj, x)
    start_inside!(p, i, walls, surface, x, v, t, traj) && return

    params = (field = ChargeScaledField(field, qm),)
    problem = charged_particle_problem(integ, Vector{T}(vcat(x, v)), t, t1, params)
    int = GeometricIntegrator(problem, integ.method)
    step = GIB.solutionstep(int, initialstate(problem))
    h = T(integ.dt)
    ttol = sqrt(eps(T)) * h

    for _ in 1:maxsteps
        t1 - t > ttol || break
        GIB.reset!(step, t + h)
        GIB.integrate!(step, int)
        q = GIB.current(step).q
        x1 = SVector{3, T}(q[1], q[2], q[3])
        v1 = SVector{3, T}(q[4], q[5], q[6])
        imp = find_impact(walls, x, v, t, x1, v1, t + h)
        if imp !== nothing
            record!(traj, imp.x)
            interact!(surface, p, i, imp)
            return
        end
        x, v, t = x1, v1, t + h
        record!(traj, x)
    end
    t1 - t > ttol && (p.status[i] = TIME_LIMIT)
    set_position!(p, i, x)
    set_velocity!(p, i, v)
    p.t[i] = t
    nothing
end
