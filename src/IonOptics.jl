"""
    IonOptics

Electrostatic ion optics and charged-particle extraction on embedded-boundary grids.

Fields are solved with `EmbeddedBoundaryFD`; particles are traced either with the native
[`Boris`](@ref) pusher or with any integrator of `GeometricIntegrators.jl` applied to the
charged-particle models of `ChargedParticleDynamics.jl`. Fields implement the
`ElectromagneticFields.jl` accessors so that they can be handed to those models directly.
"""
module IonOptics

using LinearAlgebra
using StaticArrays
using Reexport

@reexport using EmbeddedBoundaryFD
import EmbeddedBoundaryFD: geometry, side_index

import ElectromagneticFields as EMF
using GeometricIntegrators: GeometricIntegrator, RK4, initialstate
import GeometricIntegrators.GeometricIntegratorsBase as GIB
using ChargedParticleDynamics.ChargedParticle3d: Noncanonical

export Constants, IonSpecies, charge_to_mass, speed_from_energy, kinetic_energy
include("species.jl")

export ParticleStatus, ALIVE, HIT_SOLID, LEFT_BOX, TIME_LIMIT
export ParticleSet, set_particle!, particle_position, particle_velocity, isalive, species, nalive
export impact_position, impact_velocity
include("particles.jl")

export AbstractIonOpticsField, electric_field, magnetic_field, potential
export vector_potential, vector_potential_jacobian
export UniformElectricField, UniformMagneticField, FunctionField, FieldSum, ChargeScaledField
export GridElectrostaticField, update!, electric_field_grid
include("fields.jl")

export Electrode, ElectrodeSet, boundary_condition
include("electrodes.jl")

export AbstractWalls, NoWalls, Walls, default_walls, wall_value, wall_id, Impact, find_impact
export SurfaceModel, Absorb, interact!
include("walls.jl")

export ParticleIntegrator, Boris, GeometricIntegratorsPusher, boris_kick, trace!
include("tracing.jl")

end
