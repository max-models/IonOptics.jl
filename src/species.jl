"""
Physical constants in SI units (CODATA 2018).
"""
module Constants
const e = 1.602176634e-19          # elementary charge [C]
const ε₀ = 8.8541878128e-12        # vacuum permittivity [F/m]
const amu = 1.66053906660e-27      # atomic mass unit [kg]
const mₑ = 9.1093837015e-31        # electron mass [kg]
const mₚ = 1.67262192369e-27       # proton mass [kg]
const k_B = 1.380649e-23           # Boltzmann constant [J/K]
end

"""
    IonSpecies(; name, mass, charge)

A particle species: `mass` in kg, `charge` in C. Both are stored as `T` so the struct is
`isbits` apart from the name.

```julia
IonSpecies(name = :Hminus, mass = Constants.mₚ + Constants.mₑ, charge = -Constants.e)
```
"""
struct IonSpecies{T <: AbstractFloat}
    name::Symbol
    mass::T
    charge::T
end

function IonSpecies(; name::Symbol, mass, charge)
    T = float(promote_type(typeof(mass), typeof(charge)))
    mass > 0 || throw(ArgumentError("mass must be positive"))
    IonSpecies{T}(name, T(mass), T(charge))
end

charge_to_mass(s::IonSpecies) = s.charge / s.mass

Base.show(io::IO, s::IonSpecies) =
    print(io, "IonSpecies(:", s.name, ", m = ", s.mass, " kg, q = ", s.charge, " C)")

"""
    speed_from_energy(species, E)

The non-relativistic speed of a particle of `species` with kinetic energy `E` in **electron
volts** (i.e. `E` × e joules), regardless of the species' charge.
"""
speed_from_energy(s::IonSpecies, E) = sqrt(2 * E * Constants.e / s.mass)

"""
    kinetic_energy(species, v)

Kinetic energy in electron volts for speed `v`.
"""
kinetic_energy(s::IonSpecies, v) = s.mass * v^2 / 2 / Constants.e
