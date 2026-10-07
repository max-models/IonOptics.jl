"""
    Electrode(geometry; voltage, name = :electrode)

A solid conductor held at `voltage` [V]. Combine electrodes with `∪` into an
[`ElectrodeSet`](@ref), which a `PoissonProblem` accepts in place of a geometry and boundary
condition.
"""
struct Electrode{G <: AbstractGeometry, T}
    geometry::G
    voltage::T
    name::Symbol
end
Electrode(geometry::AbstractGeometry; voltage, name::Symbol = :electrode) =
    Electrode(geometry, float(voltage), name)

"""
    ElectrodeSet(electrodes...)

Several electrodes. Electrode `k` is solid `k` of the resulting geometry, so wall ids of particle
impacts index into `electrodes`.
"""
struct ElectrodeSet{E <: Tuple}
    electrodes::E
end
ElectrodeSet(e::Electrode...) = ElectrodeSet(e)

Base.union(a::Electrode, b::Electrode) = ElectrodeSet(a, b)
Base.union(a::ElectrodeSet, b::Electrode) = ElectrodeSet((a.electrodes..., b))
Base.union(a::Electrode, b::ElectrodeSet) = ElectrodeSet((a, b.electrodes...))
Base.length(s::ElectrodeSet) = length(s.electrodes)
Base.getindex(s::ElectrodeSet, i) = s.electrodes[i]

"""
    geometry(electrodes)

The labelled geometry of an [`ElectrodeSet`](@ref), as EmbeddedBoundaryFD `Solids`.
"""
geometry(s::ElectrodeSet) = Solids(map(e -> e.geometry, s.electrodes))

"""
    boundary_condition(electrodes)

The Dirichlet condition holding each electrode at its voltage.
"""
boundary_condition(s::ElectrodeSet) = DirichletBoundary(map(e -> e.voltage, s.electrodes))

"""
    PoissonProblem(grid, electrodes::ElectrodeSet, source; kwargs...)

The Poisson problem with the electrodes as embedded Dirichlet solids. Other keywords are those
of EmbeddedBoundaryFD's `PoissonProblem` (`box`, `default_face`, `cut`, `θmin`).
"""
EmbeddedBoundaryFD.PoissonProblem(grid::StructuredGrid, s::ElectrodeSet, source; kwargs...) =
    PoissonProblem(grid, geometry(s), source; boundary = boundary_condition(s), kwargs...)
EmbeddedBoundaryFD.PoissonProblem(grid::StructuredGrid, e::Electrode, source; kwargs...) =
    PoissonProblem(grid, ElectrodeSet(e), source; kwargs...)
