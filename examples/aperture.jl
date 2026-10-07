# Milestone 1 end to end, without space charge: an axisymmetric two-electrode extraction gap,
# the Shortley–Weller Laplace solve, the numerical field, and protons traced through it until
# they hit an electrode or leave the box.
#
#     julia --project=. examples/aperture.jl

using IonOptics
using IonOptics.Constants: e, mₚ
using StaticArrays

# (z, r) grid: z ∈ [0, 20] mm, r ∈ [0, 8] mm
grid = AxisymmetricGrid(z = (0.0, 20e-3, 200), r = (0.0, 8e-3, 80))

# plasma electrode at +10 kV with a 2 mm radius aperture, puller at ground with 2.5 mm aperture
plasma = Electrode(Rectangle((2e-3, 2e-3), (3e-3, 8e-3)); voltage = 10e3, name = :plasma)
puller = Electrode(Rectangle((9e-3, 2.5e-3), (11e-3, 8e-3)); voltage = 0.0, name = :puller)
electrodes = plasma ∪ puller

# zmin at the plasma potential, zmax grounded, rmax Neumann (open)
problem = PoissonProblem(grid, electrodes, 0.0;
    box = (zmin = Dirichlet(10e3), zmax = Dirichlet(0.0), rmax = Neumann()),
    cut = RefinedCut())
phi = solve(problem)
field = GridElectrostaticField(problem, phi)

println("grid: ", grid)
println("node classification: ", count_kinds(embedded_boundary(problem)))
println("E on axis at z = 6 mm: ", electric_field(field, SVector(6e-3, 0.0, 0.0)), " V/m")

# protons starting at 1 eV just upstream of the aperture, at radii 0 … 2.4 mm
proton = IonSpecies(name = :proton, mass = mₚ, charge = e)
radii = range(0.0, 2.4e-3; length = 13)
v₀ = speed_from_energy(proton, 1.0)
particles = ParticleSet(proton, [SVector(1e-3, r, 0.0) for r in radii],
    [SVector(v₀, 0.0, 0.0) for _ in radii])

trajectories = trace!(particles, field, (0.0, 1e-5);
    integrator = Boris(dt = 1e-9, max_displacement = 0.25 * minimum(spacing(grid))),
    record = true)

names = Dict(Int32(k) => electrodes[k].name for k in 1:length(electrodes))
facename = Dict(Int32(-1) => :zmin, Int32(-2) => :zmax, Int32(-4) => :rmax)
println("\n  r₀ [mm]   status      where     z [mm]  r [mm]   E [keV]  steps")
for i in 1:length(particles)
    x = impact_position(particles, i)
    where = get(names, particles.impact_id[i], get(facename, particles.impact_id[i], :none))
    E = kinetic_energy(proton, sqrt(sum(abs2, impact_velocity(particles, i)))) / 1e3
    println(rpad(round(radii[i] * 1e3; digits = 2), 9), " ", rpad(particles.status[i], 11), " ",
        rpad(where, 8), " ", rpad(round(x[1] * 1e3; digits = 2), 7), " ",
        rpad(round(hypot(x[2], x[3]) * 1e3; digits = 3), 8), " ",
        rpad(round(E; digits = 3), 8), " ", length(trajectories[i]) - 1)
end
println("\ntransmitted: ", count(==(Int32(-2)), particles.impact_id), " of ", length(particles))

# energy bookkeeping: kinetic energy gained = q (φ(start) - φ(end))
for i in (2, 7, 13)
    x₀ = SVector(1e-3, radii[i], 0.0)
    gain = potential(field, x₀) - potential(field, impact_position(particles, i))
    E = kinetic_energy(proton, sqrt(sum(abs2, impact_velocity(particles, i))))
    println("r₀ = ", round(radii[i] * 1e3; digits = 2), " mm: φ(start) = ",
        round(potential(field, x₀); digits = 2), " V, relative energy error ",
        (E - 1.0 - gain) / gain)
end
