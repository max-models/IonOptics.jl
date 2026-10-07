using IonOptics
using IonOptics.Constants: e, mₚ
using LinearAlgebra
using StaticArrays
using Test

import ElectromagneticFields as EMF
using GeometricIntegrators: RK4, Strang
using ChargedParticleDynamics.ChargedParticle3d: Noncanonical

const proton = IonSpecies(name = :proton, mass = mₚ, charge = e)

rates(err) = log2.(err[1:(end - 1)] ./ err[2:end])

@testset "IonOptics" begin
    @testset "species and particles" begin
        @test charge_to_mass(proton) ≈ e / mₚ
        @test kinetic_energy(proton, speed_from_energy(proton, 1000.0)) ≈ 1000.0
        p = ParticleSet(proton, [SVector(1.0, 2.0, 3.0)], [SVector(4.0, 5.0, 6.0)]; weight = 7.0)
        @test length(p) == 1
        @test particle_position(p, 1) == SVector(1.0, 2.0, 3.0)
        @test particle_velocity(p, 1) == SVector(4.0, 5.0, 6.0)
        @test p.weight[1] == 7.0
        @test isalive(p, 1)
        @test species(p, 1) === proton
        @test nalive(p) == 1
    end

    @testset "ElectromagneticFields interface" begin
        f = UniformElectricField((1.0, 2.0, 3.0)) + UniformMagneticField((0.0, 0.0, 2.0))
        x = [0.5, -1.0, 2.0]
        @test EMF.E♭(f, 0.0, x) == SVector(1.0, 2.0, 3.0)
        @test EMF.B♭(f, 0.0, x) == SVector(0.0, 0.0, 2.0)
        @test EMF.φ(f, 0.0, x) ≈ -dot([1.0, 2.0, 3.0], x)
        # curl A = B, from the Jacobian ∂ⱼAᵢ
        D = EMF.DA♭(f, 0.0, x)
        @test SVector(D[3, 2] - D[2, 3], D[1, 3] - D[3, 1], D[2, 1] - D[1, 2]) ≈ [0, 0, 2]
        @test EMF.A♭(f, 0.0, x) ≈ cross([0, 0, 2.0], x) / 2
        # a ChargedParticleDynamics model accepts the field as it is
        prob = Noncanonical.odeproblem([x; 1.0; 0.0; 0.0]; timespan = (0.0, 1.0),
            timestep = 0.1, parameters = (field = f,), periodic = false)
        v = zeros(6)
        prob.equation.v(v, 0.0, [x; 1.0; 0.0; 0.0], (field = f,))
        @test v ≈ [1, 0, 0, 1, 2 - 2, 3]        # E + v × B with v = x̂, B = 2ẑ
    end

    @testset "electrodes" begin
        a = Electrode(Circle((0.0, 0.0), 0.1); voltage = 5.0, name = :inner)
        b = Electrode(Rectangle((1.0, 1.0), (2.0, 2.0)); voltage = -3.0)
        s = a ∪ b
        @test s isa ElectrodeSet
        @test length(s) == 2
        @test s[1].name == :inner
        g = IonOptics.geometry(s)
        @test solid_id(g, (1.5, 1.5)) == 2
        @test boundary_condition(s).g(SVector(0.0, 0.0), 2) == -3.0
    end

    @testset "parallel plates" begin
        # Plates at x = a (0 V) and x = b (V₀), placed off the grid nodes; symmetric y faces.
        # φ = V₀ (x - a) / (b - a) is linear, which the Shortley–Weller stencil and the
        # boundary-aware gradient reproduce exactly.
        a, b, V₀ = 0.0123, 0.0871, -1000.0
        grid = CartesianGrid(0.0, 0.1, 40, 0.0, 0.05, 20)
        electrodes = Electrode(HalfSpace((a, 0.0), (1.0, 0.0)); voltage = 0.0) ∪
                     Electrode(HalfSpace((b, 0.0), (-1.0, 0.0)); voltage = V₀)
        problem = PoissonProblem(grid, electrodes, 0.0; box = (ymin = Neumann(), ymax = Neumann()))
        phi = solve(problem)
        field = GridElectrostaticField(problem, phi)
        E₀ = -V₀ / (b - a)
        exact(x) = V₀ * (x[1] - a) / (b - a)
        m = domain_mask(problem.eb)
        @test maximum(abs.(phi .- [exact(node_position(grid, I)) for I in CartesianIndices(grid)]) .* m) <
              1e-9
        for x in (SVector(0.03, 0.01, 0.0), SVector(0.0125, 0.0, 1.0), SVector(0.087, 0.05, 0.0))
            @test electric_field(field, x) ≈ SVector(E₀, 0.0, 0.0) rtol = 1e-9
            @test potential(field, x) ≈ exact(x) atol = 1e-9
        end
        @test magnetic_field(field, SVector(0.03, 0.01, 0.0)) == zero(SVector{3})

        # a proton starting at rest next to the 0 V plate is accelerated across the gap
        x₀ = a + 1e-4
        for integrator in (Boris(dt = 1e-10), Boris(dt = 1e-8, max_displacement = 1e-4),
            GeometricIntegratorsPusher(RK4(); dt = 1e-10),
            GeometricIntegratorsPusher(Strang(); dt = 1e-10, formulation = :sode))
            p = ParticleSet(proton, [SVector(x₀, 0.02, 0.0)], [SVector(0.0, 0.0, 0.0)])
            trace!(p, field, (0.0, 1e-5); integrator)
            @test p.status[1] == HIT_SOLID
            @test p.impact_id[1] == 2
            @test impact_position(p, 1)[1] ≈ b rtol = 1e-9
            @test impact_position(p, 1)[2] ≈ 0.02
            # energy gain q ΔV and time of flight from uniform acceleration
            ΔV = exact(SVector(b, 0.0)) - exact(SVector(x₀, 0.0))
            @test kinetic_energy(proton, norm(impact_velocity(p, 1))) ≈ -ΔV rtol = 1e-8
            acc = e * E₀ / mₚ
            @test p.t[1] ≈ sqrt(2 * (b - x₀) / acc) rtol = 1e-8
        end
    end

    @testset "field of a charged cylinder (coax)" begin
        # E = (V₁ - V₂) / ln(b/a) / r, radial
        a, b, V₁, V₂ = 0.1, 0.45, 100.0, 0.0
        c = SVector(0.5, 0.5)
        Eexact(x) = (d = SVector(x[1], x[2]) - c; r = norm(d);
        (V₁ - V₂) / log(b / a) / r * d / r)
        electrodes = Electrode(Circle(c, a); voltage = V₁) ∪
                     Electrode(Complement(Circle(c, b)); voltage = V₂)
        # interior points (≥ 2h from both electrodes at N = 64) and points in cut cells
        probes_interior = [c + 0.27 * SVector(cos(θ), sin(θ)) for θ in range(0, 2π; length = 13)]
        probes_wall = [c + (a + 0.003) * SVector(cos(θ), sin(θ)) for θ in range(0.1, 2π; length = 13)]
        errs_i = Float64[]
        errs_w = Float64[]
        for N in (64, 128, 256)
            grid = CartesianGrid(0.0, 1.0, N, 0.0, 1.0, N)
            problem = PoissonProblem(grid, electrodes, 0.0; cut = RefinedCut())
            field = GridElectrostaticField(problem, solve(problem))
            err(ps) = maximum(norm(electric_field(field, SVector(x[1], x[2], 0.0))[1:2] -
                                   Eexact(x)) / norm(Eexact(x)) for x in ps)
            push!(errs_i, err(probes_interior))
            push!(errs_w, err(probes_wall))
        end
        # bilinear interpolation is second order in the open and, with ghost values in the
        # electrode, also in the cells cut by it
        @test all(rates(errs_i) .> 1.8)
        @test all(rates(errs_w) .> 1.5)
        @test errs_w[end] < 2e-3

        # a particle orbiting in the numerical field conserves energy up to discretisation
        grid = CartesianGrid(0.0, 1.0, 128, 0.0, 1.0, 128)
        problem = PoissonProblem(grid, electrodes, 0.0; cut = RefinedCut())
        field = GridElectrostaticField(problem, solve(problem))
        electron = IonSpecies(name = :e, mass = 9.1093837015e-31, charge = -e)
        r₀ = 0.25
        # circular orbit speed in E ∝ 1/r: m v²/r = |q| E(r)
        v₀ = sqrt(e * norm(Eexact(c + SVector(r₀, 0.0))) * r₀ / electron.mass)
        p = ParticleSet(electron, [SVector(c[1] + r₀, c[2], 0.0)], [SVector(0.0, v₀, 0.0)])
        T = 2π * r₀ / v₀
        traj = trace!(p, field, (0.0, 3T); integrator = Boris(dt = T / 2000), record = true)
        @test p.status[1] == ALIVE
        radii = [norm(x[1:2] - c) for x in traj[1]]
        @test maximum(abs.(radii .- r₀)) / r₀ < 0.01
        energy(x, v) = electron.mass * dot(v, v) / 2 + electron.charge * potential(field, x)
        E0 = electron.mass * v₀^2 / 2 + electron.charge * potential(field, SVector(c[1] + r₀, c[2], 0.0))
        @test abs(energy(particle_position(p, 1), particle_velocity(p, 1)) - E0) / abs(E0) < 1e-3
    end

    @testset "wall impacts" begin
        grid = CartesianGrid(0.0, 1.0, 32, 0.0, 1.0, 32)
        circ = Circle((0.6, 0.5), 0.2)
        walls = Walls(grid, Solids(circ, Rectangle((0.0, 0.9), (0.2, 1.0))))
        # a straight line into the circle, through a field-free region, with steps much longer
        # than the distance to the wall: the impact is located on the circle all the same
        field = FunctionField()
        sp = IonSpecies(name = :x, mass = 1.0, charge = 1.0)
        for dt in (1e-3, 0.05, 0.3)
            p = ParticleSet(sp, [SVector(0.1, 0.5, 0.0)], [SVector(1.0, 0.0, 0.0)])
            trace!(p, field, (0.0, 10.0); integrator = Boris(; dt), walls)
            @test p.status[1] == HIT_SOLID
            @test p.impact_id[1] == 1
            @test impact_position(p, 1) ≈ SVector(0.4, 0.5, 0.0) atol = 1e-9
            @test p.t[1] ≈ 0.3 atol = 1e-9
        end
        # second solid
        p = ParticleSet(sp, [SVector(0.1, 0.5, 0.0)], [SVector(0.0, 1.0, 0.0)])
        trace!(p, field, (0.0, 10.0); integrator = Boris(dt = 0.07), walls)
        @test p.impact_id[1] == 2
        @test impact_position(p, 1)[2] ≈ 0.9 atol = 1e-9
        # leaving the box through ymin (face 3) and xmax (face 2)
        p = ParticleSet(sp, [SVector(0.1, 0.5, 0.0), SVector(0.9, 0.8, 0.0)],
            [SVector(0.0, -1.0, 0.0), SVector(1.0, 0.0, 0.0)])
        trace!(p, field, (0.0, 10.0); integrator = Boris(dt = 0.07), walls)
        @test all(p.status .== LEFT_BOX)
        @test p.impact_id == Int32[-3, -2]
        @test impact_position(p, 1) ≈ SVector(0.1, 0.0, 0.0) atol = 1e-9
        @test impact_position(p, 2) ≈ SVector(1.0, 0.8, 0.0) atol = 1e-9
        # starting inside a solid stops the particle immediately
        p = ParticleSet(sp, [SVector(0.6, 0.5, 0.0)], [SVector(1.0, 0.0, 0.0)])
        trace!(p, field, (0.0, 10.0); integrator = Boris(dt = 0.07), walls)
        @test p.status[1] == HIT_SOLID
        @test p.t[1] == 0.0
        # running out of steps
        p = ParticleSet(sp, [SVector(0.1, 0.5, 0.0)], [SVector(1.0, 0.0, 0.0)])
        trace!(p, field, (0.0, 10.0); integrator = Boris(dt = 1e-3), walls, maxsteps = 10)
        @test p.status[1] == TIME_LIMIT
        @test p.t[1] ≈ 0.01
        # still in flight at the end of the time span
        p = ParticleSet(sp, [SVector(0.1, 0.5, 0.0)], [SVector(1.0, 0.0, 0.0)])
        trace!(p, field, (0.0, 0.1); integrator = Boris(dt = 1e-3), walls)
        @test p.status[1] == ALIVE
        @test particle_position(p, 1) ≈ SVector(0.2, 0.5, 0.0)
    end

    @testset "native Boris = GeometricIntegrators Strang on the CPD splitting" begin
        grid = CartesianGrid(0.0, 1.0, 64, 0.0, 1.0, 64)
        electrodes = Electrode(Circle((0.5, 0.5), 0.1); voltage = 1.0) ∪
                     Electrode(Rectangle((0.0, 0.0), (1.0, 0.05)); voltage = -1.0)
        problem = PoissonProblem(grid, electrodes, 0.0)
        field = GridElectrostaticField(problem, solve(problem)) +
                UniformMagneticField((0.0, 0.3, 2.0))
        sp = IonSpecies(name = :x, mass = 1.0, charge = 1.0)
        x₀, v₀ = SVector(0.2, 0.7, 0.0), SVector(0.1, -0.05, 0.02)
        p1 = ParticleSet(sp, [x₀], [v₀])
        p2 = ParticleSet(sp, [x₀], [v₀])
        trace!(p1, field, (0.0, 5.0); integrator = Boris(dt = 1e-3))
        trace!(p2, field, (0.0, 5.0);
            integrator = GeometricIntegratorsPusher(Strang(); dt = 1e-3, formulation = :sode))
        @test p1.status == p2.status
        @test particle_position(p1, 1) ≈ particle_position(p2, 1) rtol = 1e-10
        @test particle_velocity(p1, 1) ≈ particle_velocity(p2, 1) rtol = 1e-10
        # and RK4 on the vector field agrees to its own accuracy
        p3 = ParticleSet(sp, [x₀], [v₀])
        trace!(p3, field, (0.0, 5.0); integrator = GeometricIntegratorsPusher(RK4(); dt = 1e-3))
        @test particle_position(p1, 1) ≈ particle_position(p3, 1) rtol = 1e-4
    end

    @testset "axisymmetric field" begin
        # a sphere of radius a at V₁ inside one of radius b at V₂, centred on the axis:
        # φ = V₂ + (V₁ - V₂)(1/ρ - 1/b)/(1/a - 1/b), E radial in 3D
        a, b, V₁, V₂ = 0.1, 0.45, 50.0, 0.0
        zc = 0.5
        grid = AxisymmetricGrid(z = (0.0, 1.0, 128), r = (0.0, 0.5, 64))
        electrodes = Electrode(Circle((zc, 0.0), a); voltage = V₁) ∪
                     Electrode(Complement(Circle((zc, 0.0), b)); voltage = V₂)
        problem = PoissonProblem(grid, electrodes, 0.0; cut = RefinedCut())
        field = GridElectrostaticField(problem, solve(problem))
        K = (V₁ - V₂) / (1 / a - 1 / b)
        Eexact(x) = (d = x - SVector(zc, 0.0, 0.0); K * d / norm(d)^3)
        for x in (SVector(zc + 0.2, 0.1, -0.05), SVector(zc - 0.15, 0.0, 0.0),
            SVector(zc, 0.0, 0.25), SVector(zc + 0.1, -0.12, 0.13))
            @test electric_field(field, x) ≈ Eexact(x) rtol = 5e-3
        end
        # the axis is not a wall: a particle starting on it, or crossing it, keeps going
        walls = default_walls(field)
        @test wall_value(walls, SVector(zc + 0.3, 0.0, 0.0)) < 0
        sp0 = IonSpecies(name = :x, mass = 1.0, charge = 1.0)
        p = ParticleSet(sp0, [SVector(zc + 0.3, 0.0, 0.0), SVector(zc + 0.3, 0.02, 0.0)],
            [SVector(0.0, 0.0, 0.0), SVector(0.0, -1.0, 0.0)])
        trace!(p, FunctionField(), (0.0, 0.03); integrator = Boris(dt = 1e-3), walls)
        @test all(p.status .== ALIVE)
        @test particle_position(p, 2) ≈ SVector(zc + 0.3, -0.01, 0.0)
        # a particle launched off-axis keeps its angular momentum about the axis (radial field)
        sp = IonSpecies(name = :x, mass = 1.0, charge = 1.0)
        p = ParticleSet(sp, [SVector(zc + 0.15, 0.1, 0.0)], [SVector(0.0, 0.0, 5.0)])
        traj = trace!(p, field, (0.0, 1.0); integrator = Boris(dt = 1e-4), record = true)
        @test p.status[1] == HIT_SOLID
        @test p.impact_id[1] == 2
        @test norm(impact_position(p, 1) - SVector(zc, 0.0, 0.0)) ≈ b atol = 1e-9
        # the force is central about (zc, 0, 0): angular momentum about that point is conserved
        L(x, v) = cross(x - SVector(zc, 0.0, 0.0), v)
        L0 = L(SVector(zc + 0.15, 0.1, 0.0), SVector(0.0, 0.0, 5.0))
        @test L(impact_position(p, 1), impact_velocity(p, 1)) ≈ L0 rtol = 0.01
    end

    @testset "allocations" begin
        grid = CartesianGrid(0.0, 1.0, 32, 0.0, 1.0, 32)
        problem = PoissonProblem(grid, Electrode(Circle((0.5, 0.5), 0.1); voltage = 1.0), 0.0)
        field = GridElectrostaticField(problem, solve(problem))
        x = SVector(0.3, 0.4, 0.0)
        electric_field(field, x)
        @test @allocated(electric_field(field, x)) == 0
        walls = default_walls(field)
        wall_value(walls, x)
        @test @allocated(wall_value(walls, x)) == 0
        sp = IonSpecies(name = :x, mass = 1.0, charge = 1.0)
        p = ParticleSet(sp, [SVector(0.1, 0.2, 0.0)], [SVector(0.01, 0.0, 0.0)])
        integrator = Boris(dt = 1e-2)
        trace!(p, field, (0.0, 1.0); integrator)
        set_particle!(p, 1, SVector(0.1, 0.2, 0.0), SVector(0.01, 0.0, 0.0))
        # 10⁴ steps without a single allocation in the kernel
        @test @allocated(IonOptics.trace_particle!(p, 1, field, walls, Absorb(), integrator,
            100.0, 10^6, nothing)) == 0
    end
end
