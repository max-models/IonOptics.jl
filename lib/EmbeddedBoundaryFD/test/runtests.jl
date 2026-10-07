using EmbeddedBoundaryFD
using LinearAlgebra
using SparseArrays
using StaticArrays
using Test

const RESOLUTIONS = (32, 64, 128, 256)

rates(e) = log2.(e[1:(end - 1)] ./ e[2:end])

nodal(f, g) = [f(node_position(g, I)) for I in CartesianIndices(g)]

"""
Max and discrete L² error over domain nodes of the solution of `problem` against `exact`.
"""
function errors(phi, exact, p)
    g = p.grid
    m = domain_mask(p.eb)
    e = abs.(phi .- nodal(exact, g)) .* m
    linf = maximum(e)
    l2 = sqrt(sum(e .^ 2) * prod(spacing(g)))
    linf, l2
end

function convergence(make_problem, exact; resolutions = RESOLUTIONS)
    linf = Float64[]
    l2 = Float64[]
    for N in resolutions
        p = make_problem(N)
        phi = solve(p)
        a, b = errors(phi, exact, p)
        push!(linf, a)
        push!(l2, b)
    end
    linf, l2
end

@testset "EmbeddedBoundaryFD" begin
    @testset "grid" begin
        g = CartesianGrid(0.0, 2.0, 4, -1.0, 1.0, 8)
        @test size(g) == (5, 9)
        @test spacing(g) == (0.5, 0.25)
        x, y = coordinates(g)
        @test x == 0.0:0.5:2.0
        @test y == -1.0:0.25:1.0
        xc, yc = cell_centers(g)
        @test xc ≈ 0.25:0.5:1.75
        @test length(yc) == 8
        @test node_position(g, CartesianIndex(2, 3)) == SVector(0.5, -0.5)
        I, ξ = locate(g, (0.6, -0.4))
        @test I == CartesianIndex(2, 3)
        @test ξ ≈ [0.2, 0.4]
        @test sum(node_volumes(g)) ≈ 4.0

        a = AxisymmetricGrid(z = (0.0, 1.0, 10), r = (0.0, 0.5, 5))
        @test size(a) == (11, 6)
        @test has_axis(a)
        @test on_axis(a, CartesianIndex(3, 1))
        @test sum(node_volumes(a)) ≈ π * 0.5^2 * 1.0
        @test_throws ArgumentError AxisymmetricGrid(z = (0, 1, 4), r = (-1, 1, 4))
    end

    @testset "geometry" begin
        c = Circle((0.0, 0.0), 1.0)
        @test value(c, (0.0, 0.0)) ≈ 1.0
        @test value(c, (2.0, 0.0)) ≈ -1.0
        @test !inside(c, (0.5, 0.0))
        @test inside(c, (1.5, 0.0))

        r = Rectangle((0.0, 0.0), (2.0, 1.0))
        @test value(r, (1.0, 0.5)) ≈ 0.5
        @test value(r, (3.0, 0.5)) ≈ -1.0
        @test value(r, (3.0, 2.0)) ≈ -√2
        @test value(r, (2.0, 0.5)) ≈ 0.0

        h = HalfSpace((1.0, 0.0), (1.0, 0.0))     # solid is x < 1
        @test value(h, (0.0, 5.0)) ≈ 1.0
        @test value(h, (3.0, 5.0)) ≈ -2.0

        # electrode with an aperture
        plate = setdiff(Rectangle((0.0, -1.0), (0.1, 1.0)), Circle((0.05, 0.0), 0.3))
        @test plate isa CSGDifference
        @test value(plate, (0.05, 0.0)) < 0          # in the aperture
        @test value(plate, (0.05, 0.8)) > 0          # in the plate
        @test value(c ∪ r, (1.5, 0.5)) > 0
        @test value(c ∩ r, (1.5, 0.5)) < 0
        @test value(c ∩ r, (0.5, 0.5)) > 0
        @test value(Complement(c), (0.0, 0.0)) ≈ -1.0

        s = Solids(Circle((0.0, 0.0), 0.5), Rectangle((2.0, 2.0), (3.0, 3.0)))
        @test nsolids(s) == 2
        @test solid_id(s, (0.1, 0.0)) == 1
        @test solid_id(s, (2.5, 2.4)) == 2
        @test value(s, (1.0, 1.0)) < 0

        @test @allocated(value(plate, SVector(0.3, 0.2))) == 0
    end

    @testset "segment_crossing" begin
        f(t) = t^3 - 0.2
        t = segment_crossing(f, f(0.0), f(1.0))
        @test t ≈ 0.2^(1 / 3) atol = 1e-12
        @test segment_crossing(t -> t - 1.0, -1.0, 0.0) == 1.0
    end

    @testset "classification" begin
        g = CartesianGrid(0.0, 1.0, 10, 0.0, 1.0, 10)
        # a vertical wall at x = 0.55: halfway between nodes 6 and 7
        eb = EmbeddedBoundary(g, HalfSpace((0.55, 0.0), (-1.0, 0.0)))
        @test eb.kind[6, 5] == BOUNDARY_ADJACENT
        @test eb.kind[7, 5] == SOLID
        @test eb.kind[3, 5] == INTERIOR
        @test theta_xp(eb)[6, 5] ≈ 0.5
        @test theta_xm(eb)[6, 5] == 1.0
        @test eb.cut_id[side_index(1, +1), 6, 5] == 1
        @test cut_point(eb, CartesianIndex(6, 5), side_index(1, +1)) ≈ [0.55, 0.4]

        # the wall exactly through a node column: that column is solid (ψ = 0)
        eb = EmbeddedBoundary(g, HalfSpace((0.5, 0.0), (-1.0, 0.0)))
        @test eb.kind[6, 5] == SOLID
        @test theta_xp(eb)[5, 5] ≈ 1.0
        @test eb.kind[5, 5] == BOUNDARY_ADJACENT

        # the wall a hair away from a node: that node is snapped onto the boundary
        eb = EmbeddedBoundary(g, HalfSpace((0.5 + 1e-9, 0.0), (-1.0, 0.0)))
        @test eb.kind[6, 5] == ON_BOUNDARY
        eb = EmbeddedBoundary(g, HalfSpace((0.5 + 1e-9, 0.0), (-1.0, 0.0)); θmin = 1e-10)
        @test eb.kind[6, 5] == BOUNDARY_ADJACENT

        # refined cuts are exact on a circle, linear ones are O(h²)
        circ = Circle((0.5, 0.5), 0.27)
        ebl = EmbeddedBoundary(g, circ)
        ebr = EmbeddedBoundary(g, circ; cut = RefinedCut())
        maxdev(eb) = maximum(abs(value(circ, cut_point(eb, I, k)))
        for I in CartesianIndices(g), k in 1:4 if eb.cut_id[k, I] != 0)
        @test maxdev(ebr) < 1e-12
        @test 1e-6 < maxdev(ebl) < 0.1 * 0.1
    end

    @testset "boundary conditions" begin
        g = CartesianGrid(0, 1, 4, 0, 1, 4)
        f = face_conditions(g, (xmin = Neumann(),))
        @test f[1] isa Neumann
        @test all(f[k] isa Dirichlet for k in 2:4)
        @test_throws ArgumentError face_conditions(g, (zmin = Neumann(),))
        a = AxisymmetricGrid(z = (0, 1, 4), r = (0, 1, 4))
        f = face_conditions(a, (zmin = Neumann(),))
        @test f[3] isa Axis
        @test_throws ArgumentError face_conditions(a, (rmin = Neumann(),))
    end

    @testset "assembly" begin
        # on a plain square with Dirichlet faces the matrix is the 5-point Laplacian
        g = CartesianGrid(0, 1, 8, 0, 1, 8)
        p = PoissonProblem(g, 1.0)
        A, b = assemble(p)
        @test size(A) == (49, 49)
        @test issymmetric(A)
        @test all(diag(A) .≈ 4 / (1 / 8)^2)
        @test b ≈ ones(49)

        # the boundary part of the RHS separates from the source
        p = PoissonProblem(g, Circle((0.5, 0.5), 0.2), 0.0; boundary = DirichletBoundary(3.0))
        A, bΓ = assemble_operator(p)
        A2, b2 = assemble(p)
        @test A == A2
        @test bΓ == b2
        @test any(bΓ .> 0)
        # the constant is the exact solution of Laplace's equation with constant data
        p = PoissonProblem(g, Circle((0.5, 0.5), 0.2), 0.0;
            boundary = DirichletBoundary(3.0), default_face = Dirichlet(3.0))
        @test solve(p) ≈ fill(3.0, size(g))
    end

    @testset "manufactured solutions, Cartesian" begin
        u(x) = sin(π * x[1]) * sin(π * x[2])
        f(x) = 2π^2 * u(x)
        bc = DirichletBoundary((x, id) -> u(x))

        cases = (
            "square" => NoSolid(),
            "circular solid" => Circle((0.5, 0.5), 0.2),
            "domain inside a circle" => Complement(Circle((0.5, 0.5), 0.41)),
            "two disconnected solids" => Solids(Circle((0.3, 0.3), 0.12),
                Rectangle((0.55, 0.5), (0.8, 0.77))),
            "plate with aperture" => setdiff(Rectangle((0.4, 0.0), (0.53, 1.0)),
                Circle((0.47, 0.5), 0.23)))
        for (name, geo) in cases, cut in (LinearCut(), RefinedCut())
            @testset "$name, $(nameof(typeof(cut)))" begin
                linf, l2 = convergence(u) do N
                    PoissonProblem(CartesianGrid(0, 1, N, 0, 1, N), geo, f;
                        boundary = bc, default_face = Dirichlet(u), cut)
                end
                @test all(rates(linf) .> 1.8)
                @test all(rates(l2) .> 1.8)
                @test linf[end] < 2e-5
            end
        end

        @testset "Neumann faces" begin
            v(x) = cos(π * x[1]) * sin(π * x[2])
            linf, l2 = convergence(v) do N
                PoissonProblem(CartesianGrid(0, 1, N, 0, 1, N), Circle((0.5, 0.5), 0.2),
                    x -> 2π^2 * v(x); boundary = DirichletBoundary((x, id) -> v(x)),
                    box = (xmin = Neumann(), xmax = Neumann()), default_face = Dirichlet(v))
            end
            @test all(rates(linf) .> 1.8)
        end
    end

    @testset "pathological cut distances" begin
        u(x) = sin(π * x[1]) * sin(π * x[2])
        f(x) = 2π^2 * u(x)
        N = 64
        h = 1 / N
        g = CartesianGrid(0, 1, N, 0, 1, N)
        bc = DirichletBoundary((x, id) -> u(x))
        # a circle passing exactly through nodes, and one passing a tiny distance away
        for δ in (0.0, 1e-14, 1e-9, 1e-7, 1e-5, 1e-3)
            geo = Circle((0.5, 0.5), 10h + δ * h)
            p = PoissonProblem(g, geo, f; boundary = bc, default_face = Dirichlet(u))
            phi = solve(p)
            @test all(isfinite, phi)
            linf, _ = errors(phi, u, p)
            @test linf < 1e-3
        end
        # a cut that is small but above θmin keeps the stencil, and the system stays solvable
        geo = Circle((0.5, 0.5), 10h + 1e-5 * h)
        p = PoissonProblem(g, geo, f; boundary = bc, default_face = Dirichlet(u), θmin = 1e-8)
        @test count(==(ON_BOUNDARY), p.eb.kind) == 0
        @test errors(solve(p), u, p)[1] < 1e-3
    end

    @testset "electrodes at different potentials (coaxial)" begin
        # φ = V₁ + (V₂ - V₁) ln(r/a) / ln(b/a) between an inner cylinder r = a at V₁ and an outer
        # one r = b at V₂
        a, b, V₁, V₂ = 0.1, 0.45, 3.0, -2.0
        c = (0.5, 0.5)
        exact(x) = (r = hypot(x[1] - c[1], x[2] - c[2]);
        V₁ + (V₂ - V₁) * log(r / a) / log(b / a))
        geo = Solids(Circle(c, a), Complement(Circle(c, b)))
        # the inner electrode is small (10 h at N = 64), so start beyond the pre-asymptotic range
        linf, l2 = convergence(exact; resolutions = (64, 128, 256)) do N
            PoissonProblem(CartesianGrid(0, 1, N, 0, 1, N), geo, 0.0;
                boundary = DirichletBoundary((V₁, V₂)), cut = RefinedCut())
        end
        @test all(rates(linf) .> 1.9)
        @test linf[end] < 2e-4

        # the solids carry their own potentials in the returned array
        g = CartesianGrid(0, 1, 32, 0, 1, 32)
        p = PoissonProblem(g, geo, 0.0; boundary = DirichletBoundary((V₁, V₂)))
        phi = solve(p)
        @test phi[17, 17] == V₁
        @test phi[1, 1] == V₂
    end

    @testset "manufactured solutions, axisymmetric" begin
        u(x) = sin(π * x[1]) * cos(π * x[2])
        f(x) = (z = x[1]; r = x[2];
        2π^2 * u(x) + (iszero(r) ? π^2 * sin(π * z) : π * sin(π * z) * sin(π * r) / r))
        bc = DirichletBoundary((x, id) -> u(x))
        cases = ("cylinder" => NoSolid(),
            "sphere on axis" => Circle((0.5, 0.0), 0.3),
            "torus" => Circle((0.5, 0.6), 0.2),
            "aperture plate" => setdiff(Rectangle((0.4, 0.0), (0.53, 1.0)),
                Rectangle((0.3, 0.0), (0.6, 0.27))))
        for (name, geo) in cases
            @testset "$name" begin
                linf, l2 = convergence(u) do N
                    PoissonProblem(AxisymmetricGrid(z = (0, 1, N), r = (0, 1, N)), geo, f;
                        boundary = bc, default_face = Dirichlet(u))
                end
                @test all(rates(linf) .> 1.8)
                @test all(rates(l2) .> 1.8)
            end
        end

        @testset "concentric spheres" begin
            a, b, V₁, V₂ = 0.1, 0.45, 1.0, 0.0
            exact(x) = (ρ = hypot(x[1] - 0.5, x[2]);
            V₂ + (V₁ - V₂) * (1 / ρ - 1 / b) / (1 / a - 1 / b))
            geo = Solids(Circle((0.5, 0.0), a), Complement(Circle((0.5, 0.0), b)))
            linf, _ = convergence(exact) do N
                PoissonProblem(AxisymmetricGrid(z = (0, 1, N), r = (0, 0.5, N ÷ 2)), geo, 0.0;
                    boundary = DirichletBoundary((V₁, V₂)), cut = RefinedCut())
            end
            @test all(rates(linf) .> 1.7)
        end
    end

    @testset "gradient" begin
        for (grid, u, du, f) in (
            (N -> CartesianGrid(0, 1, N, 0, 1, N),
                x -> sin(π * x[1]) * sin(π * x[2]),
                x -> SVector(π * cos(π * x[1]) * sin(π * x[2]), π * sin(π * x[1]) * cos(π * x[2])),
                x -> 2π^2 * sin(π * x[1]) * sin(π * x[2])),
            (N -> AxisymmetricGrid(z = (0, 1, N), r = (0, 1, N)),
                x -> sin(π * x[1]) * cos(π * x[2]),
                x -> SVector(π * cos(π * x[1]) * cos(π * x[2]), -π * sin(π * x[1]) * sin(π * x[2])),
                x -> (z = x[1]; r = x[2];
                2π^2 * sin(π * z) * cos(π * r) +
                (iszero(r) ? π^2 * sin(π * z) : π * sin(π * z) * sin(π * r) / r))))
            errs = Float64[]
            for N in RESOLUTIONS
                g = grid(N)
                p = PoissonProblem(g, Circle((0.5, 0.45), 0.2), f;
                    boundary = DirichletBoundary((x, id) -> u(x)), default_face = Dirichlet(u))
                phi = solve(p)
                G = gradient(phi, p)
                m = domain_mask(p.eb)
                e = maximum(norm(SVector(G[1][I], G[2][I]) - du(node_position(g, I)))
                for I in CartesianIndices(g) if m[I])
                push!(errs, e)
            end
            @test all(rates(errs) .> 1.7)
        end
    end

    @testset "interpolation" begin
        g = CartesianGrid(0, 1, 10, 0, 2, 10)
        lin(x) = 1 + 2x[1] - 3x[2] + x[1] * x[2]
        data = nodal(lin, g)
        for x in (SVector(0.33, 1.71), SVector(0.0, 0.0), SVector(1.0, 2.0))
            @test interpolate(g, data, x) ≈ lin(x)
        end
        mask = trues(size(g))
        @test interpolate(g, data, mask, SVector(0.33, 1.71))[1] ≈ lin(SVector(0.33, 1.71))
        # masked corners drop out, the rest are renormalised
        mask[4, 9] = false
        x = SVector(0.33, 1.71)
        v, w = interpolate(g, data, mask, x)
        @test w < 1
        @test isfinite(v)
        @test @allocated(interpolate(g, data, mask, x)) == 0
        # ghost values extrapolated into a solid make interpolation in cut cells exact for
        # linear data, without reading the (meaningless) solid values
        eb = EmbeddedBoundary(g, Circle((0.5, 1.0), 0.33))
        gm = ghost_mask(eb)
        @test any(gm)
        @test !any(gm .& domain_mask(eb))
        m = domain_mask(eb) .| gm
        affine(x) = 1 + 2x[1] - 3x[2]
        data_aff = nodal(affine, g)
        data_aff[.!domain_mask(eb)] .= 1e6
        extrapolate_ghosts!(data_aff, eb)
        @test all(abs.(data_aff .- nodal(affine, g))[gm] .< 1e-12)
        for x in (SVector(0.5, 1.0 - 0.34), SVector(0.5 + 0.335, 1.0), SVector(0.27, 0.75))
            if value(eb.geometry, x) < 0
                @test interpolate(g, data_aff, m, x)[1] ≈ affine(x)
            end
        end
        # a constant is reproduced exactly regardless of the mask
        v, _ = interpolate(g, fill(2.5, size(g)), mask, x)
        @test v ≈ 2.5
    end
end
