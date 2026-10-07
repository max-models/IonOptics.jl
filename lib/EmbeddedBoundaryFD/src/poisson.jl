#
# The Shortley–Weller discretisation of -∇²φ = f on an embedded domain.
#

"""
    PoissonProblem(grid, geometry, source; boundary, box, default_face, cut, θmin)

The problem `-∇²φ = f` on the part of `grid` outside the solids of `geometry`, with Dirichlet
data on the embedded boundary and Dirichlet or Neumann data on the faces of the bounding box. On
an [`Axisymmetric`](@ref) grid `∇²` is the axisymmetric Laplacian
`(1/r) ∂ᵣ(r ∂ᵣφ) + ∂²_z φ`, with regularity `∂ᵣφ = 0` imposed on the axis.

# Arguments
- `source`: the right-hand side `f`, an array of `size(grid)` or a function `f(x)`.
- `boundary`: a [`DirichletBoundary`](@ref) for the embedded solids (default zero).
- `box`: a `NamedTuple` of [`FaceCondition`](@ref)s for the box faces, see
  [`face_conditions`](@ref). Faces not given get `default_face` (default `Dirichlet(0.0)`).
- `cut`, `θmin`: passed to [`EmbeddedBoundary`](@ref).

The problem fixes the classification of nodes and the numbering of the unknowns; changing the
source afterwards is cheap (see [`solve!`](@ref)), changing geometry or boundary data needs a new
problem.
"""
struct PoissonProblem{N, T, G <: StructuredGrid{N, T}, EB <: EmbeddedBoundary{N, T},
    BC <: DirichletBoundary, F <: Tuple}
    grid::G
    eb::EB
    source::Array{T, N}
    boundary::BC
    faces::F
    "unknown number of each node, 0 for nodes with a prescribed value"
    dofs::Array{Int, N}
    "prescribed value of each non-unknown node (`NaN` for unknowns)"
    fixed::Array{T, N}
    "the node index of each unknown"
    nodes::Vector{CartesianIndex{N}}
end

function PoissonProblem(grid::StructuredGrid{N, T}, geometry::AbstractGeometry, source;
        boundary::DirichletBoundary = DirichletBoundary(0.0),
        box::NamedTuple = NamedTuple(), default_face::FaceCondition = Dirichlet(0.0),
        cut = LinearCut(), θmin = 1e-6) where {N, T}
    eb = EmbeddedBoundary(grid, geometry; cut, θmin)
    faces = face_conditions(grid, box; default = default_face)
    f = source_array(grid, source)

    dofs = zeros(Int, size(grid))
    fixed = fill(T(NaN), size(grid))
    nodes = CartesianIndex{N}[]
    for I in CartesianIndices(grid)
        x = node_position(grid, I)
        k = eb.kind[I]
        if k == SOLID
            fixed[I] = boundary_value(boundary, x, solid_id(geometry, x))
        elseif k == ON_BOUNDARY
            fixed[I] = boundary_value(boundary, x, eb.cut_id[nearest_cut(eb, I), I])
        elseif (c = dirichlet_face(grid, faces, I)) !== nothing
            fixed[I] = face_value(c, x)
        else
            push!(nodes, I)
            dofs[I] = length(nodes)
        end
    end

    PoissonProblem(grid, eb, f, boundary, faces, dofs, fixed, nodes)
end

PoissonProblem(grid::StructuredGrid, source; kwargs...) =
    PoissonProblem(grid, NoSolid(), source; kwargs...)

source_array(grid::StructuredGrid{N, T}, f::AbstractArray) where {N, T} =
    (size(f) == size(grid) || throw(DimensionMismatch("source must have size $(size(grid))"));
     Array{T, N}(f))
source_array(grid::StructuredGrid{N, T}, f::Number) where {N, T} = fill(T(f), size(grid))
source_array(grid::StructuredGrid{N, T}, f) where {N, T} =
    T[f(node_position(grid, I)) for I in CartesianIndices(grid)]

# The first Dirichlet face the node lies on, or `nothing`.
function dirichlet_face(grid::StructuredGrid{N}, faces, I::CartesianIndex{N}) where {N}
    for d in 1:N
        I[d] == 1 && faces[side_index(d, -1)] isa Dirichlet &&
            return faces[side_index(d, -1)]
        I[d] == grid.ncells[d] + 1 && faces[side_index(d, +1)] isa Dirichlet &&
            return faces[side_index(d, +1)]
    end
    nothing
end

ndofs(p::PoissonProblem) = length(p.nodes)
geometry(p::PoissonProblem) = p.eb.geometry
grid(p::PoissonProblem) = p.grid
embedded_boundary(p::PoissonProblem) = p.eb

# ---------------------------------------------------------------------------------------------
# One side of a node's stencil in one direction
# ---------------------------------------------------------------------------------------------

"""
    StencilSide{T}

What lies on one side of a node along one grid direction: at distance `dist`, either the unknown
number `col > 0`, or (`col == 0`) a known value `val` — a boundary cut or a node with a
prescribed value — or (`mirror == true`) a symmetry face, across which the stencil is reflected.
"""
struct StencilSide{T}
    dist::T
    col::Int
    val::T
    mirror::Bool
    cut::Bool
end

@inline function stencil_side(p::PoissonProblem{N, T}, I::CartesianIndex{N}, d::Int,
        s::Int) where {N, T}
    eb = p.eb
    k = side_index(d, s)
    h = p.grid.h[d]
    if eb.cut_id[k, I] != 0
        xΓ = cut_point(eb, I, k)
        return StencilSide(eb.theta[k, I] * h, 0,
            T(boundary_value(p.boundary, xΓ, eb.cut_id[k, I])), false, true)
    end
    J = I + CartesianIndex(ntuple(e -> e == d ? s : 0, Val(N)))
    if !checkbounds(Bool, p.dofs, J)
        ismirror(p.faces[k]) ||
            error("node $I lies on a Dirichlet face but is an unknown; this is a bug")
        return StencilSide(h, 0, zero(T), true, false)
    end
    p.dofs[J] > 0 ? StencilSide(h, p.dofs[J], zero(T), false, false) :
    StencilSide(h, 0, p.fixed[J], false, false)
end

# Resolve a symmetry face by reflecting the opposite side.
@inline function resolve_mirror(m::StencilSide, q::StencilSide, I, d)
    m.mirror && q.mirror &&
        error("node $I has symmetry faces on both sides in direction $d; grid too coarse")
    m.mirror ? q : m
end

# Metric factors of the axisymmetric radial operator (1/r) ∂ᵣ(r ∂ᵣ ·) for the two sides.
@inline radial_factors(::StructuredGrid, I, d, dm, dp) = (1, 1)
@inline function radial_factors(grid::StructuredGrid{2, T, Axisymmetric}, I::CartesianIndex{2},
        d, dm, dp) where {T}
    d == 2 || return (one(T), one(T))
    r = grid.lo[2] + (I[2] - 1) * grid.h[2]
    if iszero(r)
        # on the axis the operator tends to 2 ∂²ᵣ by l'Hôpital
        return (T(2), T(2))
    end
    ((r - dm / 2) / r, (r + dp / 2) / r)
end

# ---------------------------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------------------------

"""
    assemble(problem) -> (A, b)

The sparse matrix and right-hand side of the discrete problem `A φ = b` on the unknowns of
`problem` (see `problem.nodes` for their grid positions).

Each row is `-L φ = f` with the Shortley–Weller operator

    ∂²φ/∂x² ≈ 2/(d₋ + d₊) [ (φ₊ - φᵢ)/d₊ - (φᵢ - φ₋)/d₋ ]

per direction, where `d±` are the distances to the neighbour or the boundary cut. Known values
(boundary cuts, Dirichlet box faces) are moved to the right-hand side. On an axisymmetric grid
the radial term carries the factors `(r ± d±/2)/r` of the conservative form.
"""
function assemble(p::PoissonProblem)
    A, bΓ = assemble_operator(p)
    b = bΓ .+ view(p.source, p.nodes)
    A, b
end

"""
    assemble_operator(problem) -> (A, bΓ)

Like [`assemble`](@ref) but returns only the boundary part `bΓ` of the right-hand side, so that
`b = bΓ + f[nodes]` for any source `f`.
"""
function assemble_operator(p::PoissonProblem{N, T}) where {N, T}
    n = ndofs(p)
    rows = Int[]
    cols = Int[]
    vals = T[]
    sizehint!(rows, (2N + 1) * n)
    sizehint!(cols, (2N + 1) * n)
    sizehint!(vals, (2N + 1) * n)
    bΓ = zeros(T, n)

    for (row, I) in enumerate(p.nodes)
        diag = zero(T)
        for d in 1:N
            sm = stencil_side(p, I, d, -1)
            sp = stencil_side(p, I, d, +1)
            sm, sp = resolve_mirror(sm, sp, I, d), resolve_mirror(sp, sm, I, d)
            dm, dp = sm.dist, sp.dist
            fm, fp = radial_factors(p.grid, I, d, dm, dp)
            am = fm * 2 / (dm * (dm + dp))
            ap = fp * 2 / (dp * (dm + dp))
            diag += am + ap
            for (side, a) in ((sm, am), (sp, ap))
                if side.col > 0
                    push!(rows, row)
                    push!(cols, side.col)
                    push!(vals, -a)
                else
                    bΓ[row] += a * side.val
                end
            end
        end
        push!(rows, row)
        push!(cols, row)
        push!(vals, diag)
    end

    sparse(rows, cols, vals, n, n), bΓ
end

# ---------------------------------------------------------------------------------------------
# Linear solve
# ---------------------------------------------------------------------------------------------

"""
    solve_linear_system!(x, A, b)

Solve `A x = b`. The default uses Julia's sparse direct solver; `A` may also be a factorization.
This is the single place to swap in an iterative or external solver.
"""
function solve_linear_system!(x::AbstractVector, A, b::AbstractVector)
    x .= A \ b
    x
end

"""
    scatter!(phi, problem, x)

Write the unknowns `x` into the grid array `phi`, filling all other nodes with their prescribed
values.
"""
function scatter!(phi::AbstractArray, p::PoissonProblem, x::AbstractVector)
    size(phi) == size(p.grid) || throw(DimensionMismatch("phi must have size $(size(p.grid))"))
    phi .= p.fixed
    for (row, I) in enumerate(p.nodes)
        phi[I] = x[row]
    end
    phi
end

"""
    ShortleyWeller()

Selects the Shortley–Weller embedded-boundary discretisation, e.g. as the `poisson` option of a
higher-level simulation. See [`ShortleyWellerSolver`](@ref).
"""
struct ShortleyWeller end

"""
    ShortleyWellerSolver(problem; factorize = true)

A solver for `problem` that assembles the operator once and (by default) keeps its sparse LU
factorization, so that repeated solves with different sources cost one back-substitution each.
"""
struct ShortleyWellerSolver{P <: PoissonProblem, M, Fa, T}
    problem::P
    A::M
    factorization::Fa
    bΓ::Vector{T}
    b::Vector{T}
    x::Vector{T}
end

function ShortleyWellerSolver(p::PoissonProblem{N, T}; factorize::Bool = true) where {N, T}
    A, bΓ = assemble_operator(p)
    F = factorize ? lu(A) : A
    n = ndofs(p)
    ShortleyWellerSolver(p, A, F, bΓ, zeros(T, n), zeros(T, n))
end

ShortleyWellerSolver(::ShortleyWeller, p::PoissonProblem; kwargs...) =
    ShortleyWellerSolver(p; kwargs...)

problem(s::ShortleyWellerSolver) = s.problem

"""
    solve!(phi, solver[, source])

Solve the Poisson problem and write the solution, including the prescribed values at solid and
boundary nodes, into `phi`. With `source` (an array of `size(grid)`) the problem's own source is
replaced for this solve only.
"""
solve!(phi::AbstractArray, s::ShortleyWellerSolver) = solve!(phi, s, s.problem.source)

function solve!(phi::AbstractArray, s::ShortleyWellerSolver, source::AbstractArray)
    p = s.problem
    size(source) == size(p.grid) || throw(DimensionMismatch("source must have size $(size(p.grid))"))
    for (row, I) in enumerate(p.nodes)
        s.b[row] = s.bΓ[row] + source[I]
    end
    solve_linear_system!(s.x, s.factorization, s.b)
    scatter!(phi, p, s.x)
end

"""
    solve(problem)

Convenience: assemble, solve and return `phi` as a new array.
"""
function solve(p::PoissonProblem{N, T}) where {N, T}
    phi = zeros(T, size(p.grid))
    solve!(phi, ShortleyWellerSolver(p))
end
