#
# Gradients of a grid function that respect the embedded boundary, and interpolation that never
# reads values from inside the solids.
#

# The value and distance on one side, for gradient purposes.
@inline function gradient_side(p::PoissonProblem{N, T}, phi, I, d, s) where {N, T}
    side = stencil_side(p, I, d, s)
    if side.mirror
        return side
    end
    side.col > 0 && return StencilSide(side.dist, side.col, T(phi[p.nodes[side.col]]), false, false)
    side
end

# Second-order one-sided derivative from the node towards direction `s`, using up to two further
# points along the grid line if they are not cut; falls back to first order.
@inline function one_sided(p::PoissonProblem{N, T}, phi, I, d, s, φ0) where {N, T}
    s1 = gradient_side(p, phi, I, d, s)
    s1.mirror && return zero(T)
    if !s1.cut
        J = I + CartesianIndex(ntuple(e -> e == d ? s : 0, Val(N)))
        if p.eb.kind[J] != SOLID
            s2 = gradient_side(p, phi, J, d, s)
            if !s2.mirror
                d1, d2 = s1.dist, s1.dist + s2.dist
                # three-point one-sided formula on the points 0, d1, d2
                c1 = d2 / (d1 * (d2 - d1))
                c2 = -d1 / (d2 * (d2 - d1))
                c0 = -(c1 + c2)
                return s * (c0 * φ0 + c1 * s1.val + c2 * s2.val)
            end
        end
    end
    s * (s1.val - φ0) / s1.dist
end

"""
    gradient!(G, phi, problem)

The gradient of `phi` at every domain node, written into the tuple of arrays `G` (one per
dimension). Solid nodes get zero.

Along each grid line the derivative uses the three points the Shortley–Weller stencil uses —
the node and its two neighbours, or the boundary cuts in their place — so it reads no value
from inside a solid and is second order wherever the solution is smooth up to the boundary:

    ∂φ/∂x ≈ [d₋² (φ₊ - φᵢ) + d₊² (φᵢ - φ₋)] / [d₋ d₊ (d₋ + d₊)]

At symmetry faces the normal derivative is zero. At Dirichlet box faces, and along grid lines
where a node is `ON_BOUNDARY`, a one-sided formula into the domain is used.
"""
function gradient!(G::NTuple{N, AbstractArray}, phi::AbstractArray,
        p::PoissonProblem{N, T}) where {N, T}
    eb = p.eb
    for I in CartesianIndices(p.grid)
        if eb.kind[I] == SOLID
            for d in 1:N
                G[d][I] = zero(T)
            end
            continue
        end
        φ0 = T(phi[I])
        for d in 1:N
            km, kp = side_index(d, -1), side_index(d, +1)
            atmin = I[d] == 1
            atmax = I[d] == p.grid.ncells[d] + 1
            mirror_m = atmin && ismirror(p.faces[km])
            mirror_p = atmax && ismirror(p.faces[kp])
            tiny_m = eb.cut_id[km, I] != 0 && eb.theta[km, I] < eb.θmin
            tiny_p = eb.cut_id[kp, I] != 0 && eb.theta[kp, I] < eb.θmin
            open_m = !(atmin && !mirror_m) && !tiny_m
            open_p = !(atmax && !mirror_p) && !tiny_p
            G[d][I] = if (mirror_m && !tiny_p) || (mirror_p && !tiny_m)
                zero(T)
            elseif open_m && open_p
                sm = gradient_side(p, phi, I, d, -1)
                sp = gradient_side(p, phi, I, d, +1)
                dm, dp = sm.dist, sp.dist
                (dm^2 * (sp.val - φ0) + dp^2 * (φ0 - sm.val)) / (dm * dp * (dm + dp))
            elseif open_p
                one_sided(p, phi, I, d, +1, φ0)
            elseif open_m
                one_sided(p, phi, I, d, -1, φ0)
            else
                zero(T)
            end
        end
    end
    G
end

"""
    gradient(phi, problem)

Allocating version of [`gradient!`](@ref): returns a tuple of arrays.
"""
function gradient(phi::AbstractArray, p::PoissonProblem{N, T}) where {N, T}
    G = ntuple(_ -> zeros(T, size(p.grid)), Val(N))
    gradient!(G, phi, p)
end

# ---------------------------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------------------------

"""
    interpolate(grid, data, x)

Multilinear interpolation of the nodal array `data` at `x`. Points outside the grid are clamped
to the boundary cell.
"""
@inline function interpolate(grid::StructuredGrid{N, T}, data::AbstractArray, x) where {N, T}
    I, ξ = locate(grid, x)
    acc = zero(eltype(data))
    for c in CartesianIndices(ntuple(_ -> 0:1, Val(N)))
        w = one(T)
        for d in 1:N
            w *= c[d] == 0 ? one(T) - ξ[d] : ξ[d]
        end
        acc += w * data[I + c]
    end
    acc
end

"""
    interpolate(grid, data, mask, x) -> (value, weight)

Multilinear interpolation that uses only the corners where `mask` is `true`, renormalising the
weights of those corners. Returns the interpolated value and the total weight of the unmasked
corners (`0` if none was usable, in which case the value is zero).

This is how values near embedded boundaries are interpolated without reading from inside the
solids: in a cut cell the solid corners simply drop out. In a cell with all corners in the
domain it is ordinary multilinear interpolation. Near the boundary it is first order.
"""
@inline function interpolate(grid::StructuredGrid{N, T}, data::AbstractArray,
        mask::AbstractArray{Bool}, x) where {N, T}
    I, ξ = locate(grid, x)
    acc = zero(eltype(data))
    wsum = zero(T)
    for c in CartesianIndices(ntuple(_ -> 0:1, Val(N)))
        J = I + c
        mask[J] || continue
        w = one(T)
        for d in 1:N
            w *= c[d] == 0 ? one(T) - ξ[d] : ξ[d]
        end
        acc += w * data[J]
        wsum += w
    end
    wsum > 0 ? (acc / wsum, wsum) : (zero(acc), wsum)
end

# ---------------------------------------------------------------------------------------------
# Ghost values in the solids
# ---------------------------------------------------------------------------------------------

"""
    ghost_mask(eb)

`true` for every solid node with at least one domain node as a grid neighbour: the nodes that
[`extrapolate_ghosts!`](@ref) fills.
"""
function ghost_mask(eb::EmbeddedBoundary{N}) where {N}
    m = falses(size(eb.kind))
    for I in CartesianIndices(eb.kind)
        eb.kind[I] == SOLID || continue
        for d in 1:N, s in (-1, 1)
            J = I + CartesianIndex(ntuple(e -> e == d ? s : 0, Val(N)))
            if checkbounds(Bool, eb.kind, J) && isdomain(eb.kind[J])
                m[I] = true
            end
        end
    end
    m
end

"""
    extrapolate_ghosts!(A, eb)

Overwrite `A` at the ghost nodes (see [`ghost_mask`](@ref)) with values extrapolated linearly
from the domain along the grid lines, averaged over the directions in which that is possible:
`2A[D] - A[D₂]` from the domain neighbour `D` and the next node `D₂` beyond it, or `A[D]` where
`D₂` is not available. Values in the domain are unchanged.

Multilinear interpolation of a field extended this way is exact for linear fields and second
order for smooth ones in cells cut by the boundary, where dropping the solid corners
([`interpolate`](@ref) with a mask) is only first order.
"""
function extrapolate_ghosts!(A::AbstractArray, eb::EmbeddedBoundary{N}) where {N}
    kind = eb.kind
    for I in CartesianIndices(kind)
        kind[I] == SOLID || continue
        acc = zero(eltype(A))
        n = 0
        for d in 1:N, s in (-1, 1)
            e = CartesianIndex(ntuple(k -> k == d ? s : 0, Val(N)))
            D = I + e
            (checkbounds(Bool, kind, D) && isdomain(kind[D])) || continue
            D₂ = D + e
            # the grid line D → D₂ must not cross the boundary either
            if checkbounds(Bool, kind, D₂) && isdomain(kind[D₂]) &&
               eb.cut_id[side_index(d, s), D] == 0
                acc += 2A[D] - A[D₂]
            else
                acc += A[D]
            end
            n += 1
        end
        n > 0 && (A[I] = acc / n)
    end
    A
end
