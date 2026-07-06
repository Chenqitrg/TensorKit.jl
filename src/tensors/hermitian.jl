# HermitianTensorMap
#==========================================================#
"""
    struct HermitianTensorMap{T, S, N, TT <: AbstractTensorMap{T, S, N, N}} <: AbstractTensorMap{T, S, N, N}

Wrapper around an [`AbstractTensorMap`](@ref) signalling that its matrix blocks are Hermitian.
Requires equal number of input and output indices (N₁ = N₂ = N), enforced by the type system.

When used with `\\`, each block is solved via `LinearAlgebra.Hermitian(block, uplo) \\ rhs`,
which dispatches to Bunch-Kaufman LDLᵀ factorization instead of the generic LU.

This is analogous to `LinearAlgebra.Hermitian` for `AbstractMatrix`, and is useful when a
tensor is known to be Hermitian positive semi-definite by construction (e.g. Gram matrices
from transfer matrix products in MPS algorithms).

See also: [`DiagonalTensorMap`](@ref)
"""
struct HermitianTensorMap{T, S, N, TT <: AbstractTensorMap{T, S, N, N}} <:
       AbstractTensorMap{T, S, N, N}
    data::TT
    uplo::Char
end

# Constructors
#---------------
function HermitianTensorMap(
        data::AbstractTensorMap{T, S, N, N}; uplo::Char = 'U'
    ) where {T, S, N}
    return HermitianTensorMap{T, S, N, typeof(data)}(data, uplo)
end

# Basic interface
#-----------------
space(t::HermitianTensorMap) = space(t.data)

storagetype(::Type{<:HermitianTensorMap{T, S, N, TT}}) where {T, S, N, TT} = storagetype(TT)

dim(t::HermitianTensorMap) = dim(t.data)

# Block access: delegate to wrapped data
#----------------------------------------
function blocks(t::HermitianTensorMap)
    return blocks(t.data)
end

function block(t::HermitianTensorMap, c::Sector)
    return block(t.data, c)
end

function blocksectors(t::HermitianTensorMap)
    return blocksectors(t.data)
end

function subblock(t::HermitianTensorMap, (f₁, f₂)::Tuple)
    return subblock(t.data, (f₁, f₂))
end

blocktype(::Type{<:HermitianTensorMap{T, S, N, TT}}) where {T, S, N, TT} = blocktype(TT)

# similar: default to TensorMap (Hermitian is a property of data, not storage)
#-----------------------------------------------------------------------------
function Base.similar(h::HermitianTensorMap, ::Type{TorA}, V::TensorMapSpace) where {TorA}
    A = TorA <: Number ? similarstoragetype(h, TorA) : TorA
    TT = tensormaptype(spacetype(V), numout(V), numin(V), A)
    return TT(undef, V)
end

# Specialized \ with Hermitian-aware LDLᵀ solve
#-----------------------------------------------
function Base.:(\)(t1::HermitianTensorMap, t2::AbstractTensorMap)
    codomain(t1) == codomain(t2) ||
        throw(SpaceMismatch("non-matching codomains in HermitianTensorMap \\ t2"))
    T = float(promote_type(scalartype(t1), scalartype(t2)))
    t = similar(t2, T, domain(t1) ← domain(t2))
    uplo_sym = t1.uplo == 'U' ? :U : :L
    for (c, b) in blocks(t)
        copy!(b, Hermitian(block(t1, c), uplo_sym) \ block(t2, c))
    end
    return t
end

# Specialized /: A / H = (H \\ A')' uses LDLᵀ via Hermitian
#------------------------------------------------------------
function Base.:(/)(t1::AbstractTensorMap, t2::HermitianTensorMap)
    domain(t1) == domain(t2) ||
        throw(SpaceMismatch("non-matching domains in t1 / HermitianTensorMap"))
    T = float(promote_type(scalartype(t1), scalartype(t2)))
    t = similar(t1, T, codomain(t1) ← codomain(t2))
    uplo_sym = t2.uplo == 'U' ? :U : :L
    for (c, b) in blocks(t)
        copy!(b, block(t1, c) / Hermitian(block(t2, c), uplo_sym))
    end
    return t
end

function Base.:(/)(t1::HermitianTensorMap, t2::HermitianTensorMap)
    domain(t1) == domain(t2) ||
        throw(SpaceMismatch("non-matching domains in t1 / t2"))
    T = float(promote_type(scalartype(t1), scalartype(t2)))
    t = similar(t1, T, codomain(t1) ← codomain(t2))
    uplo_sym = t2.uplo == 'U' ? :U : :L
    for (c, b) in blocks(t)
        copy!(b, block(t1, c) / Hermitian(block(t2, c), uplo_sym))
    end
    return t
end

# VectorInterface: required so H+H → H, H+T → T
#------------------------------------------------
function VectorInterface.zerovector(h::HermitianTensorMap, ::Type{S}) where {S <: Number}
    return HermitianTensorMap(zerovector(h.data, S); uplo = h.uplo)
end

function VectorInterface.scale!(h::HermitianTensorMap, α::Number)
    scale!(h.data, α)
    return h
end

function VectorInterface.add!(
        ty::HermitianTensorMap, tx::HermitianTensorMap, α::Number, β::Number
    )
    add!(ty.data, tx.data, α, β)
    return ty
end

# Arithmetic rules (user-facing)
#---------------------------------
# H + H → HermitianTensorMap (handled by VectorInterface.add → zerovector → same-type path)
# H + T → TensorMap               (handled by VectorInterface.add → mixed-type path)
# H - H → HermitianTensorMap      (handled by VectorInterface.add with α=-1)

# H * anything → TensorMap (product is not generally Hermitian unless commuting)
# Handled by generic compose → compose_dest → tensormaptype → TensorMap

# Scalar multiplication: Real preserves Hermitian, Complex does not
Base.:*(h::HermitianTensorMap, x::Real) = HermitianTensorMap(h.data * x; uplo = h.uplo)
Base.:*(x::Real, h::HermitianTensorMap) = HermitianTensorMap(x * h.data; uplo = h.uplo)
Base.:/(h::HermitianTensorMap, x::Real) = HermitianTensorMap(h.data / x; uplo = h.uplo)

# Unary minus returns Hermitian
Base.:-(h::HermitianTensorMap) = HermitianTensorMap(-h.data; uplo = h.uplo)

# TensorOperations integration
#-------------------------------
function TO.tensoradd_type(TC, H::HermitianTensorMap, p::Index2Tuple, ::Bool)
    N = numout(p)
    M = similarstoragetype(storagetype(H), promote_permute(TC, sectortype(H)))
    TTC = tensormaptype(spacetype(H), N, N, M)
    return HermitianTensorMap{TC, spacetype(H), N, TTC}
end

function TO.tensorcontract_type(
        TC, H::HermitianTensorMap, pA::Index2Tuple, conjA::Bool,
        B::AbstractTensorMap, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple
    )
    # contraction with Hermitian → result is not guaranteed Hermitian
    S = check_spacetype(H, B)
    M = promote_storagetype(promote_permute(TC, sectortype(S)), H, B)
    return tensormaptype(S, numout(pAB), numin(pAB), M)
end

function TO.tensorcontract_type(
        TC, A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
        H::HermitianTensorMap, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple
    )
    S = check_spacetype(A, H)
    M = promote_storagetype(promote_permute(TC, sectortype(S)), A, H)
    return tensormaptype(S, numout(pAB), numin(pAB), M)
end

function TO.tensoralloc(
        ::Type{HermitianTensorMap{T, S, N, TT}}, structure::TensorMapSpace,
        istemp::Val, allocator = TO.DefaultAllocator()
    ) where {T, S, N, TT}
    t = TO.tensoralloc(TT, structure, istemp, allocator)
    return HermitianTensorMap(t; uplo = 'U')
end

has_array_view(::Type{<:HermitianTensorMap}) = false

# Index manipulations
#---------------------
# Adjoint of Hermitian is itself (H† = H by definition)
# uplo flips because adjoint transposes upper↔lower
function Base.adjoint(h::HermitianTensorMap)
    return HermitianTensorMap(h.data; uplo = h.uplo == 'U' ? 'L' : 'U')
end

# Transpose of real Hermitian is itself (Hᵀ = H for real)
Base.transpose(h::HermitianTensorMap{<:Real}) = h
function Base.transpose(h::HermitianTensorMap)
    return HermitianTensorMap(transpose(h.data); uplo = h.uplo == 'U' ? 'L' : 'U')
end

# Factorization support
#-----------------------
LinearAlgebra.ishermitian(::HermitianTensorMap) = true

function LinearAlgebra.isposdef(h::HermitianTensorMap)
    return isposdef(h.data)
end

# inv: blockwise via Hermitian → LDLᵀ + symmetrization, result is Hermitian
function LinearAlgebra.inv(h::HermitianTensorMap)
    T = float(scalartype(h))
    t = similar(h, T, domain(h) ← codomain(h))
    uplo_sym = h.uplo == 'U' ? :U : :L
    for (c, b) in blocks(t)
        copy!(b, inv(Hermitian(block(h, c), uplo_sym)))
    end
    return HermitianTensorMap(t; uplo = h.uplo)
end

# pinv: blockwise via Hermitian eigen → faster than generic SVD
function LinearAlgebra.pinv(h::HermitianTensorMap; kwargs...)
    T = float(scalartype(h))
    t = similar(h, T, domain(h) ← codomain(h))
    uplo_sym = h.uplo == 'U' ? :U : :L
    for (c, b) in blocks(t)
        copy!(b, pinv(Hermitian(block(h, c), uplo_sym); kwargs...))
    end
    return HermitianTensorMap(t; uplo = h.uplo)
end

function Base.real(h::HermitianTensorMap)
    return HermitianTensorMap(real(h.data); uplo = h.uplo)
end

# Matrix functions: blockwise via Hermitian eigen, faster than generic Schur
#----------------------------------------------------------------------------
# float is needed so similar allocates a compatible type for the result
# (e.g. exp(Hermitian{Int}) returns Hermitian{Float64}, not Hermitian{Int})
for f in (:exp, :cos, :sin, :cosh, :sinh, :sqrt, :log)
    @eval function Base.$f(h::HermitianTensorMap)
        T = float(scalartype(h))
        t = similar(h, T, space(h))
        uplo_sym = h.uplo == 'U' ? :U : :L
        for (c, b) in blocks(t)
            copy!(b, $f(Hermitian(block(h, c), uplo_sym)))
        end
        return HermitianTensorMap(t; uplo = h.uplo)
    end
end

# Utilities
#-----------
function TensorMap(h::HermitianTensorMap)
    return convert(TensorMap, h.data)
end

function Base.convert(::Type{<:HermitianTensorMap}, h::HermitianTensorMap)
    return h
end

Base.copy(h::HermitianTensorMap) = HermitianTensorMap(copy(h.data); uplo = h.uplo)

function Base.copy!(tdst::AbstractTensorMap, h::HermitianTensorMap)
    return copy!(tdst, h.data)
end

function Base.one(h::HermitianTensorMap)
    return HermitianTensorMap(one(h.data); uplo = h.uplo)
end

function Base.zero(h::HermitianTensorMap)
    return HermitianTensorMap(zero(h.data); uplo = h.uplo)
end

# ^: blockwise via Hermitian → eigen decomposition
function Base.:(^)(h::HermitianTensorMap, p::Integer)
    T = float(scalartype(h))
    t = similar(h, T, space(h))
    uplo_sym = h.uplo == 'U' ? :U : :L
    for (c, b) in blocks(t)
        copy!(b, Hermitian(block(h, c), uplo_sym)^p)
    end
    return HermitianTensorMap(t; uplo = h.uplo)
end

Base.eltype(::Type{<:HermitianTensorMap{T}}) where {T} = T

function LinearAlgebra.tr(h::HermitianTensorMap)
    return tr(h.data)
end

function LinearAlgebra.norm(h::HermitianTensorMap, p::Real = 2)
    return norm(h.data, p)
end

# Show methods
#--------------
function type_repr(::Type{HermitianTensorMap{T, S, N, TT}}) where {T, S, N, TT}
    return "HermitianTensorMap{$T, $(type_repr(S)), $N, $TT}"
end

function Base.showarg(io::IO, t::HermitianTensorMap, toplevel::Bool)
    !toplevel && print(io, "::")
    print(io, type_repr(typeof(t)))
    return nothing
end

function Base.summary(io::IO, t::HermitianTensorMap)
    print(io, typeof(t), " on ", space(t))
    return nothing
end

function Base.show(io::IO, mime::MIME"text/plain", t::HermitianTensorMap)
    summary(io, t)
    println(io, ":")
    show(io, mime, t.data)
    return nothing
end

# GPU support
#-------------
function Adapt.adapt_structure(to, h::HermitianTensorMap)
    data′ = Adapt.adapt(to, h.data)
    return HermitianTensorMap(data′; uplo = h.uplo)
end
