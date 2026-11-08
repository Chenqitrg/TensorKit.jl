using Test
using LinearAlgebra
using Random
using TensorKit

@testset "Structured tensor solves" begin
    rng = MersenneTwister(182)
    for V in (ComplexSpace(3), Z2Space(0 => 3, 1 => 2), SU2Space(0 => 3, 1 => 2)),
            T in (Float32, Float64, ComplexF32, ComplexF64),
            positive in (false, true)
        @testset "$V, $T, positive=$positive" begin
            A = zeros(T, V ← V)
            for (_, b) in blocks(A)
                n = size(b, 1)
                for j in 1:n
                    b[j, j] = (positive || isodd(j) ? 1 : -1) * (n + 1)
                    for i in 1:(j - 1)
                        b[i, j] = T <: Real ? T(0.25) : T(0.25 + 0.125im)
                        b[j, i] = conj(b[i, j])
                    end
                end
            end
            X = randn(rng, T, V ← V ⊗ V)
            B = A * X
            C = X' * A
            A0, B0, C0 = copy(A), copy(B), copy(C)
            options = positive ? (; isposdef = true) : (; ishermitian = true)
            @test Base.:(\)(A, B; options...) ≈ X
            @test Base.:(/)(C, A; options...) ≈ X'
            Ai = inv(A; options...)
            @test A * Ai ≈ one(A)
            @test Ai * A ≈ one(A)
            @test Ai ≈ Ai'
            @test (A, B, C) == (A0, B0, C0)
            # The upper triangle defines the Hermitian operator; the other
            # triangle must not accidentally enter a generic LU solve.
            upper = copy(A)
            for (_, b) in blocks(upper), j in axes(b, 2), i in (j + 1):size(b, 1)
                b[i, j] = T(NaN)
            end
            @test Base.:(\)(upper, B; options...) ≈ X
            @test Base.:(/)(C, upper; options...) ≈ X'
            @test inv(upper; options...) ≈ Ai
            @test Base.:(\)(A', B; options...) ≈ X
            @test Base.:(/)(C, A'; options...) ≈ X'
            @test inv(A'; options...) ≈ Ai
            if positive
                @test Base.:(\)(A, B; ishermitian = true, isposdef = true) ≈ X
            else
                @test_throws PosDefException Base.:(\)(A, B; isposdef = true)
                @test_throws PosDefException Base.:(/)(C, A; isposdef = true)
                @test_throws PosDefException inv(A; isposdef = true)
            end
        end
    end
    @testset "Spaces, empty tensors, and default solves" begin
        V, W = ComplexSpace(3), ComplexSpace(2)
        A = randn(rng, ComplexF64, V ← W)
        B = randn(rng, ComplexF64, V ← V)
        C = randn(rng, ComplexF64, W ← W)
        for options in ((; ishermitian = true), (; isposdef = true))
            @test_throws SpaceMismatch Base.:(\)(A, B; options...)
            @test_throws SpaceMismatch Base.:(/)(C, A; options...)
            @test_throws SpaceMismatch inv(A; options...)
            @test_throws SpaceMismatch Base.:(\)(B, C; options...)
            @test_throws SpaceMismatch Base.:(/)(B, C; options...)
            empty = zeros(ComplexF64, ComplexSpace(0) ← ComplexSpace(0))
            @test Base.:(\)(empty, empty; options...) == empty
            @test Base.:(/)(empty, empty; options...) == empty
            @test inv(empty; options...) == empty
        end
        # The default rectangular least-squares path remains available.
        X = randn(rng, ComplexF64, W ← W)
        @test A \ (A * X) ≈ X
        @test (X * A') / A' ≈ X
        @test Base.:(\)(A, A * X; ishermitian = false, isposdef = false) ≈ X
    end
    @testset "Diagonal and mixed scalar types" begin
        V = ComplexSpace(2)
        D = DiagonalTensorMap(ComplexF64[2, 4], V)
        E = DiagonalTensorMap(ComplexF64[6, 8], V)
        for options in ((; ishermitian = true), (; isposdef = true))
            @test inv(D; options...) isa DiagonalTensorMap
            @test inv(D; options...) * D ≈ one(D)
            @test Base.:(\)(D, E; options...) * D ≈ E
            @test Base.:(/)(E, D; options...) * D ≈ E
            A = TensorMap([2.0 0.5; 0.5 3.0], V ← V)
            X = randn(rng, ComplexF64, V ← V)
            @test Base.:(\)(D, D * X; options...) ≈ X
            @test Base.:(/)(X * D, D; options...) ≈ X
            @test Base.:(\)(A, A * X; options...) ≈ X
            @test Base.:(/)(X * A, A; options...) ≈ X
        end
        for values in ([1.0, -1.0], [1.0, 0.0])
            D = DiagonalTensorMap(values, V)
            @test_throws PosDefException inv(D; isposdef = true)
            @test_throws PosDefException Base.:(\)(D, D; isposdef = true)
            @test_throws PosDefException Base.:(/)(D, D; isposdef = true)
        end
    end
    @testset "Hermitian pivoting and diagonal interpretation" begin
        V = ComplexSpace(2)
        # An invertible indefinite matrix requiring a 2-by-2 pivot.
        A = TensorMap(ComplexF64[0 im; -im 0], V ← V)
        X = randn(rng, ComplexF64, V ← V)
        @test Base.:(\)(A, A * X; ishermitian = true) ≈ X
        @test Base.:(/)(X * A, A; ishermitian = true) ≈ X
        @test inv(A; ishermitian = true) ≈ A
        raw = TensorMap(ComplexF64[2 + 7im 0; 0 4 - 3im], V ← V)
        D = DiagonalTensorMap(ComplexF64[2 + 7im, 4 - 3im], V)
        expected = DiagonalTensorMap(ComplexF64[0.5, 0.25], V)
        for options in ((; ishermitian = true), (; isposdef = true))
            @test inv(raw; options...) ≈ expected
            @test inv(D; options...) ≈ expected
        end
    end
    @testset "Non-floating coefficients" begin
        V = ComplexSpace(2)
        for T in (Int, Rational{Int}, Complex{Int}, Complex{Rational{Int}})
            A = TensorMap(T[2 1; 1 3], V ← V)
            expected = TensorMap([0.6 -0.2; -0.2 0.4], V ← V)
            for options in ((; ishermitian = true), (; isposdef = true))
                Ai = inv(A; options...)
                @test Ai ≈ expected
                @test scalartype(Ai) == float(T)
            end
        end
        V2 = V ⊗ V
        A = unitary(ComplexF64, fuse(V2), V2)
        # Square numerical blocks alone do not make this an endomorphism.
        for options in ((; ishermitian = true), (; isposdef = true))
            @test_throws SpaceMismatch inv(A; options...)
            @test_throws SpaceMismatch Base.:(\)(A, A; options...)
            @test_throws SpaceMismatch Base.:(/)(A, A; options...)
        end
    end
end
