using Test, TestExtras
using TensorKit
using TensorKit: type_repr
using LinearAlgebra: LinearAlgebra
import Base: showarg

spacelist = default_spacelist(fast_tests)

for V in spacelist
    I = sectortype(first(V))
    Istr = type_repr(I)
    println("---------------------------------------")
    println("HermitianTensorMap with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "HermitianTensorMap with symmetry: $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        W = V1 ⊗ V2 ← V1 ⊗ V2
        for T in (Float64, ComplexF64)
            @timedtestset "eltype $T" begin
                # Create a Hermitian tensor: (A + A')/2
                A = randn(T, W)
                Hdata = (A + A') / 2
                H = @constinferred HermitianTensorMap(Hdata)
                @test H isa HermitianTensorMap{T, spacetype(W), 2, typeof(Hdata)}

                @timedtestset "Construction and properties" begin
                    H2 = @constinferred HermitianTensorMap(Hdata; uplo = 'L')
                    @test H.uplo == 'U'
                    @test H2.uplo == 'L'
                    @test @constinferred(space(H)) == space(Hdata)
                    @test @constinferred(dim(H)) == dim(Hdata)
                    @test @constinferred(scalartype(H)) == T
                    @test @constinferred(sectortype(H)) == I
                    @test @constinferred(codomain(H)) == codomain(Hdata)
                    @test @constinferred(domain(H)) == domain(Hdata)
                    @test @constinferred(eltype(H)) == T
                end

                @timedtestset "Block access" begin
                    for (c, b) in blocks(H)
                        @test b == block(H, c)
                        @test b == block(Hdata, c)
                    end
                    @test collect(blocksectors(H)) == collect(blocksectors(Hdata))
                end

                @timedtestset "ishermitian and isposdef" begin
                    @test LinearAlgebra.ishermitian(H)
                    # isposdef only when T <: Real for Hermitian
                end

                @timedtestset "Arithmetic rules" begin
                    # H + H → HermitianTensorMap
                    Hsum = @constinferred H + H
                    @test Hsum isa HermitianTensorMap
                    @test Hsum.data ≈ Hdata + Hdata
                    @test Hsum.uplo == H.uplo

                    # H + TensorMap → TensorMap
                    A2 = randn(T, W)
                    @test (H + A2) isa TensorMap
                    @test (A2 + H) isa TensorMap
                    @test H + A2 ≈ Hdata + A2
                    @test A2 + H ≈ A2 + Hdata

                    # H - H → HermitianTensorMap
                    Hdiff = @constinferred H - H
                    @test Hdiff isa HermitianTensorMap
                    @test Hdiff.data ≈ Hdata - Hdata

                    # H - TensorMap → TensorMap
                    @test (H - A2) isa TensorMap
                    @test (A2 - H) isa TensorMap

                    # H * anything → TensorMap
                    H3data = (randn(T, V1 ⊗ V2 ← V1 ⊗ V2) + randn(T, V1 ⊗ V2 ← V1 ⊗ V2)') / 2
                    H3 = HermitianTensorMap(H3data)
                    @test (H * H3) isa TensorMap
                    @test H * H3 ≈ Hdata * H3data
                    @test (H * A2) isa TensorMap
                    @test (A2 * H) isa TensorMap

                    # Scalar multiplication
                    α = rand(T isa Real ? T : Float64)
                    @test (@constinferred α * H) isa HermitianTensorMap
                    @test (α * H).data ≈ α * Hdata
                    @test (@constinferred H * α) isa HermitianTensorMap
                    @test (H * α).data ≈ Hdata * α

                    # Unary minus
                    @test (@constinferred -H) isa HermitianTensorMap
                    @test (-H).data ≈ -Hdata
                end

                @timedtestset "Linear solve and division" begin
                    rhs = randn(T, V1 ⊗ V2 ← V1 ⊗ V2)
                    x = @constinferred H \ rhs
                    @test x isa TensorMap
                    @test Hdata * x ≈ rhs

                    # H \\ H = I
                    x2 = @constinferred H \ Hdata
                    @test x2 ≈ one(Hdata)

                    # T / H
                    x3 = @constinferred rhs / H
                    @test x3 isa TensorMap
                    @test x3 * Hdata ≈ rhs

                    # H / H
                    x4 = @constinferred H / H
                    @test x4 isa TensorMap
                    @test x4 ≈ one(Hdata)

                    # H / T
                    x5 = @constinferred H / rhs
                    @test x5 isa TensorMap
                    @test x5 ≈ Hdata / rhs
                end

                @timedtestset "inv and pinv" begin
                    # Need well-conditioned H
                    Hreg = HermitianTensorMap(Hdata + one(Hdata))
                    Hinv = @constinferred inv(Hreg)
                    @test Hinv isa HermitianTensorMap
                    @test Hreg * Hinv ≈ one(Hdata)

                    Hpinv = @constinferred pinv(Hreg)
                    @test Hpinv isa HermitianTensorMap
                    @test Hreg * Hpinv * Hreg ≈ Hreg
                end

                @timedtestset "Matrix functions" begin
                    # Use A*A' + I to guarantee positive definiteness
                    Hpos = HermitianTensorMap(Hdata * Hdata' + one(Hdata))
                    @test (@constinferred exp(Hpos)) isa HermitianTensorMap
                    @test TensorMap(exp(Hpos)) ≈ exp(TensorMap(Hpos))
                    @test (@constinferred sqrt(Hpos)) isa HermitianTensorMap
                    @test TensorMap(sqrt(Hpos)) ≈ sqrt(TensorMap(Hpos))
                    @test (@constinferred log(Hpos)) isa HermitianTensorMap
                    @test TensorMap(log(Hpos)) ≈ log(TensorMap(Hpos))
                    # For cos/sin, no positivity needed
                    @test (@constinferred cos(H)) isa HermitianTensorMap
                    @test TensorMap(cos(H)) ≈ cos(Hdata)
                    @test (@constinferred sin(H)) isa HermitianTensorMap
                    @test TensorMap(sin(H)) ≈ sin(Hdata)
                end

                @timedtestset "Power" begin
                    Hpow2 = @constinferred H^2
                    @test Hpow2 isa HermitianTensorMap
                    @test Hpow2.data ≈ Hdata^2

                    Hpow3 = @constinferred H^3
                    @test Hpow3 isa HermitianTensorMap
                    @test Hpow3.data ≈ Hdata^3
                end

                @timedtestset "Adjoint and transpose" begin
                    @test (@constinferred adjoint(H)) isa HermitianTensorMap
                    @test H'.uplo == (H.uplo == 'U' ? 'L' : 'U')
                    @test TensorMap(H') ≈ Hdata' ≈ Hdata  # Hermitian: adjoint = self

                    if T <: Real
                        @test (@constinferred transpose(H)) isa HermitianTensorMap
                        @test transpose(H) === H  # identity for real
                    else
                        Ht = @constinferred transpose(H)
                        @test Ht isa HermitianTensorMap
                        @test TensorMap(Ht) ≈ transpose(Hdata)
                    end
                end

                @timedtestset "Utilities" begin
                    # copy
                    Hc = @constinferred copy(H)
                    @test Hc isa HermitianTensorMap
                    @test Hc.data == H.data
                    @test Hc.uplo == H.uplo

                    # TensorMap conversion
                    tm = @constinferred TensorMap(H)
                    @test tm isa TensorMap
                    @test tm ≈ Hdata

                    # one / zero
                    @test (@constinferred one(H)) isa HermitianTensorMap
                    @test one(H).data ≈ one(Hdata)
                    @test (@constinferred zero(H)) isa HermitianTensorMap
                    @test zero(H).data ≈ zero(Hdata)

                    # tr / norm
                    @test @constinferred(tr(H)) ≈ tr(Hdata)
                    @test isa(@constinferred(norm(H)), real(T))
                    @test norm(H) ≈ norm(Hdata)

                    # real
                    Hreal = @constinferred real(H)
                    @test Hreal isa HermitianTensorMap
                    @test TensorMap(Hreal) ≈ real(Hdata)
                end

                @timedtestset "Show methods" begin
                    @test startswith(sprint(show, MIME("text/plain"), H),
                                    "HermitianTensorMap")
                    s = sprint(summary, H)
                    @test startswith(s, "HermitianTensorMap")
                    @test sprint(showarg, H, true) == type_repr(typeof(H))
                end
            end
        end

        # N=1 case: single-space operator
        @timedtestset "N=1 single-space" begin
            T = ComplexF64
            V = V1
            W = V ← V
            A = randn(T, W)
            Hdata = (A + A') / 2
            H = @constinferred HermitianTensorMap(Hdata)
            @test H isa HermitianTensorMap{T, spacetype(W), 1, typeof(Hdata)}
            @test numout(H) == numin(H) == 1

            # Basic solve
            b = randn(T, W)
            x = H \ b
            @test Hdata * x ≈ b

            # uplo mixed addition
            Hu = HermitianTensorMap(Hdata; uplo = 'U')
            Hl = HermitianTensorMap(Hdata; uplo = 'L')
            @test (Hu + Hu).uplo == 'U'
            @test (Hl + Hl).uplo == 'L'
            @test (Hu + Hl).uplo == 'U'  # default when mixed
        end
    end
end
