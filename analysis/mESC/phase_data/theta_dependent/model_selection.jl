if !@isdefined(xG1) || !@isdefined(xG2M) || !@isdefined(fits_G1) || !@isdefined(fits_G2M)
    apath = normpath(joinpath(@__DIR__, "../../"))
    include(apath*"load_data.jl")
end

srcpath = normpath(joinpath(@__DIR__, "../../../../src/"))
include(srcpath*"dists.jl")
include(srcpath*"mle.jl")
include(srcpath*"bic.jl")

fitpath = normpath(datapath*"fits_phase/")
println("\nLoading fits from $fitpath\n")

# --- Model selection ---

# G1 phase

fits_PoissonTheta_G1 = load(fitpath*"G1_fits_PoissonTheta.jld2", "fits_PoissonTheta")
fits_ZIPoissonTheta_G1 = load(fitpath*"G1_fits_ZIPoissonTheta.jld2", "fits_ZIPoissonTheta")
fits_NBTheta_G1 = load(fitpath*"G1_fits_NBTheta.jld2", "fits_NBTheta")
fits_ZINBTheta_G1 = load(fitpath*"G1_fits_ZINBTheta.jld2", "fits_ZINBTheta")
fits_BPTheta_G1 = load(fitpath*"G1_fits_BPTheta.jld2", "fits_BPTheta")
fits_ZIBPTheta_G1 = load(fitpath*"G1_fits_ZIBPTheta.jld2", "fits_ZIBPTheta")

nconv = 2
@time BICs_PoissonTheta_G1 = get_BICs(fits_PoissonTheta_G1, nconv, xG1, thetaG1)
@time BICs_ZIPoissonTheta_G1 = get_BICs(fits_ZIPoissonTheta_G1, nconv, xG1, thetaG1)
@time BICs_NBTheta_G1 = get_BICs(fits_NBTheta_G1, nconv, xG1, thetaG1)
@time BICs_ZINBTheta_G1 = get_BICs(fits_ZINBTheta_G1, nconv, xG1, thetaG1)
@time BICs_BPTheta_G1 = get_BICs(fits_BPTheta_G1, nconv, xG1, thetaG1)
@time BICs_ZIBPTheta_G1 = get_BICs(fits_ZIBPTheta_G1, nconv, xG1, thetaG1)

# G2/M phase

fits_PoissonTheta_G2M = load(fitpath*"G2M_fits_PoissonTheta.jld2", "fits_PoissonTheta")
fits_ZIPoissonTheta_G2M = load(fitpath*"G2M_fits_ZIPoissonTheta.jld2", "fits_ZIPoissonTheta")
fits_NBTheta_G2M = load(fitpath*"G2M_fits_NBTheta.jld2", "fits_NBTheta")
fits_ZINBTheta_G2M = load(fitpath*"G2M_fits_ZINBTheta.jld2", "fits_ZINBTheta")
fits_BPTheta_G2M = load(fitpath*"G2M_fits_BPTheta.jld2", "fits_BPTheta")
fits_ZIBPTheta_G2M = load(fitpath*"G2M_fits_ZIBPTheta.jld2", "fits_ZIBPTheta")

nconv = 4
@time BICs_PoissonTheta_G2M = get_BICs(fits_PoissonTheta_G2M, nconv, xG2M, thetaG2M)
@time BICs_ZIPoissonTheta_G2M = get_BICs(fits_ZIPoissonTheta_G2M, nconv, xG2M, thetaG2M)
@time BICs_NBTheta_G2M = get_BICs(fits_NBTheta_G2M, nconv, xG2M, thetaG2M)
@time BICs_ZINBTheta_G2M = get_BICs(fits_ZINBTheta_G2M, nconv, xG2M, thetaG2M)
@time BICs_BPTheta_G2M = get_BICs(fits_BPTheta_G2M, nconv, xG2M, thetaG2M)
@time BICs_ZIBPTheta_G2M = get_BICs(fits_ZIBPTheta_G2M, nconv, xG2M, thetaG2M)

# --- Sort the BICs to obtain the best fits for each gene ---
 
indsarr_G1 = sort_BICs(BICs_PoissonTheta_G1, BICs_ZIPoissonTheta_G1, BICs_NBTheta_G1, BICs_ZINBTheta_G1, BICs_BPTheta_G1, BICs_ZIBPTheta_G1, BICtol=10)
inds_PoissonTheta_G1, inds_ZIPoissonTheta_G1, inds_NBTheta_G1, inds_ZINBTheta_G1, inds_BPTheta_G1, inds_ZIBPTheta_G1 = indsarr_G1;
indsarr_G2M = sort_BICs(BICs_PoissonTheta_G2M, BICs_ZIPoissonTheta_G2M, BICs_NBTheta_G2M, BICs_ZINBTheta_G2M, BICs_BPTheta_G2M, BICs_ZIBPTheta_G2M, BICtol=10)
inds_PoissonTheta_G2M, inds_ZIPoissonTheta_G2M, inds_NBTheta_G2M, inds_ZINBTheta_G2M, inds_BPTheta_G2M, inds_ZIBPTheta_G2M = indsarr_G2M;

# put best fit distributions for each gene and phase in separate arrays (convenient later)
fits_G1 = Vector{Distribution}(undef, ngenes)
fits_G1[inds_PoissonTheta_G1] = fits_PoissonTheta_G1[inds_PoissonTheta_G1]
fits_G1[inds_ZIPoissonTheta_G1] = fits_ZIPoissonTheta_G1[inds_ZIPoissonTheta_G1]
fits_G1[inds_NBTheta_G1] = fits_NBTheta_G1[inds_NBTheta_G1]
fits_G1[inds_ZINBTheta_G1] = fits_ZINBTheta_G1[inds_ZINBTheta_G1]
fits_G1[inds_BPTheta_G1] = fits_BPTheta_G1[inds_BPTheta_G1]
fits_G1[inds_ZIBPTheta_G1] = fits_ZIBPTheta_G1[inds_ZIBPTheta_G1]

fits_G2M = Vector{Distribution}(undef, ngenes)
fits_G2M[inds_PoissonTheta_G2M] = fits_PoissonTheta_G2M[inds_PoissonTheta_G2M]
fits_G2M[inds_ZIPoissonTheta_G2M] = fits_ZIPoissonTheta_G2M[inds_ZIPoissonTheta_G2M]
fits_G2M[inds_NBTheta_G2M] = fits_NBTheta_G2M[inds_NBTheta_G2M]
fits_G2M[inds_ZINBTheta_G2M] = fits_ZINBTheta_G2M[inds_ZINBTheta_G2M]
fits_G2M[inds_BPTheta_G2M] = fits_BPTheta_G2M[inds_BPTheta_G2M]
fits_G2M[inds_ZIBPTheta_G2M] = fits_ZIBPTheta_G2M[inds_ZIBPTheta_G2M]

# Check how many genes are best fit by specific distribution in G1 phase
println(length(inds_PoissonTheta_G1))
println(length(inds_ZIPoissonTheta_G1))
println(length(inds_NBTheta_G1))
println(length(inds_ZINBTheta_G1))
println(length(inds_BPTheta_G1))
println(length(inds_ZIBPTheta_G1))

# Check how many genes are best fit by specific distribution in G2M phase
println(length(inds_PoissonTheta_G2M))
println(length(inds_ZIPoissonTheta_G2M))
println(length(inds_NBTheta_G2M))
println(length(inds_ZINBTheta_G2M))
println(length(inds_BPTheta_G2M))
println(length(inds_ZIBPTheta_G2M))

# Filter out genes that at least in one phase are fit by Poisson or zero-inflated Poisson

inds = union(inds_PoissonTheta_G1, inds_PoissonTheta_G2M, inds_ZIPoissonTheta_G1, inds_ZIPoissonTheta_G2M)
println("Genes filtered out: $(length(inds))")
inds = (1:ngenes)[1:end .∉ [inds]]
println("Genes left: $(length(inds))")