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

fits_Poisson_G1 = load(fitpath*"G1_fits_Poisson.jld2", "fits_Poisson")
fits_ZIPoisson_G1 = load(fitpath*"G1_fits_ZIPoisson.jld2", "fits_ZIPoisson")
fits_NB_G1 = load(fitpath*"G1_fits_NB.jld2", "fits_NB")
fits_ZINB_G1 = load(fitpath*"G1_fits_ZINB.jld2", "fits_ZINB")
fits_BP_G1 = load(fitpath*"G1_fits_BP.jld2", "fits_BP")
fits_ZIBP_G1 = load(fitpath*"G1_fits_ZIBP.jld2", "fits_ZIBP")

nconv = 2
@time BICs_Poisson_G1 = get_BICs(fits_Poisson_G1, nconv, xG1)
@time BICs_ZIPoisson_G1 = get_BICs(fits_ZIPoisson_G1, nconv, xG1)
@time BICs_NB_G1 = get_BICs(fits_NB_G1, nconv, xG1)
@time BICs_ZINB_G1 = get_BICs(fits_ZINB_G1, nconv, xG1)
@time BICs_BP_G1 = get_BICs(fits_BP_G1, nconv, xG1)
@time BICs_ZIBP_G1 = get_BICs(fits_ZIBP_G1, nconv, xG1)

# G2/M phase

fits_Poisson_G2M = load(fitpath*"G2M_fits_Poisson.jld2", "fits_Poisson")
fits_ZIPoisson_G2M = load(fitpath*"G2M_fits_ZIPoisson.jld2", "fits_ZIPoisson")
fits_NB_G2M = load(fitpath*"G2M_fits_NB.jld2", "fits_NB")
fits_ZINB_G2M = load(fitpath*"G2M_fits_ZINB.jld2", "fits_ZINB")
fits_BP_G2M = load(fitpath*"G2M_fits_BP.jld2", "fits_BP")
fits_ZIBP_G2M = load(fitpath*"G2M_fits_ZIBP.jld2", "fits_ZIBP")

nconv = 4
@time BICs_Poisson_G2M = get_BICs(fits_Poisson_G2M, nconv, xG2M)
@time BICs_ZIPoisson_G2M = get_BICs(fits_ZIPoisson_G2M, nconv, xG2M)
@time BICs_NB_G2M = get_BICs(fits_NB_G2M, nconv, xG2M)
@time BICs_ZINB_G2M = get_BICs(fits_ZINB_G2M, nconv, xG2M)
@time BICs_BP_G2M = get_BICs(fits_BP_G2M, nconv, xG2M)
@time BICs_ZIBP_G2M = get_BICs(fits_ZIBP_G2M, nconv, xG2M)

# --- Sort the BICs to obtain the best fits for each gene ---
 
indsarr_G1 = sort_BICs(BICs_Poisson_G1, BICs_ZIPoisson_G1, BICs_NB_G1, BICs_ZINB_G1, BICs_BP_G1, BICs_ZIBP_G1, BICtol=10)
inds_Poisson_G1, inds_ZIPoisson_G1, inds_NB_G1, inds_ZINB_G1, inds_BP_G1, inds_ZIBP_G1 = indsarr_G1
indsarr_G2M = sort_BICs(BICs_Poisson_G2M, BICs_ZIPoisson_G2M, BICs_NB_G2M, BICs_ZINB_G2M, BICs_BP_G2M, BICs_ZIBP_G2M, BICtol=10)
inds_Poisson_G2M, inds_ZIPoisson_G2M, inds_NB_G2M, inds_ZINB_G2M, inds_BP_G2M, inds_ZIBP_G2M = indsarr_G2M

# put best fit distributions for each gene and phase in separate arrays (convenient later)
fits_G1 = Vector{Distribution}(undef, ngenes)
fits_G1[inds_Poisson_G1] = fits_Poisson_G1[inds_Poisson_G1]
fits_G1[inds_ZIPoisson_G1] = fits_ZIPoisson_G1[inds_ZIPoisson_G1]
fits_G1[inds_NB_G1] = fits_NB_G1[inds_NB_G1]
fits_G1[inds_ZINB_G1] = fits_ZINB_G1[inds_ZINB_G1]
fits_G1[inds_BP_G1] = fits_BP_G1[inds_BP_G1]
fits_G1[inds_ZIBP_G1] = fits_ZIBP_G1[inds_ZIBP_G1]

fits_G2M = Vector{Distribution}(undef, ngenes)
fits_G2M[inds_Poisson_G2M] = fits_Poisson_G2M[inds_Poisson_G2M]
fits_G2M[inds_ZIPoisson_G2M] = fits_ZIPoisson_G2M[inds_ZIPoisson_G2M]
fits_G2M[inds_NB_G2M] = fits_NB_G2M[inds_NB_G2M]
fits_G2M[inds_ZINB_G2M] = fits_ZINB_G2M[inds_ZINB_G2M]
fits_G2M[inds_BP_G2M] = fits_BP_G2M[inds_BP_G2M]
fits_G2M[inds_ZIBP_G2M] = fits_ZIBP_G2M[inds_ZIBP_G2M]

# Check how many genes are best fit by specific distribution in G1 phase
length(inds_Poisson_G1)
length(inds_ZIPoisson_G1)
length(inds_NB_G1)
length(inds_ZINB_G1)
length(inds_BP_G1)
length(inds_ZIBP_G1)

# Check how many genes are best fit by specific distribution in G2M phase
length(inds_Poisson_G2M)
length(inds_ZIPoisson_G2M)
length(inds_NB_G2M)
length(inds_ZINB_G2M)
length(inds_BP_G2M)
length(inds_ZIBP_G2M)

# Filter out genes that at least in one phase are fit by Poisson or zero-inflated Poisson

inds = union(inds_Poisson_G1, inds_Poisson_G2M, inds_ZIPoisson_G1, inds_ZIPoisson_G2M)
println("Genes filtered out: $(length(inds))")
inds = (1:ngenes)[1:end .∉ [inds]]
println("Genes left: $(length(inds))")