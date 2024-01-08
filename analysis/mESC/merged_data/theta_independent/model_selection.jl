apath = normpath(joinpath(@__DIR__, "../../"))
include(apath*"load_merged_data.jl")

srcpath = normpath(joinpath(@__DIR__, "../../../../src/"))
include(srcpath*"dists.jl")
include(srcpath*"mle.jl")
include(srcpath*"bic.jl")

fitpath = normpath(datapath*"fits_merged/")
println("\nLoading fits from $fitpath\n")

# --- Model selection ---

data = counts_merged
nconv = 2

@load fitpath*"fits_Poisson.jld2" fits_Poisson
@load fitpath*"fits_ZIPoisson.jld2" fits_ZIPoisson
@load fitpath*"fits_NB.jld2" fits_NB
@load fitpath*"fits_ZINB.jld2" fits_ZINB
@load fitpath*"fits_BP.jld2" fits_BP
@load fitpath*"fits_ZIBP.jld2" fits_ZIBP

@time BICs_Poisson = get_BICs(fits_Poisson, nconv, data)
@time BICs_ZIPoisson = get_BICs(fits_ZIPoisson, nconv, data)
@time BICs_NB = get_BICs(fits_NB, nconv, data)
@time BICs_ZINB = get_BICs(fits_ZINB, nconv, data)
@time BICs_BP = get_BICs(fits_BP, nconv, data)
@time BICs_ZIBP = get_BICs(fits_ZIBP, nconv, data)

# --- Sort the BICs to obtain the best fits for each gene ---

indsarr = sort_BICs(BICs_Poisson, BICs_ZIPoisson, BICs_NB, BICs_ZINB, BICs_BP, BICs_ZIBP, BICtol=10)
inds_Poisson, inds_ZIPoisson, inds_NB, inds_ZINB, inds_BP, inds_ZIBP = indsarr

# put best fit distributions for each gene and phase in separate arrays (convenient later)
fits = Vector{Distribution}(undef, ngenes)
fits[inds_Poisson] = fits_Poisson[inds_Poisson]
fits[inds_ZIPoisson] = fits_ZIPoisson[inds_ZIPoisson]
fits[inds_NB] = fits_NB[inds_NB]
fits[inds_ZINB] = fits_ZINB[inds_ZINB]
fits[inds_BP] = fits_BP[inds_BP]
fits[inds_ZIBP] = fits_ZIBP[inds_ZIBP]

# Check how many genes are best fit by specific distribution in G1 phase
println(length(inds_Poisson))
println(length(inds_ZIPoisson))
println(length(inds_NB))
println(length(inds_ZINB))
println(length(inds_BP))
println(length(inds_ZIBP))

# Filter out genes that at least in one phase are fit by Poisson or zero-inflated Poisson

inds = union(inds_Poisson, inds_ZIPoisson)
println("Genes filtered out: $(length(inds))")
inds = (1:ngenes)[1:end .∉ [inds]]
println("Genes left: $(length(inds))")