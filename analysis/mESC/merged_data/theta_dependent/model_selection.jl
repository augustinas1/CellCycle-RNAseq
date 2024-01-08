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
theta = theta_merged
nconv = 2

@load fitpath*"fits_PoissonTheta.jld2" fits_PoissonTheta
@load fitpath*"fits_ZIPoissonTheta.jld2" fits_ZIPoissonTheta
@load fitpath*"fits_NBTheta.jld2" fits_NBTheta
@load fitpath*"fits_ZINBTheta.jld2" fits_ZINBTheta
@load fitpath*"fits_BPTheta.jld2" fits_BPTheta
@load fitpath*"fits_ZIBPTheta.jld2" fits_ZIBPTheta

@time BICs_PoissonTheta = get_BICs(fits_PoissonTheta, nconv, data, theta)
@time BICs_ZIPoissonTheta = get_BICs(fits_ZIPoissonTheta, nconv, data, theta)
@time BICs_NBTheta = get_BICs(fits_NBTheta, nconv, data, theta)
@time BICs_ZINBTheta = get_BICs(fits_ZINBTheta, nconv, data, theta)
@time BICs_BPTheta = get_BICs(fits_BPTheta, nconv, data, theta)
@time BICs_ZIBPTheta = get_BICs(fits_ZIBPTheta, nconv, data, theta)

# --- Sort the BICs to obtain the best fits for each gene ---
 
indsarr = sort_BICs(BICs_PoissonTheta, BICs_ZIPoissonTheta, BICs_NBTheta, BICs_ZINBTheta, BICs_BPTheta, BICs_ZIBPTheta, BICtol=10)
inds_PoissonTheta, inds_ZIPoissonTheta, inds_NBTheta, inds_ZINBTheta, inds_BPTheta, inds_ZIBPTheta = indsarr

# put best fit distributions for each gene and phase in separate arrays (convenient later)
fits = Vector{Distribution}(undef, ngenes)
fits[inds_PoissonTheta] = fits_PoissonTheta[inds_PoissonTheta]
fits[inds_ZIPoissonTheta] = fits_ZIPoissonTheta[inds_ZIPoissonTheta]
fits[inds_NBTheta] = fits_NBTheta[inds_NBTheta]
fits[inds_ZINBTheta] = fits_ZINBTheta[inds_ZINBTheta]
fits[inds_BPTheta] = fits_BPTheta[inds_BPTheta]
fits[inds_ZIBPTheta] = fits_ZIBPTheta[inds_ZIBPTheta]

# Check how many genes are best fit by specific distribution in G1 phase
println(length(inds_PoissonTheta))
println(length(inds_ZIPoissonTheta))
println(length(inds_NBTheta))
println(length(inds_ZINBTheta))
println(length(inds_BPTheta))
println(length(inds_ZIBPTheta))

# Filter out genes that at least in one phase are fit by Poisson or zero-inflated Poisson

inds = union(inds_PoissonTheta, inds_ZIPoissonTheta)
println("Genes filtered out: $(length(inds))")
inds = (1:ngenes)[1:end .∉ [inds]]
println("Genes left: $(length(inds))")