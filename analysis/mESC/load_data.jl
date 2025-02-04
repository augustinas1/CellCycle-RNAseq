using InteractiveUtils
versioninfo()
println()

using PyCall, JLD2, StatsBase

# read in the data
datapath = joinpath(@__DIR__, "../../data/mESC/")
anndata = pyimport("anndata")
adata = anndata.read_h5ad(datapath*"mESC_data.h5ad")
gene_names = adata.var_names.tolist()

phase = adata.obs_vector("phase")
phase = phase.tolist()
counts_spliced = adata.layers.get("discrete_spliced").toarray()
counts_spliced = Int.(counts_spliced)
ngenes = size(counts_spliced)[2]

theta = adata.obs_vector("cell_cycle_theta")
inds = sortperm(theta)

counts_spliced = counts_spliced[inds, :]
phase = phase[inds]
G1_inds = findall(phase .== "G1")
S_inds = findall(phase .== "S")
G2M_inds = findall(phase .== "G2M")

theta = theta[inds]
thetaG1 = theta[G1_inds]
thetaS = theta[S_inds]
thetaG2M = theta[G2M_inds]

println("$(length(G1_inds)) cells in G1")
println("$(length(S_inds)) cells in S")
println("$(length(G2M_inds)) cells in G2M")

theta_all_cells = deepcopy(theta)

# remove all cells assigned to θ ∈ [0, 0.09] range and rescale the remaining θ values into the same [0, 0.99] range
th0 = 0.1
inds = findall(theta .>= th0)
theta = theta[inds]
theta = (theta .- th0) ./ (0.99 - th0) .* 0.99
phase = phase[inds]
counts_spliced = counts_spliced[inds, :]

rind = inds[1]-1
println("Cut off $rind cells in G1")
G1_inds = G1_inds[inds[1]:end] .- rind
S_inds .-= rind
G2M_inds .-= rind

thetaG1 = theta[G1_inds]
thetaS = theta[S_inds]
thetaG2M = theta[G2M_inds]

xG1 = @views [eachcol(counts_spliced[G1_inds, :])...]
xS = @views [eachcol(counts_spliced[S_inds, :])...]
xG2M = @views [eachcol(counts_spliced[G2M_inds, :])...]

# remove all genes with log expression (mean abundance lower than one in either G1, S or G2/M phases)
inds1 = findall(mean.(xG1) .>= 1)
inds2 = findall(mean.(xS) .>= 1)
inds3 = findall(mean.(xG2M) .>= 1)
inds = intersect(inds1, inds2, inds3)
println("Considering $(length(inds)) genes out of $ngenes in total")
ngenes = length(inds)
xG1 = xG1[inds]
xS = xS[inds]
xG2M = xG2M[inds]
counts_spliced = @views [eachcol(counts_spliced[:, inds])...]
gene_names = gene_names[inds]