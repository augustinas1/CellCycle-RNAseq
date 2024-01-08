include("load_data.jl")

# merged data G1 + G2/M
merged_inds = vcat(G1_inds, G2M_inds)
theta_merged = theta[merged_inds]
println("$(length(merged_inds)) cells in the merged (G1 + G2M) dataset")
counts_merged = vcat.(xG1, xG2M);