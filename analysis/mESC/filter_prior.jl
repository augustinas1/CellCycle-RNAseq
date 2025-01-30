# --- Filtering prior to inference using the age-dependent models ---

# --- Step 1 ---
# Filter by decay rates
# units: h^(-1)

using XLSX

xf = XLSX.readxlsx(datapath*"decay_rates.xlsx")
sh = xf["Sheet1"]
# geneSymbol column
sh_gene_names = sh["B7:B19983"]
sh_gene_names = vcat(sh_gene_names...)

# MC1 LIF+ column
sh_decay_rates_MC1 = sh["C7:C19983"]
sh_decay_rates_MC1 = vcat(Float64.(sh_decay_rates_MC1)...)
# MC2 LIF+ column
sh_decay_rates_MC2 = sh["D7:D19983"]
sh_decay_rates_MC2 = vcat(Float64.(sh_decay_rates_MC2)...)

# Consider only those genes that have positive decay rates
# Take MC1 values mainly and use MC2 values only when the MC1 rate is negative but the MC2 one is positive
inds1 = findall(sh_decay_rates_MC1 .> 0)
inds2 = findall(sh_decay_rates_MC2 .> 0)

sh_decay_rates = vcat(sh_decay_rates_MC1[inds1], sh_decay_rates_MC2[setdiff(inds2, inds1)])
sh_gene_names = sh_gene_names[vcat(inds1, setdiff(inds2, inds1))]

inds = findall(gene in sh_gene_names || gene*"*" in sh_gene_names for gene in gene_names)
xG1 = xG1[inds]
xS = xS[inds]
xG2M = xG2M[inds]
counts_spliced = counts_spliced[inds]
gene_names = gene_names[inds]

G1_th_ind_fits = G1_th_ind_fits[inds]
G2M_th_ind_fits = G2M_th_ind_fits[inds]
ndiff = ngenes - length(inds)
ngenes = length(inds)

inds = [findfirst(sh_gene_names .== gene .|| sh_gene_names .== gene*"*") for gene in gene_names]
decay_rates = sh_decay_rates[inds]

println("Removed $ndiff genes without a measured decay rate.")
println("$ngenes genes left remaining.")

# --- Step 2 ---
# Remove all genes whose mean expression either in G1 or G2/M cell cycle phase is negatively correlated with the cell age θ.

function get_corr_G1(ind::Int)
    counts_G1 = [xG1[ind][_inds] for _inds in inds_θs_G1]
    mx_G1 = mean.(counts_G1)
    yG1 = mx_G1
    cor(θs_G1, yG1)
end

function get_corr_G2M(ind::Int)
    counts_G2M = [xG2M[ind][_inds] for _inds in inds_θs_G2M]
    mx_G2M = mean.(counts_G2M)
    yG2M = mx_G2M
    cor(θs_G2M, yG2M)
end;

corrs_G1 = get_corr_G1.(1:ngenes)
corrs_G2M = get_corr_G2M.(1:ngenes)

# remove all genes that have negative correlation wrt. θ in G1 or G2/M phase
inds = findall(corrs_G1 .> 0 .&& corrs_G2M .> 0)

xG1 = xG1[inds]
xS = xS[inds]
xG2M = xG2M[inds]

counts_spliced = counts_spliced[inds]
gene_names = gene_names[inds]
decay_rates = decay_rates[inds]

G1_th_ind_fits = G1_th_ind_fits[inds]
G2M_th_ind_fits = G2M_th_ind_fits[inds]

ndiff = ngenes - length(inds)
ngenes = length(inds)

println("Removed $ndiff genes with negative correlation between mean transcription and cell age θ.")
println("$ngenes genes left remaining.")