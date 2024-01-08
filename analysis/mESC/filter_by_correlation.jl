θs_G1 = sort(unique(thetaG1))
θs_S = sort(unique(thetaS))
θs_G2M = sort(unique(thetaG2M))

inds_θs_G1 = [findall(th .== thetaG1) for th in θs_G1]
inds_θs_S = [findall(th .== thetaS) for th in θs_S]
inds_θs_G2M = [findall(th .== thetaG2M) for th in θs_G2M]

_xG1 = log(2) .* θs_G1
_xS = log(2) .* θs_S
_xG2M = log(2) .* θs_G2M

θ_G1_f = maximum(thetaG1)
θ_S_i = minimum(thetaS)
θ_S_f = maximum(thetaS)
θ_G2M_i = minimum(thetaG2M)
θ_G1_S = θ_G1_f + (θ_S_i - θ_G1_f)/2
θ_S_G2M = θ_S_f + (θ_G2M_i - θ_S_f)/2;

function get_corr_G1(ind::Int)
    counts_G1 = [xG1[ind][_inds] for _inds in inds_θs_G1]
    mx_G1 = mean.(counts_G1)
    yG1 = log.(mx_G1)
    cor(_xG1, yG1)
end

function get_corr_S(ind::Int)
    counts_S = [xS[ind][_inds] for _inds in inds_θs_S]
    mx_S = mean.(counts_S)
    yS = log.(mx_S)
    cor(_xS, yS)
end

function get_corr_G2M(ind::Int)
    counts_G2M = [xG2M[ind][_inds] for _inds in inds_θs_G2M]
    mx_G2M = mean.(counts_G2M)
    yG2M = log.(mx_G2M)
    cor(_xG2M, yG2M)
end

r1 = get_corr_G1.(1:ngenes)
r2 = get_corr_S.(1:ngenes)
r3 = get_corr_G2M.(1:ngenes)

rinds = union(findall(r1 .< 0), findall(r2 .< 0), findall(r3 .< 0))
inds = setdiff(1:ngenes, rinds)

xG1 = xG1[inds]
xS = xS[inds]
xG2M = xG2M[inds]
counts_spliced = counts_spliced[inds]
counts_merged = counts_merged[inds]
gene_names = gene_names[inds]

G1_th_ind_fits = G1_th_ind_fits[inds]
G2M_th_ind_fits = G2M_th_ind_fits[inds]
G1_th_dep_fits = G1_th_dep_fits[inds]
G2M_th_dep_fits = G2M_th_dep_fits[inds]
merged_th_ind_fits = merged_th_ind_fits[inds]
merged_th_dep_fits = merged_th_dep_fits[inds]
ngenes = length(inds)

println("Removed $(length(rinds)) genes with r < 0.")
println("$ngenes genes left remaining.")