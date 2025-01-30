# --- Filtering after inference using the age-dependent models ---

# --- Step 1 ---

# Remove all genes for which the predicted mean expression given by the age-dependent model in either G1 or G2/M 
# cell cycle phase is negatively correlated with the cell age θ. Usually indicative of a bad model fit (due to larger 
# deviations from the ratio of 2 between the cells in θ_f and in θ_i).

inds = Vector{Bool}(undef, ngenes)
for i in 1:ngenes
    mG1 = mean.(Ref(fits_main[i]), θs_G1, Ref(T_cycle), Ref(decay_rates[i]), Ref(θᵣ), Ref(θ_G1_S), Ref(θ_S_G2M))
    mG2M = mean.(Ref(fits_main[i]), θs_G2M, Ref(T_cycle), Ref(decay_rates[i]), Ref(θᵣ), Ref(θ_G1_S), Ref(θ_S_G2M))
    inds[i] = (cor(mG1, θs_G1) .> 0) .&& (cor(mG2M, θs_G2M) .> 0)
end
inds = findall(inds)

xG1 = xG1[inds]
xS = xS[inds]
xG2M = xG2M[inds]

counts_spliced = counts_spliced[inds]
gene_names = gene_names[inds]
decay_rates = decay_rates[inds]

G1_th_ind_fits = G1_th_ind_fits[inds]
G2M_th_ind_fits = G2M_th_ind_fits[inds]
fits_main = fits_main[inds]

ndiff = ngenes - length(inds)
ngenes = length(inds)

println("Removed $ndiff genes with negative correlation between the mean predicted by the cell division model and cell age θ.")
println("$ngenes genes left remaining.")

# --- Step 2 ---
# Remove genes that have a clearly bad fit -- indicated by a negative R^2 value either for the mean or variance over the entire cell cycle

function compute_rsq_mean(ind::Int, m::DiscreteUnivariateDistribution=fits_main[ind])
    counts_G1 = [xG1[ind][_inds] for _inds in inds_θs_G1]
    counts_S = [xS[ind][_inds] for _inds in inds_θs_S]
    counts_G2M = [xG2M[ind][_inds] for _inds in inds_θs_G2M]

    yG1 = mean.(counts_G1)
    yS = mean.(counts_S)
    yG2M = mean.(counts_G2M)
    y = vcat(yG1, yS, yG2M)

    ms = mean.(Ref(m), θs, Ref(T_cycle), Ref(decay_rates[ind]), Ref(θᵣ), Ref(θ_G1_S), Ref(θ_S_G2M))
    r2 = 1 - sum((y .- ms).^2) / sum((y.- mean(y)).^2)
    r2
end

function compute_rsq_var(ind::Int, m::DiscreteUnivariateDistribution=fits_main[ind])
    counts_G1 = [xG1[ind][_inds] for _inds in inds_θs_G1]
    counts_S = [xS[ind][_inds] for _inds in inds_θs_S]
    counts_G2M = [xG2M[ind][_inds] for _inds in inds_θs_G2M]

    yG1 = var.(counts_G1)
    yS = var.(counts_S)
    yG2M = var.(counts_G2M)
    y = vcat(yG1, yS, yG2M)

    vars = var.(Ref(m), θs, Ref(T_cycle), Ref(decay_rates[ind]), Ref(θᵣ), Ref(θ_G1_S), Ref(θ_S_G2M))
    r2 = 1 - sum((y .- vars).^2) / sum((y.- mean(y)).^2)
    r2
end

mean_rsqs = compute_rsq_mean.(1:ngenes)
var_rsqs = compute_rsq_var.(1:ngenes)

inds = findall(mean_rsqs .> 0 .&& var_rsqs .> 0)

xG1 = xG1[inds]
xS = xS[inds]
xG2M = xG2M[inds]

counts_spliced = counts_spliced[inds]
gene_names = gene_names[inds]
decay_rates = decay_rates[inds]

G1_th_ind_fits = G1_th_ind_fits[inds]
G2M_th_ind_fits = G2M_th_ind_fits[inds]
fits_main = fits_main[inds]

ndiff = ngenes - length(inds)
ngenes = length(inds)

println("Removed $ndiff genes with negative R^2 in variance.")
println("$ngenes genes left remaining.");