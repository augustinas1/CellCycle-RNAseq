if !@isdefined(inds) || !@isdefined(fitpath)
    include("model_selection.jl")
end

println("Computing confidence intervals")

# using profile likelihood
include(srcpath*"confidence.jl")

# test compilation
#xdata = xG1[inds[1:10]]
#xfits = fits_G1[inds[1:10]]
#nconv = 2
#@time CIs_G1_PL = get_confidence_intervals(xfits, xdata, get_confidence_intervals_PL; nconv, time_limit=60.0)

# G1 phase
xdata = xG1[inds]
xfits = fits_G1[inds]
nconv = 2
@time CIs_G1_PL = get_confidence_intervals(xfits, xdata, get_confidence_intervals_PL; nconv, time_limit=120.0)

# G2/M phase
xdata = xG2M[inds]
xfits = fits_G2M[inds]
nconv = 4
@time CIs_G2M_PL = get_confidence_intervals(xfits, xdata, get_confidence_intervals_PL; nconv, time_limit=120.0)

# Save the best fits and their corresponding confidence intervals
th_ind_res = Dict("gene_inds" => inds, "G1_fits" => fits_G1[inds], "G2M_fits" => fits_G2M[inds], 
                   "G1_CIs" => CIs_G1_PL, "G2M_CIs" => CIs_G2M_PL)

@save fitpath*"th_ind_CI_results.jld2" th_ind_res