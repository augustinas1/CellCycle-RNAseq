if !@isdefined(inds) || !@isdefined(fitpath)
    include("model_selection.jl")
end

println("Computing confidence intervals")

# using profile likelihood
include(srcpath*"confidence.jl")

# test compilation
#xdata = data[inds[1:2]]
#xfits = fits[inds[1:2]]
#@time CIs_PL = get_confidence_intervals(xfits, xdata, get_confidence_intervals_PL; nconv, time_limit=600.0)

xdata = data[inds]
xfits = fits[inds]
@time CIs_PL = get_confidence_intervals(xfits, xdata, get_confidence_intervals_PL; nconv, time_limit=600.0)

# Save the best fits and their corresponding confidence intervals
th_ind_res = Dict("gene_inds" => inds, "fits" => fits[inds], "CIs" => CIs_PL)
@save fitpath*"merged_th_ind_CI_results.jld2" th_ind_res 