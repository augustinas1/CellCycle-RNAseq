### Log-likelihood computation

function log_likelihood(d::DiscreteUnivariateDistribution, nconv::Int, y::AbstractArray)
    yy, ww = fit_hist(y) 
    convf(x) = resolve_logconv(x, nconv)   
    sum(convf(logpdf.(d, yy)) .* ww)
end

### Bayesian Information Criterion

function get_BIC(d::Distribution, nconv::Int, y::AbstractArray) 
    # the larger the better (negative of the standard BIC)
    2*log_likelihood(d, nconv, y) - numparams(d)*log(length(y))
end

get_BICs(fits::AbstractArray, nconv::Int, ydata::AbstractArray) = @views [get_BIC(fits[i], nconv, ydata[i]) for i in eachindex(ydata)]
    
function sort_BICs(BICs...; BICtol=10)
    
    BICmat = hcat(BICs...)
    BICmat = transpose(BICmat)
    
    indsarr = []
    rinds = 1:size(BICmat)[2]

    niters = size(BICmat)[1]-1
    for i in 1:niters
        # compute BICs relative to i-th distribution fits
        for c in eachcol(BICmat)
            c .-= c[1]
        end
        temp_inds = findall([maximum(c) < BICtol for c in eachcol(BICmat) ])
        inds = rinds[temp_inds]
        push!(indsarr, inds)

        # remove genes best fit by i-th distribution
        rinds = setdiff(rinds, inds)
        if i == niters
            push!(indsarr, rinds);
        else
            BICmat = BICmat[2:end, setdiff(1:size(BICmat)[2], temp_inds)]
        end
    end
    
    indsarr
    
end
