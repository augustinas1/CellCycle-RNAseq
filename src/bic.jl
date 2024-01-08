### Log-likelihood computation

function log_likelihood(d::DiscreteUnivariateDistribution, nconv::Int, y::AbstractArray)
    yy, ww = fit_hist(y) 
    convf(x) = resolve_logconv(x, nconv)   
    sum(convf(logpdf.(d, yy)) .* ww)
end

function log_likelihood(d::DiscreteUnivariateDistribution, nconv::Int, y::AbstractArray, theta::AbstractArray)
    thetas = sort(unique(theta))
    yw = fit_hist.(y[theta .== th] for th in thetas)
    convf(x) = resolve_logconv(x, nconv)
    
    ll = 0.0
    @views for i in eachindex(thetas)
        prob = logpdf.(Ref(d), thetas[i], yw[i][1])
        ll += sum(convf(prob) .* yw[i][2])
    end
    ll
end

### Bayesian Information Criterion

function get_BIC(d::Distribution, nconv::Int, y::AbstractArray) 
    # the larger the better (negative of the standard BIC)
    2*log_likelihood(d, nconv, y) - numparams(d)*log(length(y))
end

get_BICs(fits::AbstractArray, nconv::Int, ydata::AbstractArray) = @views [get_BIC(fits[i], nconv, ydata[i]) for i in eachindex(ydata)]

# for theta-dependent fits
function get_BIC(d::Distribution, nconv::Int, y::AbstractArray, theta::AbstractArray) 
    # the larger the better
    2*log_likelihood(d, nconv, y, theta) - numparams(d)*log(length(y))
end

function get_BICs(fits::AbstractArray, nconv::Int, ydata::AbstractArray, theta::AbstractArray)
    fit_inds = findall([isassigned(fits, i) for i in eachindex(fits)])
    BICs = fill(-Inf, length(fits))
    @views Threads.@threads for i in fit_inds
		BICs[i] = get_BIC(fits[i], nconv, ydata[i], theta)			
	end
	BICs
end
    
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
