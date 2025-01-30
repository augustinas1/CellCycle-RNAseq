### Convolution functions ###

function conv(x1::Array{T}, x2::Array{T}) where T
    y = zeros(T, length(x1))
    for i in eachindex(x1)
        for j in 1:i
            y[i] += x1[j]*x2[i-j+1]
        end
    end
    y
end

conv2(x::AbstractArray) = conv(x, x)
conv4(x::AbstractArray) = conv2(conv2(x))

function convn(x::AbstractArray, n::Int)
    # convolution of an array with itself n times sequentially
    # e.g. for n=2 (f * f) will call convolve(x, x) once 
   
    isone(n) && return x
    
    y = x
    for i in n:-2:2
        y = conv2(y)
    end
    
    if isodd(n)
        conv(y, x)
    else
        y
    end
    
end

function logsumexp(w::Vector{T}) where T
    # based on discussion in https://discourse.julialang.org/t/fast-logsumexp/22827/6
    offset, ind = findmax(w)
    N = length(w)
    s = zero(T)
    
    for i = 1:ind-1
        s += exp(w[i]-offset)
    end
    for i = ind+1:N
        s += exp(w[i]-offset)
    end
    
    log1p(s) + offset

end

function logconv(lx1::Array{T}, lx2::Array{T}) where T
    # using log-sum-exp trick on log{P(z)} = log{ sum_{k=0}^N exp(c)exp(log{p(k)} + log{p(z-k)})/exp(c) }
    ly = similar(lx1)
    for i in eachindex(lx1)
        z = zeros(T, i)
        for j in 1:i
            z[j] = lx1[j] + lx2[i-j+1]
        end
        ly[i] = logsumexp(z) 
    end
    ly
end

logconv2(x::AbstractArray) = logconv(x, x)
logconv4(x::AbstractArray) = logconv2(logconv2(x))

function logconvn(x::AbstractArray, n::Int)
   
    isone(n) && return x
    
    y = x
    for i in n:-2:2
        y = logconv2(y)
    end
    
    if isodd(n)
        logconv(y, x)
    else
        y
    end
    
end

function resolve_logconv(x, nconv::Int)   
    nconv > 0 || throw(DomainError(nconv, "nconv < 1"))
    if nconv == 1
        identity(x)
    elseif nconv == 2
        logconv2(x)
    elseif nconv == 4
        logconv4(x)
    else
        logconvn(x, nconv)
    end
end