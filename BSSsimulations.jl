using HypergeometricFunctions # for the Gauss hypergeometric 2F1 function
using LinearAlgebra 
using Distributions # for generating the multivariate normal
using Plots # for plotting the paths
using DSP # for convolutions 
using DelimitedFiles # for writing the dataset to CSV
using StatsBase

# Essentially uses the same functions and algorithm as the R version, just faster


function hybridSchemeCovarianceMatrix(kappa, n, alpha)
# Covariance matrix used in the hybrid scheme simulations
    # create empty matrix
    Sigma = zeros(kappa + 1, kappa + 1)
    # fill in top corner = Var(W_i)
    Sigma[1,1] = 1/n
    # loop over other columns
    for j = 2:(kappa + 1)
        # fill in according to given expressions
        Sigma[1,j] = ((j-1)^(alpha+1) - (j-2)^(alpha+1))/(alpha+1)/n^(alpha+1)

        Sigma[j,j] = ((j-1)^(2*alpha+1) - (j-2)^(2*alpha+1))/(2*alpha+1)/n^(2*alpha+1)
        # fill in remaining rows
        if j < kappa + 1 
            for k = (j+1):(kappa+1)
              Sigma[j,k] = 1/(alpha + 1)/n^(2*alpha + 1) * ((j - 1)^(alpha + 1) * (k - 1)^alpha * _₂F₁(-alpha, 1, alpha + 2, (j - 1)/(k - 1) ) - (j - 2)^(alpha + 1) * (k - 2)^alpha * _₂F₁(-alpha, 1, alpha + 2, (j - 2)/(k - 2) ))
            end
        end

    end
    # fill in lower triangle so that S[i,j] = S[j,i]
    # return matrix
    return Matrix(Symmetric(Sigma))

end

function ornsteinUhlenbeck(N, n, T, theta)
# simulating the OU process
    v = zeros(Int(N + n*T + 1))
    # initialise values
    v[1] = 1 / sqrt(2*theta) * randn()
    # loop according to the stochastic differential equation form
    for i = 2:Int(N + n*T + 1)
        v[i] = v[i-1] - theta * v[i-1] * 1/n + sqrt(1/n) * randn()
    end
 
    return v
end


g(x, alpha, lambda) = x^alpha * exp(-lambda*x)
# gamma kernel function


b_star(k, alpha) =  ((k^(alpha + 1) - (k-1)^(alpha + 1))/(alpha + 1))^(1/alpha)
# optimal discretisation b* value function 


function gammaKernelBSS(N::Int, n::Int, T::Int, kappa::Int, alpha, lambda, theta) 
# simulating from a gamma kernel BSS process
    
    # create empty vectors for the 'lower' part of the hybrid scheme sums
    X_lower = zeros(n*T + 1)
    Y_lower = zeros(n*T + 1)
    
    # split indices into lower and upper sums
    k_lower = 1:kappa
    k_upper = (kappa + 1):N

    # vector of the L_g(k/n)
    L_g = exp.(-lambda.*k_lower/n)

    # vector of g(b*/n) for hybrid scheme
    g_b_star = [zeros(kappa); g.(b_star.(k_upper, alpha)/n, alpha, lambda)]
        
    ## generate the volatility process
    sigma = exp.(0.125 .* ornsteinUhlenbeck(N, n, T, theta))
    
    ## generate the Brownian increments according to hybrid scheme

    # create the required covariance matrix
    Sigma_W = hybridSchemeCovarianceMatrix(kappa, n, alpha)

    # generate the multivariate distribution
    d = MvNormal(Sigma_W)

    # sample N + n*T random variables from this multivariate Gaussian
    W = transpose(rand(d, N + n*T))

    ## create the sample hybrid scheme and Riemann sum sample paths
    # split into cases as when kappa = 1, we are dealing with a scalar not a matrix in the first sum
    if kappa == 1 
        # loop over each time i/n with i = 0, ..., n*T
        for i = 1:(n*T + 1) 
        # calculate X[i] = X(i-1/n) from hybrid scheme
        # sum first kappa terms and remaining N - kappa terms separately
            
            # generate the Gaussian core element
            X_lower[i] <- sum( L_g  * W[(i + N - kappa):(i+N-1), 2])
            
            # generate the BSS sample path element
            Y_lower[i] = sum( L_g  .* sigma[(i + N - kappa):(i+N-1)] .* W[(i + N - kappa):(i+N-1), 2])
        end

    else # if kappa > 1
        # loop over each time i/n with i = 0, ..., n*T
        for i = 1:(n*T + 1)
        # sum first kappa terms and remaining N - kappa terms separately
        
        # for the Gaussian core
        X_lower[i] = sum( L_g  .* diag(reverse(W[(i + N - kappa):(i+N-1), 2:(kappa + 1)], dims = 1)))    
        
        # for the BSS process
        Y_lower[i] = sum( L_g  .* sigma[(i + N - kappa):(i+N-1)] .* diag(reverse(W[(i + N - kappa):(i+N-1), 2:(kappa + 1)], dims = 1)))
        
        end
        
        # Gaussian core, convolve only with Brownian increments
        X = X_lower + conv( g_b_star, (W[:,1]))[N:(N+n*T)]
        
        # BSS sample path, convolve with volatility process * Brownian increments
        Y = Y_lower + conv( g_b_star, (sigma[1:(end-1)] .* W[:,1]))[N:(N+n*T)] 
    end

    # return the Gaussian core X, the BSS process Y, and the volatility process sigma
    return X, Y, sigma[(end - n*T): end]
end



### to produce the entire simulated dataset: 

# set parameter values

num_simulations = 10
n = 25000
N = 150000
T = 1
kappa = 3
alpha = -0.2
lambda = 1
theta = 2

# create matrices to store the datasets

all_X = zeros(num_simulations, n*T + 1)
all_Y = zeros(num_simulations, n*T + 1)
all_sigma = zeros(num_simulations, n*T + 1)

# simulate data

for i = 1:num_simulations
    simulation = gammaKernelBSS(N, n, T, kappa, alpha, lambda, theta)
    all_X[i,:] = simulation[1]    
    all_Y[i,:] = simulation[2]
    all_sigma[i,:] = simulation[3]
    println(i)
end

# write the data to csv files

writedlm("all_X.csv", all_X, ",")
writedlm("all_Y.csv", all_Y, ",")
writedlm("all_sigma.csv", all_sigma, ",")


plot(autocor(all_Y[2,:], 1:500))


plot(all_Y[1,:])





