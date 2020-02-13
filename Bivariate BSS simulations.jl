using HypergeometricFunctions # for the Gauss hypergeometric 2F1 function
using LinearAlgebra 
using Distributions # for generating the multivariate normal
using Plots # for plotting the paths
using DSP # for convolutions 
using DelimitedFiles # for writing the dataset to CSV
using StatsBase
using SpecialFunctions

# Essentially uses the same functions and algorithm as the BSS process simulation
# Difference in the kernel which includes a rotational part

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
              Sigma[j,k] = 1/(alpha + 1)/n^(2*alpha + 1) * ((j - 1)^(alpha + 1) * 
              (k - 1)^alpha * _₂F₁(-alpha, 1, alpha + 2, (j - 1)/(k - 1) ) - 
              (j - 2)^(alpha + 1) * (k - 2)^alpha * _₂F₁(-alpha, 1, alpha + 2, (j - 2)/(k - 2) ))
            end
        end

    end
    # fill in lower triangle so that S[i,j] = S[j,i]
    # return matrix
    return Matrix(Symmetric(Sigma))

end


# kernal function split into x and y components
# kernel is unnormalised
# gamma kernel for each, parameters will be different in general

g_x(t, alpha, lambda) = t^alpha * exp(-lambda * t)
g_y(t, alpha, lambda) = t^alpha * exp(-lambda * t)


b_star(k, alpha) =  ((k^(alpha + 1) - (k-1)^(alpha + 1))/(alpha + 1))^(1/alpha)
# optimal discretisation b* value function 


function bivariateGammaKernelBSS(N::Int, n::Int, T::Int, kappa::Int, alphas, lambdas, rho) 
# simulating from a Matern process using the hybrid Scheme
# X and Y are now the two dimensions of the process

    # create empty vectors for the 'lower' part of the hybrid scheme sums
    X_lower = zeros(n*T + 1)
    Y_lower = zeros(n*T + 1)
    
    # split indices into lower and upper sums
    k_lower = 1:kappa
    k_upper = (kappa + 1):N

    # vector of the L_g(k/n)
    L_g_x = exp.(-lambdas[1].*k_lower/n) 
    L_g_y = exp.(-lambdas[2].*k_lower/n) 

    # vector of g(b*/n) for hybrid scheme
    g_b_star_x = [zeros(kappa); g_x.(b_star.(k_upper, alphas[1])/n, alphas[1], lambdas[1])]
    g_b_star_y = [zeros(kappa); g_y.(b_star.(k_upper, alphas[2])/n, alphas[2], lambdas[2])]
    ## generate the volatility process
   
    # create the required covariance matrix
    Sigma_W_x = hybridSchemeCovarianceMatrix(kappa, n, alphas[1])
    Sigma_W_y = hybridSchemeCovarianceMatrix(kappa, n, alphas[2])
    
    # generate the multivariate distribution
    d_x = MvNormal(Sigma_W_x)
    d_y = MvNormal(Sigma_W_y)

    # sample N + n*T random variables from this multivariate Gaussian
    # same Brownian increments need to be used for both X and Y directions
    W_x = transpose(rand(d_x, N + n*T))
    W_orthog = transpose(rand(d_y, N + n*T))

    W_y = rho .* W_x + sqrt(1 - rho^2) .* W_orthog
    ## create the sample hybrid scheme and Riemann sum sample paths
    # split into cases as when kappa = 1, we are dealing with a scalar not a matrix in the first sum
    if kappa == 1 
        # loop over each time i/n with i = 0, ..., n*T
        for i = 1:(n*T + 1) 
        # calculate X[i] = X(i-1/n) from hybrid scheme
        # sum first kappa terms and remaining N - kappa terms separately
            
            # generate the lower sums for the two directions
            X_lower[i] <- sum( L_g_x  * W_x[(i + N - kappa):(i+N-1), 2])
            Y_lower[i] = sum( L_g_y  .* W_y[(i + N - kappa):(i+N-1), 2])
        end

    else # if kappa > 1
        # loop over each time i/n with i = 0, ..., n*T
        for i = 1:(n*T + 1)
        # sum first kappa terms and remaining N - kappa terms separately
        
        # generate the lower sums for the two directions
        X_lower[i] = sum( L_g_x  .* diag(reverse(W_x[(i + N - kappa):(i+N-1), 2:(kappa + 1)], dims = 1)))        
        Y_lower[i] = sum( L_g_y  .* diag(reverse(W_y[(i + N - kappa):(i+N-1), 2:(kappa + 1)], dims = 1)))
        
        end
        
        # convolve with the Brownian increments in each dimension
        X = X_lower + conv( g_b_star_x, (W_x[:,1]))[N:(N+n*T)]
        Y = Y_lower + conv( g_b_star_y, (W_y[:,1]))[N:(N+n*T)] 
    end

    # return the sample path as (X,Y) coordinate vectors
    return X, Y

end



### to produce an entire simulated dataset: 

# set parameter values

const num_simulations = 10
const n = 25000
const N = 150000
const T = 1
const kappa = 5
const alphas = [-0.499, 0.499]
const lambdas = [1 1]
const rho = 1


bssp = bivariateGammaKernelBSS(N, n, T, kappa, alphas, lambdas, rho)
plot(bssp[1], bssp[2])

writedlm("x_coord.csv", cumsum(mp[1]), ",")
writedlm("y_coord.csv", cumsum(mp[2]), ",")




# create matrices to store the datasets

all_X = zeros(num_simulations, n*T + 1)
all_Y = zeros(num_simulations, n*T + 1)

# simulate data

for i = 1:num_simulations
    simulation = 
    all_X[i,:] = simulation[1] 
    all_Y[i,:] = simulation[2]
    println(i)
end

# write the data to csv files

writedlm("all_X.csv", all_X, ",")
writedlm("all_Y.csv", all_Y, ",")



