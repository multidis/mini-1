using HypergeometricFunctions # for the Gauss hypergeometric 2F1 function
using LinearAlgebra 
using Distributions # for generating the multivariate normal
using Plots # for plotting the paths
using DSP # for convolutions 
using DelimitedFiles # for writing the dataset to CSV
using StatsBase

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

    




### Testing the code reproduces the R code

# test the Covariance matrix returns the same result

for kappa = 1:3
    for n = 10:10:20
        for alpha = -0.45:0.05:0.45
            display(hybridSchemeCovarianceMatrix(kappa, n, alpha))
        end
    end
end

# All ok 



n = 25000
N = 150000
T = 1
kappa = 3
alpha = -0.2
lambda = 1
theta = 2



X_lower = zeros(n*T + 1)
Y_lower = zeros(n*T + 1)

# split indices into lower and upper sums
k_lower = 1:kappa
k_upper = (kappa + 1):N

# vector of the L_g(k/n)
L_g = exp.(-lambda.*k_lower/n)

# vector of g(b*/n) for hybrid scheme
g_b_star = [zeros(kappa); g.(b_star.(k_upper, alpha)/n, alpha, lambda)]
    

## generate the Brownian increments according to hybrid scheme




# test to see if given sigma and W, the two reproduce the same results 

# generate the Brownian increments and volatility process 
## generate the volatility process

Sigma_W = hybridSchemeCovarianceMatrix(kappa, n, alpha)
d = MvNormal(Sigma_W)
W = transpose(rand(d, N + n*T))

sigma = exp.(0.125 .* ornsteinUhlenbeck(N, n, T, theta))

# write them to file so that they can be read into R and used there

writedlm("W.csv", W, ",")
writedlm("sigma.csv", sigma, ",")

# read from the files 
W = readdlm("W.csv", ',', Float64, '\n')
sigma = readdlm("sigma.csv", ',', Float64, '\n')

# do the hybrid scheme simulation


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


# compare results with the results from R eg plot them etc

plot(X)
plot(Y)

 # looks ok compared with R - all values in the X and Y sample paths are the same (up to decimal accuracy)!