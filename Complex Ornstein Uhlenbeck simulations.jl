using LinearAlgebra 
using Distributions # for generating the multivariate normal
using Plots # for plotting the paths
using DSP # for convolutions 
using DelimitedFiles # for writing the dataset to CSV
using StatsBase
using SpecialFunctions

# function to generate a complex valued OU process

function complexOrnsteinUhlenbeck(n::Int, T::Int, ω, c, A)
    # construct expty array and initialise from stationary distrbution     
    Z = Array{ComplexF64}(undef, n*T + 1)
    Z[1] = A/sqrt(2c) * randn()

    # loop over indices updating according to the SDE
    for i = 2:n*T + 1
        dW = randn() + randn()im
        Z[i] = ((ω*im - c)/n + 1) * Z[i-1] + A * dW
    end
    # return the vector of complex valued z = x + iy OU process
    return Z

end


OUprocess = complexOrnsteinUhlenbeck(25000, 1, 40, 1, 4)
plot(OUprocess)


writedlm("x_coord.csv", real(OUprocess), ",")
writedlm("y_coord.csv", imag(OUprocess), ",")




