using JLD 
using LinearAlgebra

err_u = zeros(6)

data_6 = load("data/data_6.jld")
data_12 = load("data/data_12.jld")
data_24 = load("data/data_24.jld")
data_48 = load("data/data_48.jld")
data_96 = load("data/data_96.jld")

err_u[1] = norm(data_6["uf"]-data_6["uf0"])