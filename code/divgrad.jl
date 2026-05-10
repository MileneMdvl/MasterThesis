#File to build the discrete divergence and gradient as matrix operators 

include("mesh_functions.jl")

#Function: Divergence 
#Input: C, F: Vector of size nc and nf respectively 
#list of cells and faces (where each face is only stored once)
#Output: D Array of size nc x nf, divergence operator 
function Divergence(C, F)
    nc = length(C)
    nf = length(F)
    D = NDSparseArray{Float64}(nc, nf)
    for i in 1:nc 
        for j in 1:nf 
            K = C[i]
            e = F[j]
            if isFace(e,K)
                D[i,j] = Volume(e)/Volume(K) * NormalIndicator(e,K)
            end
        end
    end
    return D 
end

#Function: Gradient 
#Input: C, F: Vector of size nc and nf respectively 
#       BF: Vector of faces which also lie on the boundary 
#list of cells and faces (where each face is only stored once)
#Output: G Array of size nf x nc, gradient operator 
function Gradient(C,F,BF)
    nc = length(C)
    nf = length(F)
    G = NDSparseArray{Float64}(nf,nc)
    for i in 1:nf
        for j in 1:nc
            K = C[j]
            e = F[i]
            if isFace(e,K) && e ∉ BF
                G[i,j] = - NormalIndicator(e,K) / DualEdge(e)
            end
        end
    end
    return G
end

