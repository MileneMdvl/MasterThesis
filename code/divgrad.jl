#File to build the discrete divergence and gradient as matrix operators 

#Inputs for all functions are list of cells, faces and boundary faces 

include("mesh_functions.jl")

#Function: Divergence 
#Input: C, F: Vector of size nc and nf respectively 
#list of cells and faces (where each face is only stored once)
#Output: D Array of size nc x nf, divergence operator 
function Divergence(C, F)
    nc = length(C)
    nf = length(F)
    D = spzeros(nc,nf)
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
    G = spzeros(nf,nc)
    for i in 1:nf
        for j in 1:nc
            K = C[j]
            e = F[i]
            # if isFace(e,K) && e ∉ BF
            #     G[i,j] = - NormalIndicator(e,K) / DualEdge(e)
            # end
            if isFace(e,K)
                G[i,j] = -NormalIndicator(e,K) / Volume(e) 
            end
        end
    end
    return G
end

#Function Convection 
#Input: uf face-centred velocity 
#       phi_c cell-centred vector to take the convection of 
#Output: conv, the discretisation of u⋅∇ϕ 
function Convection(uf,phi_c)
    nc = length(cell_list)
    conv = zeros(size(phi_c))
    #Compute the gradient of phi_c 
    if phi_c isa Vector 
        Gphi_c = SparseMatVec(G,phi_c)
    elseif phi_c isa Matrix 
        Gphi_c = SparseMatMat(G,phi_c)
    end
    for i in 1:nc 
        local K = cell_list[i]
        local e_K = Faces(K)
        for k in 1:3
            e = e_K[k,:]
            ind_e = findall(x->x==e, face_list)[1]
            conv[i,:] += DualEdge(e) * Volume(e) * uf[ind_e] * Gphi_c[ind_e,:]
        end
    end
    return conv 
end
