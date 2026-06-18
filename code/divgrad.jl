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
        K = C[i]
        for j in cf_info[i]
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
    for j in 1:nc
        K = C[j]
        for i in cf_info[j]
            e = F[i]
            if isFace(e,K)
                G[i,j] = - NormalIndicator(e,K) / DualEdge(e)
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
    Gphi_c = G*phi_c
    for i in 1:nc 
        local K = cell_list[i]
        for j in cf_info[i]
            e = face_list[j]
            conv[i,:] += DualEdge(e) * Volume(e) * uf[j] * Gphi_c[j,:]
        end
    end
    return conv
end

#Regularised convections from  Verstappen 
function RegularisedConvection(Filter,uf,uc,order::Int)
    uc_filt = Filter * uc
    uc_res = uc - uc_filt
    uf_filt = CellToFaceInterpolation(Filter * uc)
    uf_res = uf - uf_filt
    if order == 2
        c2 = Filter*Convection(uf_filt,uc_filt)
        return c2 
    elseif order == 4 
        c4 = Convection(uf_filt,uc_filt) + Filter * Convection(uf_filt,uc_res) + Filter * Convection(uf_res,uc_filt)
        return c4 
    elseif order == 6 
        c6 = Convection(uf_filt,uc_filt) + Convection(uf_filt,uc_res) + Convection(uf_res,uc_filt) + Filter * Convection(uf_res,uc_res) 
        return c6 
    elseif order == 0
        return Convection(uf,uc)
    end
end
