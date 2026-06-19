"
This file contains the functions for the discrete divergence, gradient and convection. Furthermore, for the convection we include some regularisation, as detailed in [1]. 

The detail of each function is given in my master thesis. 

These operators should hold for both two and three dimensions. 

[1] On restraining the prodvction of small scales of motion in a turbulent channel flow, R. Verstappen, 2008. 
"


"
Function: 
    Divergence 

Input: 
    C: Vector{Vector{Int}} (cell_list)
    F: Vector{Vector{Int}} (face_list)

Output: 
    D: Matrix{Float}
    of size nc x nf, where nc is the number of cells and nf the number of faces in the triangulation 

In order to speed up computations, for each cell we only look at the faces surrounding it, and we do this using the cf_info dictionary.
"
function Divergence(C, F)
    nc = length(C)
    nf = length(F)
    D = spzeros(nc,nf)
    for i in 1:nc 
        K = C[i]
        for j in cf_info[i]
            e = F[j]
            D[i,j] = Volume(e)/Volume(K) * NormalIndicator(e,K)
        end
    end
    return D 
end

#Function: Gradient 
#Input: C, F: Vector of size nc and nf respectively 
#       BF: Vector of faces which also lie on the boundary 
#list of cells and faces (where each face is only stored once)
#Output: G Array of size nf x nc, gradient operator 
"
Function: 
    Gradient 

Input: 
    C: Vector{Vector{Int}} (cell_list)
    F: Vector{Vector{Int}} (face_list)

Output: 
    G: Matrix{Float}
    of size nf x nc

In order to speed up computations, for each cell we only look at the faces surrounding it, and we do this using the cf_info dictionary.
"
function Gradient(C,F)
    nc = length(C)
    nf = length(F)
    G = spzeros(nf,nc)
    for j in 1:nc
        K = C[j]
        for i in cf_info[j]
            e = F[i]
            G[i,j] = - NormalIndicator(e,K) / DualEdge(e)
        end
    end
    return G
end

"
Function: 
    Convection 

Input: 
    uf: Vector{Float}
        face-centred velocity 
    vc: Matrix{Float}
        cell-centred velocity of which we aim to take the convection

Output: 
    conv: Matrix{Float}
        u⋅∇v, cell-centred convection of v

In order to speed up computations, for each cell we only look at the faces surrounding it, and we do this using the cf_info dictionary.
"
function Convection(uf,vc)
    conv = zeros(size(vc))
    #Compute the gradient of vc 
    Gvc = G*vc
    for i in 1:nc 
        local K = cell_list[i]
        for j in cf_info[i]
            e = face_list[j]
            conv[i,:] += DualEdge(e) * Volume(e) * uf[j] * Gvc[j,:]
        end
    end
    return conv
end


"
The 2nd, 4th and 6th order regularisations from Verstappen (2008). These regularisations rely on some predefined filter. 

If the order chosen is zero then this simply returns the convection defined above. 
"
function RegularisedConvection(Filter,uf,vc,order::Int)
    vc_filt = Filter * vc
    vc_res = vc - vc_filt
    uf_filt = CellToFaceInterpolation(Filter * vc)
    uf_res = uf - uf_filt
    if order == 2
        c2 = Filter*Convection(uf_filt,vc_filt)
        return c2 
    elseif order == 4 
        c4 = Convection(uf_filt,vc_filt) + Filter * Convection(uf_filt,vc_res) + Filter * Convection(uf_res,vc_filt)
        return c4 
    elseif order == 6 
        c6 = Convection(uf_filt,vc_filt) + Convection(uf_filt,vc_res) + Convection(uf_res,vc_filt) + Filter * Convection(uf_res,vc_res) 
        return c6 
    elseif order == 0
        return Convection(uf,vc)
    end
end
