"
This file contains the functions to implement the face-to-cell and cell-to-face interpolations. Both my interpolation and the one from Trias et al. [1] are implemented, with the toggle Trias to change between the two. 

[1] Symmetry-preserving discretization of Navier-Stokes equations on collocated unstructured grids, F. X. Trias et al., 2014. 
"

"
Function: 
    FaceToCellInterpolation

Input: 
    phi_f: Vector{Float}
        face-centred vector 
    Trias: Boolean 
        if true, then the interpolation from [1] is computed 
        if false, then the interpolation is the one given in my master thesis

Output: 
    phi_c: Matrix{Float}
        cell-centred matrix (in the case of the velocity which has components in each spatial direction)
"
function FaceToCellInterpolation(phi_f,Trias=false)
    nf = length(face_list)
    nc = length(cell_list)
    #Get dimension of the problem
    d = length(face_list[1])
    phi_c = zeros(nc,d)
    if nf != length(phi_f)
        println("Error: vector not defined on face-centres")
    else 
        for i in 1:nc 
            K = cell_list[i]
            num = [0; 0]
            denom = 0
            # denom = Volume(K)
            for j in cf_info[i]
                e = face_list[j] 
                num += phi_f[j] * Volume(e) * NormalVector(e) 
                if Trias 
                    denom = Volume(K)
                else 
                    denom += Volume(e)
                end
            end
            phi_c[i,:] = num/denom 
        end
    end
    return phi_c 
end

#Function FaceToCellInterpolation
#Input:  phi_c, cell-centred matrix, of dimension nc * d 
#        where d is the dimension of the problem 
#Output: phi_f, face-centred vector
"
Function: 
    FaceToCellInterpolation

Input: 
    phi_c: Matrix{Float}
        cell-centred matrix (in the case of the velocity, otherwise it is simply a vector)
    Trias: Boolean 
        if true, then the interpolation from [1] is computed 
        if false, then the interpolation is the one given in my master thesis

Output: 
    phi_f: Vector{Float}
        face-centred vector

Furthermore, note that here the cell-to-face interpolation also includes the Neumann boundary conditions (as explained in the thesis)
"
function CellToFaceInterpolation(phi_c,Trias=false)
    d = length(face_list[1])
    nf = length(face_list)
    nc = length(cell_list)
    if phi_c isa Vector || size(phi_c)[1] != nc
        println("Error: input should be a matrix of dimension ",(nc,d))
        return 
    end
    phi_f = zeros(nf)
    for i in 1:nf 
        e = face_list[i]
        num = 0
        denom = 0
        for j in AdjInds(e)
            K = cell_list[j]
            if e ∉ boundary_list
                if Trias 
                    num += dot(phi_c[j,:],NormalVector(e))
                    denom = 2
                else
                    num += dot(phi_c[j,:],NormalVector(e)) * Volume(K)
                    denom += Volume(K) 
                end
            else 
                num = dot(phi_c[j,:],NormalVector(e))
                denom = 2
            end
        end
        phi_f[i] = num/denom 
    end
    return phi_f 
end