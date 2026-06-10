#File to implement the interpolation for either: 
# - face centers to vertices 
# - cell centers to vertices 
# - face centers to cell centers 
# - cell centers to face centers

#Inputs are lists of vertices (V), cells (C), faces (F) and boundary faces (BF)

include("mesh_functions.jl")

#Function: VertexInterpolation
#Input: phi, vector of either cell-centred of face-centred values 
#Output: phi_v, vector of values at vertices 
function VertexInterpolation(phi)
    nn = length(phi) #either nn = nc or nn = nf 
    phi_v = zeros(nv)
    if nn == length(cell_list)
        type = "cell"
        local list = cell_list
    elseif  nn == length(face_list)
        type = "face"
        local list = face_list
    else 
        println("Error: vector to be interpolated should be defined on cell-centres of face-centres")
        return 
    end
    for i in 1:nv 
        list_with_vertex = WithVertex(i,type)
        num = 0
        denom = 0
        for j in 1:nn
            A = list[j]
            if A in list_with_vertex 
                num += phi[j] * Volume(A) 
                denom += Volume(A)
            end
        end
        phi_v[i] = num/denom 
    end
    return phi_v 
end


#Function FaceToCellInterpolation
#Input: phi_f, face-centred vector 
#Output: phi_c, cell-centred vector 
function FaceToCellInterpolation(phi_f)
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
            for j in 1:nf 
                e = face_list[j] 
                if isFace(e,K)
                    # num += phi_f[j] * Volume(e) * NormalVector(e) * NormalIndicator(e,K)
                    num += phi_f[j] * Volume(e) * NormalVector(e) 
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
function CellToFaceInterpolation(phi_c)
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
        for j in 1:nc 
            K = cell_list[j]
            if isFace(e,K)
                if e ∉ boundary_list
                    num += dot(phi_c[j,:],NormalVector(e)) * Volume(K)
                    denom += Volume(K) 
                    # num += dot(phi_c[j,:],NormalVector(e))
                else 
                    num = dot(phi_c[j,:],NormalVector(e))
                    denom = 2
                end
            end
        end
        phi_f[i] = num/denom 
    end
    return phi_f 
end