#File to implement the interpolation for either: 
# - face centers to vertices 
# - cell centers to vertices 
# - face centers to cell centers 
# - cell centers to face centers

#Is cell to face interpolation needed? Also, need to take into account the
#boundary faces 

include("mesh_functions.jl")

#Function: VertexInterpolation
#Input: G Tensor to interpolate 
#Output: g Vector evaluated at vertices 
function VertexInterpolation(G)
    n = length(vertex_list)
    g = zeros(n)
    for i in 1:n 
        num = 0
        denom = 0   
        for j in eachindex(G)
            if hasindex(G,j)
                ind = collect(Tuple.(j))
                if i in ind
                    num += G[j] * Volume(ind)
                    denom += Volume(ind)
                end
            end
        end
        g[i] = num/denom 
    end
    return g
end


#Function FaceToCellInterpolation
#Input: G_face Tensor defined on face centers to interpolate
#Output: G_cell Tensor of interpolation at cell centers
function FaceToCellInterpolation(G_face)
    #dimension d of the problem 
    d = length(vertex_list[1])
    #Initialise the cell interpolation tensor 
    G_cell = NDSparseArray{Float64}(ntuple(i->n,d+1))
    for K in cell_list
        e_K = Faces(K)
        num = 0
        denom = 0
        for i in eachindex(K)
            e = e_K[i,:]
            ind_e = CartesianIndex(Tuple(e))
            num += G_face[ind_e] * Volume(e)
            denom += Volume(e)
        end
        ind_K = CartesianIndex(Tuple(K))
        G_cell[ind_K] = num / denom
    end
    return G_cell
end



function CellToFaceInterpolation(G_cell)
    d = length(vertex_list[1])
    G_face = NDSparseArray{Float64}(ntuple(i->n,d))
    for e in face_list
        adj_cells = Adjacent(e)
        num = 0
        denom = 0
        for i in 1:2 
            K = adj_cells[i,:]
            if 0 ∉ K 
                ind_K = CartesianIndex(Tuple(K))
                num += G_cell[ind_K] * Volume(K) 
                denom += Volume(K) 
            end
        end
        ind_e = CartesianIndex(Tuple(e))
        G_face[ind_e] = num/denom 
    end
    return G_face
end