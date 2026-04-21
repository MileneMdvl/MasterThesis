#File for the discrete divergence and gradient functions. The input variables
#are tensors such that the velocity and pressure are defined at the face and
#cell centers respectively. Note, when taking the gradient of the velocity we
#need to interpolate the velocity at the cell centers. 

#To do: 
#       - Include the possibility to take tensors in divergence gradient 
#       - Put equations for average and tensor expansion in a different file 
#       - Fix average function  
#       - Include more boundary condition types 

#Variables considered as inputs are: 
#d:                dimension of the problem (2 or 3)
#n:                number of vertices of the discretisation 
#K of size (d+1):  cell (triangle or tetrahedron)
#e of size d:      edge of a cell (line or triangle)
#g of size n:      vector of function values at the vertices 

include("mesh_functions.jl")


#Function: Evaluate
#Input: Tensor G, Array A (either face or cell)
#Output: G[A], the value of G on A 
function Evaluate(G,A)
    ind_A = CartesianIndex(Tuple(A))
    return G[ind_A]
end


#Function: DivergenceElement 
#Input: G Tensor evaluated on face centers: function to take the divergence of 
#       K Array of size (d+1): cell to take the divergence over
#Output: div g_K Float: divergence of g on K 
function DivergenceElement(G,K)
    e_K = Faces(K)
    div = 0 
    for i in eachindex(K)
        e = e_K[i,:]
        div += Evaluate(G,e) * Volume(e) * NormalIndicator(e,K)
    end
    div /= Volume(K) 
    return div
end

#Function: Divergence
#Input: G_face Tensor evaluated on face centers 
#Output: div G Tensor evaluated on cell centers
function Divergence(G_face)
    d = length(vertex_list[1])
    div = NDSparseArray{Float64}(ntuple(i->n,d+1))
    for K in cell_list
        ind_K = CartesianIndex(Tuple(K))
        div[ind_K] = DivergenceElement(G_face,K)
    end
    return div
end


#Function GradientElement
#Input: G_cell Tensor evaluated on cell centers
#       e Array of size d: face to take the gradient over
#Output: grad Float: gradient of G on e 
function GradientElement(G_cell,e)
    if e in boundary_list
        println("Error: edge on the boundary")
    end
    #Find the two adjacent triangles 
    adj_cells = Adjacent(e)
    K,L = adj_cells[1,:], adj_cells[2,:]
    grad = (Evaluate(G_cell,L) - Evaluate(G_cell,K))/DualEdge(e) * NormalIndicator(e,K)
    return grad 
end

#Function: Gradient
#Input: G_cell Tensor evaluated on cell centers 
#Output: grad G Tensor evaluated on face centers
function Gradient(G_cell)
    grad = NDSparseArray{Float64}(ntuple(i->n,d))
    for e in face_list 
        if e ∉ boundary_list
            ind_e = CartesianIndex(Tuple(e))
            grad[ind_e] = GradientElement(G_cell,e)
        end
    end
    return grad 
end