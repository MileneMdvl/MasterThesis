#Third attempt at writing the discrete divergence and gradient 
#Now: the input functions u,p are vectors which are defined on the vertices of
#the discrete mesh 

#To do: 
#       - Include the possibility to take tensors in divergence gradient 
#       - Put equations for average and tensor expansion in a different file 
#       - Fix average function  

#Variables considered as inputs are: 
#d:                dimension of the problem (2 or 3)
#n:                number of vertices of the discretisation 
#K of size (d+1):  cell (triangle or tetrahedron)
#e of size d:      edge of a cell (line or triangle)
#g of size n:      vector of function values at the vertices 

include("mesh_functions.jl")

#Function: Average 
#Input: g Array of size n: function to take the average of 
#       A Array: cell/face to take the average over 
#Output: g_A Float: average of g on A
function Average(g,A)
    g_A = 0
    for i in A
        g_A += g[i] 
    end
    g_A *= Volume(A) 
    return g_A
end


#Need a function to turn vector g into tensor G where e.g. in 2D 
#G[i,j] has entry iff there is an edge (i,j) (Target dim = 2, type='face')

#Function TensorExpand 
#Input: g Array of size n: Array to turn into a tensor 
#       type String: either "cell" or "face"
#Output: G Tensor of desired dimension
function TensorExpand(g,type::String)
    n = length(g)
    d = length(vertex_list[1])
    if type == "face" 
        if d == 2 
            G = NDSparseArray{Float64}(n, n) 
            for e in face_list
                G[e[1],e[2]] = Average(g,e) 
            end
        elseif d == 3 
             G = NDSparseArray{Float64}(n, n, n) 
            for e in face_list
                G[e[1],e[2],e[3]] = Average(g,e) 
            end
        end
    elseif type == "cell"
        if d == 2
            G = NDSparseArray{Float64}(n, n, n) 
            for K in cell_list
                G[K[1],K[2],K[3]] = Average(g,K) 
            end
        elseif d == 3 
             G = NDSparseArray{Float64}(n, n, n, n) 
            for K in cell_list
                G[K[1],K[2],K[3],K[4]] = Average(g,K) 
            end
        end
    else 
        println("Error: incorrect type, should be either 'face' or 'cell'")
    end
    return G
end


#Function: Divergence 
#Input: g Array of size n: function to take the divergence of 
#       K Array of size (d+1): cell to take the divergence over
#Output: div Float: divergence of g on K 
function Divergence(g,K)
    e_K = Faces(K)
    div = 0 
    for i in eachindex(K)
        e = e_K[i,:]
        div += Average(g,e) * Volume(e) * NormalIndicator(e,K)
    end
    div /= Volume(K) 
    return div
end



#Function Gradient
#Input: g Array of size n: function to take the gradient of 
#       e Array of size d: face to take the gradient over
#Output: grad Float: gradient of g on e 
function Gradient(g,e)
    if e in boundary_list
        println("Error: edge on the boundary")
    end
    #Find the two adjacent triangles 
    adj_cells = Adjacent(e)
    K,L = adj_cells[1,:], adj_cells[2,:]
    grad = (Average(g,L) - Average(g,K))/DualEdge(e) * NormalIndicator(e,K)
    return grad 
end