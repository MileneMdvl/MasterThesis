"
This file contains the functions to extract information from the triangulated mesh. It assumes that the 'vertex_list', 'face_list', 'boundary_list' and 'cell_list' lists from the file 'build_mesh.jl' are already stored in memory. These lists need to be of type Vector{Vector{Int}}. 

This file must then be run after 'build_mesh.jl'. 

All functions are explained with regards to their inputs, outputs, and what they do concretely. 

All functions hold for both 2 and 3-dimensional domains, i.e. whether cells are triangles or tetrahedra, and whether faces are lines or triangles. 

This file is needed for 'divgrad.jl'. 
"

using LinearAlgebra

"
Function: 
    CyclicPermutations

Input: 
    A: Vector{Int}
    either a face or a cell 

Output: 
    PA: Vector{Vector{Int}}
    all cyclic permutations of A stored as vectors inside a vector

This functions is required in order for boundary_list to be a subset of face_list
"
function CyclicPermutations(A)
    perms = [circshift(A, i) for i in eachindex(A)]
    return perms 
end

"
Function: 
    UniqueList

Input: 
    list: Vector{Vector{Int}}

Output: 
    unique_list: Vector{Vector{Int}}
    the unique elements in the list, i.e. if the list contains both [a,b] and [b,a] then unique_list contains only [a,b]. 

This function is required for boundary_list and face_list to only store each face once 
"
function UniqueList(list)
    unique_list = copy(list)
    nn = length(list[1])
    for i in unique_list
        ii = CyclicPermutations(i)
        for i in 1:(nn-1)
            ind = findall(x->x==(ii[i]),unique_list)
            deleteat!(unique_list,ind)
        end
    end
    return unique_list
end

"
Function: 
    VerticesCoordinates

Input: 
    A: Vector{Int}
    either a cell or a face 

Output: 
    P: Matrix{Float}
    P has dimension length(A) x d, where d is the dimension of the domain 
    P[i,:] is the vector of coordinates of vertex A[i] 

This function collects the coordinates of all vertices, used when determining geometric properties such as volume, area, circumcentre, etc. 
"
function VerticesCoordinates(A)
    d = length(vertex_list[1])
    P = zeros(length(A),d)
    for i in eachindex(A)
        P[i,:] = vertex_list[A[i]]
    end
    return P
end

"
Function: 
    Volume

Input: 
    A: Vector{Int}
    either a face or a cell

Output: 
    |A| Float 
    volume/area/length of A, depending on if A has four/three/two vertices 

This functions computes |K|, |e| for K cell and e face, needed for e.g. divergence/gradient. 
This function calls AreaTriangle and VolumeTetrahedron if needed, and computes the length of an edge from the Euclidean norm.
"
function Volume(A)
    nn = length(A)
    P = VerticesCoordinates(A)
    #If A is a line
    if nn == 2
        return LinearAlgebra.norm(P[1,:] - P[2,:])
    #If A is a triangle
    elseif nn == 3
        return AreaTriangle(P[1,:],P[2,:],P[3,:])
    #If A is a tetrahedron
    elseif nn == 4
        return VolumeTetrahedron(P[1,:],P[2,:],P[3,:],P[4,:])
    end
end

"
Function: 
    AreaTriangle

Input: 
    p₁, p₂, p₃: Float 
    coordinates of the vertices of a triangle

Output: 
    a: Float 
    area of the triangle, using Heron's formula
"
function AreaTriangle(p₁, p₂, p₃)
    #Compute the length of each edge 
    l = zeros(3)
    l[1] = LinearAlgebra.norm(p₁ - p₂)
    l[2] = LinearAlgebra.norm(p₂ - p₃)
    l[3] = LinearAlgebra.norm(p₃ - p₁)
    #Compute the semi-perimeter
    p = 0
    for i in 1:3
        p += l[i]/2
    end
    #Compute the area 
    a = copy(p)
    for i in 1:3 
        a *= p-l[i]
    end
    return sqrt(a)
end

"
Function: 
    VolumeTetrahedron

Input: 
    p₁, p₂, p₃, p₄: Float 
    coordinates of the vertices of a tetrahedron

Output: 
    v: Float 
    volume of the tetrahedron
"
function VolumeTetrahedron(p₁, p₂, p₃, p₄)
    p = [p₁, p₂, p₃, p₄]
    Mat = Matrix{Float64}(undef, 4, 4)
    for i in 1:4 
        for j in 1:3
            Mat[i,j] = p[i][j]
        end
    end
    Mat[:,4] = [1,1,1,1]
    v = abs(1/6 * det(Mat))
    return v
end

"
Function: 
    Circumcentre 

Input: 
    A: Vector{Int}
    either a face or a cell 

Output: 
    c: Vector{Float}
    coordinates of the circumcentre in d-dimensions 
"
function Circumcentre(A)
    nn = length(A)

    #Get coordinates of points 
    p₁ = vertex_list[A[1]]
    p₂ = vertex_list[A[2]]

    #If A has two indices it is a line
    if nn == 2
        return [p₁[1]+p₂[1], p₁[2]+p₂[2]]/2
    #If A has three indices then it is a triangle
    elseif nn == 3 
        p₃ = vertex_list[A[3]]
        p = [p₁, p₂, p₃]
        Matx = Matrix{Float64}(undef, 3, 3)
        Maty = Matrix{Float64}(undef, 3, 3)
        Mata = Matrix{Float64}(undef, 3, 3)
        for i in 1:3
            Matx[i,1] = p[i][1]^2 + p[i][2]^2
            Matx[i,2] =  p[i][2]
            Matx[i,3] = 1

            Maty[i,1] = p[i][1]^2 + p[i][2]^2
            Maty[i,2] = p[i][1]
            Maty[i,3] = 1

            Mata[i,1] = p[i][1]
            Mata[i,2] = p[i][2]
            Mata[i,3] = 1
        end
        bx = -det(Matx)
        by = det(Maty) 
        a = det(Mata)
        return [-bx, -by]/(2*a)

    # If A has four indices it is a tetrahedron
    elseif nn == 4
        p₃ = vertex_list[A[3]]
        p₄ = vertex_list[A[4]]
        p = [p₁, p₂, p₃, p₄]
        Matx = Matrix{Float64}(undef, 4, 4)
        Maty = Matrix{Float64}(undef, 4, 4)
        Matz = Matrix{Float64}(undef, 4, 4)
        Mata = Matrix{Float64}(undef, 4, 4)
        for i in 1:4
            Matx[i,1] = p[i][1]^2 + p[i][2]^2 + p[i][3]^2
            Matx[i,2] = p[i][2]
            Matx[i,3] = p[i][3]
            Matx[i,4] = 1

            Maty[i,1] = p[i][1]^2 + p[i][2]^2 + p[i][3]^2
            Maty[i,2] = p[i][1]
            Maty[i,3] = p[i][3]
            Maty[i,4] = 1

            Matz[i,1] = p[i][1]^2 + p[i][2]^2 + p[i][3]^2
            Matz[i,2] = p[i][1]
            Matz[i,3] = p[i][2]
            Matz[i,4] = 1

            Mata[i,1] = p[i][1]
            Mata[i,2] = p[i][2]
            Mata[i,3] = p[i][3]
            Mata[i,4] = 1
        end
        Dx = det(Matx)
        Dy = -det(Maty)
        Dz = det(Matz)
        a = det(Mata)
        return [Dx, Dy, Dz]/(2*a)
    end
end

"
Function: 
    Circumradius 

Input: 
    K: Vector{Int}
    a cell 

Output: 
    R: Float 
    radius of the circumcentre 

Needed when computing the length of the dual edge for boundary faces. 
This function calls Faces. 
"
function Circumradius(K) 
    e_K = Faces(K)
    R = 1 / (4 * Volume(K))
    for i in eachindex(K)
        e = e_K[i,:]
        R *= Volume(e)
    end
    return R
end

"
Function: 
    DualEdge

Input: 
    e: Vector{Int}
    a face 

Output: 
    |̂e| Float
    lengths of the dual edge to e 

This function calls Adjacent, Circumcentre and Circumradius (if e is a boundary face)
"
function DualEdge(e)
    #Get the labels for the two adjacent cells  
    adj = Adjacent(e)
    K, L = adj[1,:], adj[2,:]
    #If e is a boundary face, then by definition of Adjacent(e), L = 0 
    #Then, dual edge is computed differently (see thesis)
    if e ∈ boundary_list
        R = Circumradius(K)
        dual = 2 * sqrt(R^2 - Volume(e)^2 / 4)
    else
        #Get the circumcentres of the adjacent triangles 
        cK = Circumcentre(K)
        cL = Circumcentre(L)
        dual = LinearAlgebra.norm(cK - cL) 
    end
    return dual 
end

"
Function: 
    FacesInd 

Input: 
    K: Vector{Int}
    a cell

Output: 
    inds_e_K: 
    indices of the faces within face_list that border the cell K 

This functions is required to speed up computations when finding the faces on the boundary of a given cell. It uses the dictionaries vc_info and cf_info. These dictionaries are defined in the main file (here: 2d_mesh_test.jl)
"
function FacesInd(K)
    #Get the number of vertices that make up K 
    nK = length(K) 
    
    #Find the vertices a,b,c such that K = [a,b,c]
    #First, find the vertices a,b, and intersect their cell info to get the
    #index of the face [a,b]
    vc_a = vc_info[K[1]]
    vc_b = vc_info[K[2]]
    inter = intersect(vc_a,vc_b)
    #Then, find the vertex c and intersect to find the indices of [a,c] and [b,c]
    vc_c = vc_info[K[3]]
    inter = intersect(inter,vc_c)

    if nK == 4
        #If K is a tetrahedron then proceed once again 
        vc_d = vc_info[K[4]]
        inter = intersect(inter,vc_d)
    end

    return cf_info[inter[1]]
end

#Function Faces
#Input: K Vector of size (d+1): cell to find the boundary of 
#Output e_K Matrix of size (d+1)*d: Matrix of Vectors of faces
"
Function: 
    Faces

Input: 
    K: Vector{Int}
    a cell

Output: 
    e_K: Matrix{Int}
    e_K if of size length(K) x d where d is the dimension of the domain 
    it contains the vertices of the faces bordering cell K 

This function calls FacesInd. 
"
function Faces(K)
    d = length(K) - 1
    e_K = zeros(Int,d+1,2)

    faces_ind = FacesInd(K)
    for i in eachindex(faces_ind)
        e_K[i,:] = face_list[faces_ind[i]]
    end
    return e_K
end

"
Function: 
    isFace

Input: 
    e: Vector{Int}
    K: Vector{Int}
    a face and a cell

Output: 
    Boolean
    true if e is on the boundary of K 
"
function isFace(e,K)
    e_K = Faces(K)
    istrue = false
    for k in 1:3 
        if e in CyclicPermutations(e_K[k,:])
            istrue = true
        end
    end
    return istrue 
end

# #Function: WithVertex
# #Input: v Int: vertex label
# #       type String: either 'cell' or 'face'
# #Output: A Array of cells/faces with vertex i 
# function WithVertex(v,type::String)
#     local list = []
#     if type == "cell"
#         for K in cell_list
#             if v in K 
#                 push!(list,K)
#             end
#         end
#     elseif type == "face"
#         for e in face_list 
#             if v in e 
#                 push!(list,e)
#             end
#         end
#     else 
#         println("Error: type should either be 'cell' or 'face'")
#     end
#     return list 
# end

"
Function: 
    AdjInds

Input: 
    e: Vector{Int}

Output: 
    inds_K_e: Vector{Int}
    indices of cells K and L within cell_list such that these cells are adjacent to e

If e is a boundary face, then this function only returns the index of K. It uses the dictionary vc_info. 
"
function AdjInds(e)
    d = length(e)
    adj = Array{Int,1}()
    vc_a = vc_info[e[1]]
    vc_b = vc_info[e[2]]
    inter = intersect(vc_a,vc_b)
    if  d == 3
        vc_c = vc_info[e[3]]
        inter = intersect(inter,vc_c)
    end
    return inter 
end

"
Function: 
    Adjacent

Input: 
    e: Vector{Int}
    a face

Output: 
    [K; L] Matrix{Int}
    vertices of the adjacent cells

If e is a boundary face, then L = 0. This function calls AdjInds. 
"
function Adjacent(e)
    d = length(e) 
    adjacent_cells = zeros(Int,2,d+1)
    cells = AdjInds(e)
    for i in eachindex(cells)
        adjacent_cells[i,:] = cell_list[cells[i]]
    end
    return adjacent_cells
end


"
Function: 
    NormalVector

Input: 
    e: Vector{Int}
    a face 

Output: 
    n_e: Vector{Float}
    the outward normal vector to face e in d-dimension 
"
function NormalVector(e)
    d = length(e)
    if d == 2 
        #Get the two points that make the edge
        p₁ = vertex_list[e[1]]
        p₂ = vertex_list[e[2]]
        #Get the normal to the line
        nor = nullspace((p₂ - p₁)')
    elseif d == 3
        #Get the three points that make up the triangle face
        p₁ = vertex_list[e[1]]
        p₂ = vertex_list[e[2]]
        p₃ = vertex_list[e[3]]
        #Compute normal to tringle face e 
        nor = cross((p₂-p₁),(p₃-p₁))
    end
    return vec(nor)
end

#Function: NormalIndicator 
#Input: e Vector of size d: face on cell K 
#       K Vector of size d+1: cell 
#Output: indicator = ±1 Int 
# +1 if normal at e is outward to cell K and -1 if it is inward
"
Function: 
    NormalIndicator

Input: 
    e: Vector{Int}; K: Vector{Int}
    a cell and a face

Output: 
    indicator: ± 1
    positive if n_e is outward to K 
    
This function calls the function NormalVector, and determines if it is inward or outward to the cell. It also returns an error if the given face is not on the boundary of the given cell. 
"
function NormalIndicator(e,K)
    d = length(e)
    #Check if e is a face 
    if e ⊆ K
        nor = NormalVector(e)
        if d == 2 
            #Get one of the two points that make the edge
            p₂ = vertex_list[e[2]]
            #Find the third points of the triangle
            p₃ = vertex_list[K[(!in).(K,Ref(e))][]]

            #The normal is outward if the dot product with the third point is negative
            indicator = - sign(dot(nor,(p₃ - p₂)))

        elseif d == 3
            #Get on of the three points that make up the triangle face
            p₃ = vertex_list[e[3]]
            #Find the fourth point of the tetrahedron
            p₄ = vertex_list[K[(!in).(K,Ref(e))][]]

            #Check if p₄ is behind the triangle or in front of it
            #if it is behind then n is outward normal 
            #so we compute the dot product, the indicator will then be minus the sign 
            indicator = - sign(dot(nor,(p₃-p₄)))
        end
    else 
        println("Error: face e not a boundary of cell K!")
        indicator = 0
    end
    return indicator
end

