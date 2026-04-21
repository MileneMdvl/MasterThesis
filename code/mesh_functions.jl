#This file contains all the functions needed for e.g. divergence and gradient
#that rely only on the given discrete mesh (either 2D or 3D)

#Variables considered as inputs are: 
#d:                dimension of the problem (2 or 3)
#K of size (d+1):  cell (triangle or tetrahedron)
#e of size d:      edge of a cell (line or triangle)

#vertex_list: Array of positions of all vertices in the discretisation
#face_list:   Array of labels of all faces
#cell_list:   Array of labels of all cells 

using LinearAlgebra

#Function: CyclicPermutations
#Input: A Array
#Output: Vector of Vectors : all cyclic permutations of A 
function CyclicPermutations(A)
    perms = [circshift(A, i) for i in eachindex(A)]
    return perms 
end

#Function UniqueList 
#Input: list Array
#Output: unique_list Array of unique values of list
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

#Function: VerticesCoordinates
#Input: A Array of vertices: cell or face 
#Output: P Array of points: coordinates of vertices of A 
function VerticesCoordinates(A)
    P = zeros(length(A),2)
    for i in eachindex(A)
        P[i,:] = vertex_list[A[i]]
    end
    return P
end


#Function: Volume 
#Input: A Array of vertices: cell or face to take the volume of 
#Output: |A| Float: Volume of A
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

#Function: AreaTriangle (using Heron's formula)
#Input: p₁, p₂, p₃, points coordinates 
#Output: a Float: area of the triangle formed by these points 
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

#Function: VolumeTetrahedron
#Input: p₁, p₂, p₃, p₄, points coordinates 
#Output: v Float: volume of the tetrahedron formed by these points 
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

#Function: Circumcenter 
#Input:  A Vector of size nn (face or cell)
#Output: c Array: coordinates of circumcenter 
#if 2D: c = [cx, cy], if 3D: c = [cx, cy, cz]
function Circumcenter(A)
    nn = length(A)

    #Get coordinates of points 
    p₁ = vertex_list[A[1]]
    p₂ = vertex_list[A[2]]

    #If A has two indices it is a line
    if nn == 2
        return [p₁[1]+p₂[1], p₁[1]+p₂[1]]/2
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

#Function: DualEdge
#Input: e Vector of size (n-1): primal face
#Output: |ê| Scalar: length of dual edge 
function DualEdge(e)
    if e in boundary_list
        println("Error: face e on boundary, no dual edge!")
        return 0
    end 
    #Get the labels for the two adjacent cells  
    adjacent_cells = Adjacent(e)
    #Get the circumcenters of the adjacent triangles 
    cK = Circumcenter(adjacent_cells[1,:])
    cL = Circumcenter(adjacent_cells[2,:])
    return LinearAlgebra.norm(cK - cL) 
end

#Function Faces
#Input: K Vector of size (d+1): cell to find the boundary of 
#Output e_K Matrix of size (d+1)*d: Matrix of Vectors of faces
function Faces(K)
    d = length(K) - 1
    KK = CyclicPermutations(K)
    e_K = zeros(Int,d+1,2)
    for i in 1:(d+1) 
       e_K[i,:] = KK[i][1:d]
    end
    return e_K
end

#Function: Adjacent 
#Input: e Vector of size d: face 
#Output: [K; L] Matrix of size 2*(d+1): adjacent cells
function Adjacent(e)
    d = length(e) 
    adjacent_cells = zeros(Int,2,d+1)

    #If e is on the domain boundary, set the second adjacent cell to [0]ₙ
    if e in boundary_list
        [adjacent_cells[2,i] = 0 for i in 1:(d+1)]
    end 

    #Find the adjacent cells to edge e
    i = 1
    for j in eachindex(cell_list)
        if e ⊆ cell_list[j] 
            adjacent_cells[i,:] = cell_list[j]
            i+=1
        end
    end
    return adjacent_cells
end


#Function: NormalIndicator 
#Input: e Vector of size d: face on cell K 
#       K Vector of size d+1: cell 
#Output: indicator = ±1 Int 
# +1 if normal at e is outward to cell K and -1 if it is inward
function NormalIndicator(e,K)
    d = length(e)
    #Check if e is a face 
    if e ⊆ K
        if d == 2 
            #Get the two points that make the edge
            p₁ = vertex_list[e[1]]
            p₂ = vertex_list[e[2]]
            #Find the third points of the triangle
            p₃ = vertex_list[K[(!in).(K,Ref(e))][]]
            #Get the normal to the line
            nor = nullspace((p₂ - p₁)')
            #The normal is outward if the dot product with the third point is negative
            indicator = - sign(dot(nor,(p₃ - p₂)))

        elseif d == 3
            #Get the three points that make up the triangle face
            p₁ = vertex_list[e[1]]
            p₂ = vertex_list[e[2]]
            p₃ = vertex_list[e[3]]
            #Find the fourth point of the tetrahedron
            p₄ = vertex_list[K[(!in).(K,Ref(e))][]]

            #Compute normal to tringle face e 
            nor = cross((p₂-p₁),(p₃-p₁))
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
