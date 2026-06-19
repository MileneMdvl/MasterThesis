"
This file contains the functions to generate points and build a triangulation. There are two options for points generation: random or regular. So far, the point generation only holds for a 2-dimensional domain. 

The triangulation uses the package Delaunay: https://github.com/eschnett/Delaunay.jl 

The input for the generation of points is a parameter, N which determines how many points are to be generated. 

The input for the triangulation is the aforementioned points. It's output is then used in the file mesh_functions.jl. 
"

using Meshes
using Delaunay, GeometryBasics
using Random, Distributions

"
Function: 
    BuildTriangulation

Input: 
    pts: Array{Float,2}
    the set of point to triangulate

Output: 
    points: Vector{Point{2, Float}}
        the points of the triangulation (as GeometryBasics object)
    m: GeometryBasics.Mesh 
        the mesh of the triangulation 
    vertex_list: Vector{Vector{Float}}
        the coordinates in d-dimensions of the vertices 
    face_list: Vector{Vector{Int}}
        the faces given in terms of the vertices indices 
        e.g. in 2D a face is given by [a,b] and in 3D by [a,b,c]
        if [a,b] is stored, then [b,a] is not (and similarly for 3D)
    boundary_list: Vector{Vector{Int}}
        the faces on the boundary of the domain
        this is a subset of face_list
    cell_list: Vector{Vector{Int}}
        the cells given in terms of the vertices indices 
    
The outputs points and mesh are needed to plot the resulting triangulation. 
The outputs vertex_list, face_list, boundary_list and cell_list are needed for all other .jl files. 

This file should then be run first. However, note that when collecting the aforementioned lists, we also make use of the functions CyclicPermutations and UniqueList from 'mesh_functions.jl'.
"
function BuildTriangulation(pts)
    mesh = Delaunay.delaunay(pts)

    tris = [GeometryBasics.TriangleFace(mesh.simplices[i, :]...) for i in 1:size(mesh.simplices, 1)]
    points = Makie.to_vertices(mesh.points)
    m = GeometryBasics.Mesh(points, tris) 


    lines = GeometryBasics.decompose(LineFace{Int}, tris)

    #Store the faces 
    #First store all cyclic permuations of faces 
    face_list = collect([lines[i][1], lines[i][2]] for i in 1:size(lines,1))
    face_list = UniqueList(face_list)

    #Store the boundary faces 
    #First store all cyclic permutations of boundary faces 
    boundary_list = collect(mesh.convex_hull[i,:] for i in 1:size(mesh.convex_hull,1))
    for e in boundary_list
        ee = CyclicPermutations(e)
        for i in eachindex(e)
            if ee[i] ∉ boundary_list
                push!(boundary_list,ee[i])
            end
        end
    end
    #Then, only keep those which are in the same order as in the list of faces 
    ind = []
    for i in eachindex(boundary_list)
        local e = boundary_list[i]
        if e ∉ face_list
            push!(ind,i)
        end
    end
    deleteat!(boundary_list,ind)
    boundary_list = UniqueList(boundary_list)

    cell_list = collect(mesh.simplices[i,:] for i in 1:size(mesh.simplices,1))
    vertex_list = collect(mesh.points[i,:] for i in 1:size(mesh.points,1))

    return points, m, vertex_list, face_list, boundary_list, cell_list
end

"
Function: 
    RegularPoints

Input: 
    N: Int 
    the number of points on each side of the rectangular domain

Output: 
    pts: Array{Float,2}
    the set of point given in a regular mesh to triangulate

The domain is is predefined to be [0,1]x[0,√3/2], where √3/2 is the height of an equilateral triangle of side length 1. N determines the number of points on each side of the domain, which in turn determines how many points there are in total. 
"
function RegularPoints(N)
    dx = 1/(N-1)

    #Determine what the x-coordinate of the point is, depending on if we are on
    #an even row or not
    x_odd = range(0,1,length=N)
    x_even = zeros(N+1)
    for i=2:N
        x_even[i] = dx/2 + (i-2) * dx
    end
    x_even[N+1] = 1

    #Height of equilateral triangle with length 1
    h = sqrt(3)/2 
    y = range(0,h,length=N)
    Ny = length(y)

    if Ny % 2 ==0 
        Npts = Ny/2 * N + Ny/2 * (N+1)
    else 
        Npts = floor(Ny/2) * (N+1) + ceil(Ny/2) * N
    end
    Npts = trunc(Int,Npts)

    pts = zeros(Npts,2) 
    ind = 1
    for j = 1:Ny 
        #If odd row 
        if j % 2 == 1
            for i=ind:(ind+N-1)
                k = i - ind + 1
                pts[i,1] = x_odd[k]
                pts[i,2] = y[j]
            end
            ind += N
        else 
            for i=ind:(ind+N)
                k = i - ind + 1
                pts[i,1] = x_even[k]
                pts[i,2] = y[j]
            end
            ind += N+1 
        end
    end
    return pts
end

"
Function: 
    RandomPoints(N)

Input: 
    N: Int 
    the number of total desired points 

Output: 
    points: Array{Float,2}
    the set of randomly generated points to triangulate 

The points are here uniformly distributed on [0,1]. We set a tolerance such that no two points can be closer than a distance of 1/N from one another. This aims to avoid invalid Delaunay triangulations. 
"
function RandomPoints(N)
    #Define the boundary 
    bnd = [0 0; 0 1; 1 0; 1 1]
    n_bnd = Int(floor(N/20))
    n_inside = N-4*(n_bnd+1)

    #Set a tolerance 
    tol = 1/N

    if N == 1 
        points = [0.4 0.7]
    elseif N == 2
        points = [0.4 0.7; 0.6 0.3]
    else
        points = rand(Uniform(0+tol,1-tol),n_inside,2)
        for i = 1:n_inside
            for j = 1:(i-1) 
                while norm(points[i,:] - points[j,:]) < tol
                    points[i,:] = rand(Uniform(0+tol,1-tol),2)
                end

            end
        end
    end

    #Place points on boundary if needed 
    for j in 1:2
        if j ==  1
            k = 1
        elseif j == 2
            k = 4
        end
        points_bnd = rand(Uniform(0,1),n_bnd,2)
        [points_bnd[i,1] = bnd[k,1] for i in 1:n_bnd]
        bnd = vcat(bnd,points_bnd)

        points_bnd = rand(Uniform(0,1),n_bnd,2)
        [points_bnd[i,2] = bnd[k,2] for i in 1:n_bnd]
        bnd = vcat(bnd,points_bnd)
    end

    points = vcat(points,bnd)
    return points 
end



# function BuildTetrahedralisation()

# end