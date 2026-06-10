#This file contains the functions to build a two- and three-dimensional
#triangulation, given as inputs: 

# - num_pts: Int: number of vertices desired for the triangulation 
# - bnd = the localisation of the 4 boundary vertices for the domain 
# - plot: Boolean, whether to make a plot or not 

#Note the points in the mesh will be uniformly distributed on (0,1)

using Meshes
using Delaunay, GeometryBasics
using Random, Distributions


function BuildTriangulation(num_pts,plot=false)
    bnd = [0 0; 0 1; 1 0; 1 1]
    n_bnd = Int(floor(num_pts/20))
    n_inside = num_pts-4*(n_bnd+1)

    tol = 1/num_pts

    if num_pts == 1 
        points = [0.4 0.7]
    elseif num_pts == 2
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

    mesh = Delaunay.delaunay(points)

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

    if plot 
        CairoMakie.activate!()
        set_theme!(theme_latexfonts())
        fig = Figure()
        ax = Axis(fig[1,1],title="2D triangulated mesh with $(length(points)) vertices")
        Makie.wireframe!(ax,m,transparency = true)
        Makie.scatter!(ax,points)
        display(fig)
    end
    return vertex_list, face_list, boundary_list, cell_list
end


function BuildTetrahedralisation()

end