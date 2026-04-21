#File to test the functions in mesh_functions and divgrad files 
#To do: 
#      - Clean up this file 
#      - Fix error in RHS Poisson (problem in dimensions)
#      - Also try on 3D mesh 


using Meshes
using Delaunay, GeometryBasics
using CairoMakie, GLMakie
using Random, Distributions
using NDimensionalSparseArrays
include("mesh_functions.jl")
include("divgrad.jl")

n_inside = 1
n_bnd = Int(floor(n_inside/10))

points = rand(Uniform(0,1),n_inside,2)
points = [0.4 0.7]
bnd = [0 0; 0 1; 1 0; 1 1]


# for j in 1:2
#     if j ==  1
#         k = 1
#     elseif j == 2
#         k = 4
#     end
#     points_bnd = rand(Uniform(0,1),n_bnd,2)
#     [points_bnd[i,1] = bnd[k,1] for i in 1:n_bnd]
#     global bnd = vcat(bnd,points_bnd)

#     points_bnd = rand(Uniform(0,1),n_bnd,2)
#     [points_bnd[i,2] = bnd[k,2] for i in 1:n_bnd]
#     global bnd = vcat(bnd,points_bnd)
# end

points = vcat(points,bnd)

mesh = Delaunay.delaunay(points)

tris = [GeometryBasics.TriangleFace(mesh.simplices[i, :]...) for i in 1:size(mesh.simplices, 1)]
points = Makie.to_vertices(mesh.points)
m = GeometryBasics.Mesh(points, tris) 


lines = GeometryBasics.decompose(LineFace{Int}, tris)
face_list = collect([lines[i][1], lines[i][2]] for i in 1:size(lines,1))

for e in face_list
    ee = CyclicPermutations(e)
    for i in eachindex(e)
        if ee[i] ∉ face_list
            push!(face_list,ee[i])
        end
    end
end



boundary_list = collect(mesh.convex_hull[i,:] for i in 1:size(mesh.convex_hull,1))
for e in boundary_list
    ee = CyclicPermutations(e)
    for i in eachindex(e)
        if ee[i] ∉ boundary_list
            push!(boundary_list,ee[i])
        end
    end
end

cell_list = collect(mesh.simplices[i,:] for i in 1:size(mesh.simplices,1))
vertex_list = collect(mesh.points[i,:] for i in 1:size(mesh.points,1))


# CairoMakie.activate!()
# set_theme!(theme_latexfonts())
# fig = Figure()
# ax = Axis(fig[1,1],title="2D triangulated mesh with $(length(points)) vertices")
# wireframe!(ax,m,transparency = true)
# scatter!(ax,points)

# display(fig)

# save("figures/2dtriangles_vertexlabels.pdf",fig,pt_per_unit=1)

nothing

##
include("divgrad.jl")

n = length(vertex_list)
u_face = NDSparseArray{Float64}(n, n)
for e in face_list
    u_face[e[1],e[2]] = dot(vertex_list[e[1]],vertex_list[e[2]])
    if e in boundary_list 
        u_face[e[1],e[2]] = 0
    end
end

p_cell = NDSparseArray{Float64}(n, n, n)
for K in cell_list
    p_cell[K[1],K[2],K[3]] = vertex_list[K[1]][1]
end

d=2
divu = Divergence(u_face)
gradp = Gradient(p_cell)

include("sparse_operations.jl")

println("(p,div(u))ₖ   = ",SparseInnerProduct(p_cell,divu,"cell"))
println("-(u,grad(p))ₑ = ",-SparseInnerProduct(u_face,gradp,"face"))

##
include("interpolation.jl")
include("poisson.jl")
n = length(vertex_list)
A = StiffnessMatrix(n)
f = LoadVector(n,u_face)

##
include("sparse_operations.jl")
b = ones(n)
SparseMatVec(u_face,b)



