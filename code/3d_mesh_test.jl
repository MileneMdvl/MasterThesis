#File to build 3d triangulation and test with divergence and gradient
#To do: 
#      - Fix divgrad to take in tensors as input (Otherwise this file gives
#      errors)

using Meshes, CoordRefSystems, Unitless
using Delaunay, GeometryBasics
using CairoMakie, GLMakie
using Random, Distributions

using NDimensionalSparseArrays
include("divgrad.jl")

n_inside = 20 
n_bnd = Int(floor(n_inside/10))

points = rand(Uniform(0,1),n_inside,3)

bnd = [0 0 0; 0 1 0; 1 0 0; 1 1 0; 0 0 1; 0 1 1; 1 0 1; 1 1 1]


for j in 1:4
    if j ≤ 2 
        k = 1
    elseif j > 2
        k = 8
    end
    points_bnd = rand(Uniform(0,1),n_bnd,3)
    [points_bnd[i,1] = bnd[k,1] for i in 1:n_bnd]
    bnd = vcat(bnd,points_bnd)

    points_bnd = rand(Uniform(0,1),n_bnd,3)
    [points_bnd[i,2] = bnd[k,2] for i in 1:n_bnd]
    bnd = vcat(bnd,points_bnd)

    points_bnd = rand(Uniform(0,1),n_bnd,3)
    [points_bnd[i,3] = bnd[k,3] for i in 1:n_bnd]
    bnd = vcat(bnd,points_bnd)
end

points = vcat(points,bnd)

mesh = Delaunay.delaunay(points)

tetras = [GeometryBasics.TetrahedronFace(mesh.simplices[i, :]...) for i in 1:size(mesh.simplices, 1)]
points = Makie.to_vertices(mesh.points)
m = GeometryBasics.Mesh(points, tetras)  

CairoMakie.activate!()
fig = Figure()
ax = Axis3(fig[1,1],title="3D triangulated mesh")
wireframe!(ax,m,transparency = true)
scatter!(points)
display(fig)

tris = GeometryBasics.decompose(TriangleFace{Int}, tetras)
face_list = collect([tris[i][1], tris[i][2], tris[i][3]] for i in 1:size(tris,1))

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

np = length(vertex_list)

u = NDSparseArray{Float64}(np, np, np) 
for e in face_list
    u[e[1],e[2],e[3]] = e[1]*e[2]*e[3]
    if e in boundary_list 
        u[e[1],e[2],e[3]] = 0
    end
end

p = NDSparseArray{Float64}(np, np, np, np) 
for K in cell_list
    p[K[1],K[2],K[3],K[4]] = K[1]*K[2]
end

divu = NDSparseArray{Float64}(np, np, np, np) 
for K in cell_list
    divu[K[1],K[2],K[3],K[4]] = Divergence(u,K)
end

gradp = NDSparseArray{Float64}(np, np, np) 
for e in face_list 
    if e ∉ boundary_list
        gradp[e[1],e[2],e[3]] = Gradient(p,e)
    end
end

println("(p,div(u))ₖ   = ",SparseInnerProduct(p,divu,"cell"))
println("-(u,grad(p))ₑ = ",-SparseInnerProduct(u,gradp,"face"))