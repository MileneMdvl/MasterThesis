#File to test the functions in mesh_functions and divgrad files 
#To do: 
#      - Clean up this file 
#      - Fix issue with average 
#      - Also try on 3D mesh 
#      - Implement Poisson equation files 
#      - Move vectorise function in a different file


using Meshes, CoordRefSystems, Unitless
using Delaunay, GeometryBasics
using CairoMakie, GLMakie
using Random, Distributions
using NDimensionalSparseArrays, SparseArrays
include("mesh_functions.jl")
include("divgrad_v3.jl")

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
include("divgrad_v3.jl")
println("Using v3")

n = length(vertex_list)

u = zeros(n)
p = zeros(n)
for i in 1:n 
    u[i] = 1
    p[i] = vertex_list[i][1]
end

p_cell = TensorExpand(p,"cell")
u_face = TensorExpand(u,"face")

nothing

##
include("divgrad_v3.jl")
println("Using v3")

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


#Function Vectorise
#Input: G Tensor 
#Output: g Vector: average of G on each vertex
function Vectorise(G)
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

p = Vectorise(p_cell)
u = Vectorise(u_face)

divu = NDSparseArray{Float64}(n, n, n) 
for K in cell_list
    divu[K[1],K[2],K[3]] = Divergence(u,K)
end 

gradp = NDSparseArray{Float64}(n, n) 
for e in face_list 
    if e ∉ boundary_list
        gradp[e[1],e[2]] = Gradient(p,e)
    end
end



println("(p,div(u))ₖ   = ",SparseInnerProduct(p_cell,divu,"cell"))
println("-(u,grad(p))ₑ = ",-SparseInnerProduct(u_face,gradp,"face"))


##
include("divgrad_v2.jl")
println("Using v2")

divu_2 = NDSparseArray{Float64}(n, n, n) 
for K in cell_list
    divu_2[K[1],K[2],K[3]] = Divergence_v2(u_face,K)
end 

gradp_2 = NDSparseArray{Float64}(n, n) 
for e in face_list 
    if e ∉ boundary_list
        gradp_2[e[1],e[2]] = Gradient_v2(p_cell,e)
    end
end

println("(p,div(u))ₖ   = ",SparseInnerProduct(p_cell,divu_2,"cell"))
println("-(u,grad(p))ₑ = ",-SparseInnerProduct(u_face,gradp_2,"face"))

##
println("divergence of u")
for i in eachindex(divu)
    if hasindex(divu,i)
        println("v3: ",divu[i])
        println("v2: ",divu_2[i])
        println(" ")
    end
end

println("----------------------------")

println("gradient of p")

for i in eachindex(gradp)
    if hasindex(gradp,i)
        println("v3: ",gradp[i])
        println("v2: ",gradp_2[i])
        println(" ")
    end
end

## 
include("divgrad_v3.jl")
u_face = NDSparseArray{Float64}(n, n) 
for e in face_list
    u_face[e[1],e[2]] = 0
    if e in boundary_list 
        u_face[e[1],e[2]] = 0
    end
end
u_face[1,2] = 1
u_face[2,1] = 1

u = Vectorise(u_face)
println("Vector u: ",u)
u_face_2 = TensorExpand(u,"face")


println("Comparing the tensors u")
for i in eachindex(u_face)
    if hasindex(u_face,i)
        println("T1: ",u_face[i])
        println("T2: ",u_face_2[i])
        println(" ")
    end
end

# unique_face = UniqueList(face_list)
# for i in eachindex(unique_face)
#     println(unique_face[i])
#     println(Volume(unique_face[i]))
# end

