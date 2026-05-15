#File to test the functions in mesh_functions and divgrad files 
#To do: 
#      - Fix error in RHS Poisson (problem in dimensions)
#      - Also try on 3D mesh 


using Meshes
using Delaunay, GeometryBasics
using CairoMakie, GLMakie
using Random, Distributions
using NDimensionalSparseArrays

include("mesh_functions.jl")
include("divgrad.jl")
include("sparse_operations.jl")
include("interpolation.jl")

##

n_inside = 1
n_bnd = Int(floor(n_inside/10))

points = rand(Uniform(0,1),n_inside,2)
points = [0.4 0.7]
bnd = [0 0; 0 1; 1 0; 1 1]

#Place points on boundary if needed 
for j in 1:2
    if j ==  1
        k = 1
    elseif j == 2
        k = 4
    end
    points_bnd = rand(Uniform(0,1),n_bnd,2)
    [points_bnd[i,1] = bnd[k,1] for i in 1:n_bnd]
    global bnd = vcat(bnd,points_bnd)

    points_bnd = rand(Uniform(0,1),n_bnd,2)
    [points_bnd[i,2] = bnd[k,2] for i in 1:n_bnd]
    global bnd = vcat(bnd,points_bnd)
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

nf = length(face_list)
nc = length(cell_list)
nv = length(vertex_list)

uf = zeros(nf)
for i in 1:nf
    #Set the boundary condition that u vanishes at the boundary 
    if face_list[i] ∈ boundary_list
        uf[i] = 0
    else
        uf[i] = rand(Float16)
    end
end

pc = zeros(nc) 
for i in 1:nc
    pc[i] = rand(Float16)
end

D = Divergence(cell_list,face_list)
G = Gradient(cell_list,face_list,boundary_list)
#Laplacian 
L = SparseMatMat(D,G)

nothing

#%% 

Duf = SparseMatVec(D,uf)
Gpc = SparseMatVec(G,pc)

println("(p,div(u))ₖ   = ",InnerProdCell(pc,Duf))
println("-(u,grad(p))ₑ = ",-InnerProdFace(uf,Gpc))

#%%
nu = 1 #viscosity
dt = 0.1 #timestep 

L = SparseMatMat(D,G)

uc = FaceToCellInterpolation(uf)
uc_x = uc[:,1]
uc_y = uc[:,2]

# u_intermediate_x = uc_x + dt * Convection(uf,uc_x) + nu * dt * SparseMatVec(L,uc_x)
nothing 

