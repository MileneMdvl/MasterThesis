#File to test the functions in mesh_functions and divgrad files 
#To do: 
#      - Fix error in RHS Poisson (problem in dimensions)
#      - Also try on 3D mesh 


using Meshes
using Delaunay, GeometryBasics
using CairoMakie, GLMakie
using Random, Distributions
using SparseArrays

import LinearSolve as LS

include("mesh_functions.jl")
include("divgrad.jl")
include("sparse_operations.jl")
include("interpolation.jl")

##

n_inside = 1
n_bnd = Int(floor(n_inside/10))

# points = rand(Uniform(0,1),n_inside,2)
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
# Makie.wireframe!(ax,m,transparency = true)
# Makie.scatter!(ax,points)

# display(fig)

# save("figures/2dtriangles_vertexlabels.pdf",fig,pt_per_unit=1)

##

nf = length(face_list)
nc = length(cell_list)
nv = length(vertex_list)

D = Divergence(cell_list,face_list)
G = Gradient(cell_list,face_list,boundary_list)
#Laplacian 
Laplacian = sparse(SparseMatMat(D,G))


#%%
#Implementing the manufactured solution 
include("manufactured_sol.jl")

#Collect the xy coordinates of cell centers 
xc = zeros(nc)
yc = zeros(nc)
for i = 1:nc 
    K = cell_list[i]
    local p = Circumcenter(K)
    xc[i] = p[1]
    yc[i] = p[2]
end

#Initialise the manufactured solution 
#Also get the solution to Navier-Stokes for comparison 
uc0 = zeros(nc,2)
rho = zeros(nc)
fc = zeros(nc, 2)
for i = 1:nc
    uc0[i,1] = u_eval(xc[i],yc[i])
    uc0[i,2] = v_eval(xc[i],yc[i])
    rho[i] = rho_eval(xc[i],yc[i])
    fc[i,1] = f_x(xc[i],yc[i])
    fc[i,2] = f_y(xc[i],yc[i])
end

uc = copy(uc0)

#Also store the density as a matrix, where rho[i,1] = rho[i,2]
#for i the cell index 
rho2 = reshape(transpose(repeat(rho,2)),nc,2)

#number of timesteps 
Nt = 100
dt = 1/Nt #timestep 

error = zeros(Nt)

Re = 1600

for t in 1:Nt 
    uf = CellToFaceInterpolation(uc)
    #Intermediate velocity
    u_star_c = uc - dt * Convection(uf,uc) + dt/Re * SparseMatMat(Laplacian,uc) + dt * fc

    #Solve the Poisson pressure problem 
    u_star_f = CellToFaceInterpolation(u_star_c)
    RHS = 1/dt * SparseMatVec(D,u_star_f) 
    prob = LS.LinearProblem(Laplacian,RHS)
    sol = LS.solve(prob)
    local pc = sol.u 

    #Update velocity 
    global uc = u_star_c - dt * FaceToCellInterpolation(SparseMatVec(G,pc))

    error[t] = norm(uc-uc0,2)
    println((t,error[t]))
end

CairoMakie.activate!()
set_theme!(theme_latexfonts())
fig = Figure(fontsize = 16)
ax = Axis(fig[1,1],title="Error for manufactured solution",
    xlabel = "Time iterates",
    ylabel = L"||u_c - u_0||_2")
Makie.lines!(ax,error,linewidth=2)
display(fig)