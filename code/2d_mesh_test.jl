#File to test the functions in mesh_functions and divgrad files 
#To do: 
#      - Implement manufactured solution
#      - Also try on 3D mesh 


# using Meshes
# using Delaunay, GeometryBasics
using CairoMakie, GLMakie, LaTeXStrings
# using Random, Distributions
using SparseArrays
using ProgressBars
import LinearSolve as LS

include("mesh_functions.jl")
include("divgrad.jl")
include("sparse_operations.jl")
include("interpolation.jl")
include("manufactured_sol.jl")
include("build_mesh.jl")
##
num_pts = 10

vertex_list, face_list, boundary_list, cell_list = BuildTriangulation(num_pts,true)


nf = length(face_list)
nc = length(cell_list)
nv = length(vertex_list)

D = Divergence(cell_list,face_list)
G = Gradient(cell_list,face_list,boundary_list)

# StaggeredVol = spzeros(nf,nf)
# for i = 1:nf 
#     e = face_list[i]
#     StaggeredVol[i,i] = 1/Volume(e)
# end
# G = - StaggeredVol * transpose(D)

# G = -transpose(D)

Laplacian = D*G

#Implementing the manufactured solution 
#Collect the xy coordinates of cell centers and face centers
xc = zeros(nc)
yc = zeros(nc)
xf = zeros(nf)
yf = zeros(nf)

for i = 1:nc 
    local K = cell_list[i]
    local p = Circumcenter(K)
    xc[i] = p[1]
    yc[i] = p[2]
end

for i=1:nf 
    local e = face_list[i]
    local p = Circumcenter(e)
    xf[i] = p[1]
    yf[i] = p[2]
end

#Initialise the manufactured solution 
#Also get the solution to Navier-Stokes for comparison 
uc0 = zeros(nc,2)
pc0 = zeros(nc)
# rho = zeros(nc)
fc = zeros(nc, 2)
for i = 1:nc
    uc0[i,1] = u_eval(xc[i],yc[i])
    uc0[i,2] = v_eval(xc[i],yc[i])
    # rho[i] = rho_eval(xc[i],yc[i])
    fc[i,1] = f_x(xc[i],yc[i])
    fc[i,2] = f_y(xc[i],yc[i])
    pc0[i] = p_eval(xc[i],yc[i])
end

#%%
dt = 1e-2
Nt = trunc(Int,1/dt)

# Nt = 100
# dt = 1/Nt 

norm_u = zeros(Nt)
norm_p = zeros(Nt)

norm_div = zeros(Nt)
where_max = zeros(Nt)

uc = copy(uc0)

include("divgrad.jl")
include("interpolation.jl")

uf = CellToFaceInterpolation(uc)


for t in tqdm(1:Nt)
    #Intermediate Velocity
    u_star_c = uc - dt * Convection(uf,uc) + dt/Re * Laplacian*uc + dt*fc 
    # u_star_c = zeros(nc,2)

    #Solve the Poisson pressure problem 
    u_star_f = CellToFaceInterpolation(u_star_c)
    RHS = 1/dt * D*u_star_f

    prob = LS.LinearProblem(Laplacian,RHS)
    sol = LS.solve(prob)
    local pc = sol.u 

    norm_p[t] = norm(pc-pc0)

    #Update velocity 
    # global uc = u_star_c - dt * FaceToCellInterpolation(G*pc)
    global uf = u_star_f - dt* G*pc

    global uc = FaceToCellInterpolation(uf)

    norm_div[t] = norm(D*uf)
    norm_u[t] = norm(uc-uc0)
end


println("Nt = ",Nt)
println("At last time step:")
println("||u-u0||   = ",norm_u[end])
println("||∇⋅u|| = ",norm_div[end])
println("||p-p0||   = ",norm_p[end])

CairoMakie.activate!()
set_theme!(theme_latexfonts())

fig = Figure(fontsize = 16)
ax = Axis(fig[1,1],title=latexstring("\\text{Velocity norm, } Re=$(Re),\\ dt=$(dt),\\ $(nv)\\ \\text{vertices} "),
    yscale=log10,
    xlabel = "Time iterates",
    ylabel = L"||u_c-u_0||_2")
Makie.lines!(ax,norm_u,linewidth=2)
display(fig)

fig = Figure(fontsize = 16)
ax = Axis(fig[1,1],title=latexstring("\\text{Pressure norm, } Re=$(Re),\\ dt=$(dt),\\ $(nv)\\ \\text{vertices} "),
    yscale=log10,
    xlabel = "Time iterates",
    ylabel = L"||p_c-p_0||_2")
Makie.lines!(ax,norm_p,linewidth=2)
display(fig)

# fig = Figure(fontsize = 16)
# ax = Axis(fig[1,1],title=latexstring("\\text{Velocity divergence norm, } Re=$(Re),\\ dt=$(dt),\\ $(nv)\\ \\text{vertices}"), 
#     yscale=log10,
#     xlabel = "Time iterates",
#     ylabel = L"||Du_f||_2")
# Makie.lines!(ax,norm_div,linewidth=2)
# display(fig)

# fig = Figure(fontsize = 16)
# ax = Axis(fig[1,1],title=latexstring("\\text{Location of maximum, } Re=$(Re),\\ dt=$(dt)"),
#     xlabel = "Time iterates",
#     ylabel = L"||Du_f||_2")
# Makie.scatter!(ax,where_max)
# display(fig)

#%%

# include("divgrad.jl")

# g = zeros(nc)
# for i = 1:nc 
#     g[i] = xc[i]^2
# end
# dg = zeros(nf)
# for i = 1:nf 
#     dg[i] = xf[i] * 2
# end

# # println(g)
# # println(dg)
# G = Gradient(cell_list,face_list,boundary_list)
# # println(G*g)
# println(norm(G*g-dg))
# StaggeredVol = spzeros(nf,nf)
# for i = 1:nf 
#     e = face_list[i]
#     StaggeredVol[i,i] = DualEdge(e)
# end
# G = - StaggeredVol * transpose(D)
# # println(G*g)
# println(norm(G*g-dg))
