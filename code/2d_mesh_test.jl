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

#To save the data 
using JLD


include("mesh_functions.jl")
include("divgrad.jl")
include("sparse_operations.jl")
include("interpolation.jl")
include("manufactured_sol.jl")
include("build_mesh.jl")
##
N = 6
dx = 1/N
# pts = GenerateRandomPoints(N)
global pts = RegularPoints(N)

global points, mesh, vertex_list, face_list, boundary_list, cell_list = BuildTriangulation(pts)

CairoMakie.activate!()
set_theme!(theme_latexfonts())

fig = Figure(fontsize = 12)

ax1 = Axis(fig[1,1],title=latexstring("\\text{2D triangulated mesh with $(length(points)) vertices}"))
Makie.wireframe!(ax1,mesh,transparency = true)
Makie.scatter!(ax1,points)

display(fig)

#%%
global nf = length(face_list)
global nc = length(cell_list)
global nv = length(vertex_list)


#%%
#Indices for each cell containing the i-th vertex, where the i-th entry corresponds to
#the i-th vertex 
vc_info = Dict{Int,Array{Int,1}}()
for i = 1:nc 
    local K = cell_list[i]
    for j in K 
        if j ∉ keys(vc_info) 
            vc_info[j] = [i]
        else
            push!(vc_info[j],i)
        end
    end
end
vc_info

#gives the faces indices around each cell index 
cf_info = Dict{Int,Array{Int,1}}()
for i = 1:nf 
    local e = face_list[i]
    cell_inds = AdjAux(e)
    for j in cell_inds
        if j ∉ keys(cf_info) 
            cf_info[j] = [i]
        else
            push!(cf_info[j],i)
        end
    end
end
cf_info



#%%
global D = Divergence(cell_list,face_list)
global G = Gradient(cell_list,face_list,boundary_list)
# StaggeredVol = spzeros(nf,nf)
# for i = 1:nf 
#     e = face_list[i]
#     StaggeredVol[i,i] = 1/Volume(e)   
# end
# G = - StaggeredVol * transpose(D)

# G = -transpose(D)

#Laplacian 
global L = D*G

#%%

#Implement the manufactured solution
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

uf0 = zeros(nf)
for i = 1:nf
    local ue = zeros(2)
    ue[1] = u_eval(xf[i],yf[i])
    ue[2] = v_eval(xf[i],yf[i])
    local ne = NormalVector(face_list[i])
    uf0[i] = dot(ue,ne)
end
#%%
#Time marching 
Nt = N 
dt = 1/Nt 

# Nt = 100
# dt = 1/Nt 

println("Nt = ",Nt)

norm_u = zeros(Nt)
norm_p = zeros(Nt)

norm_div = zeros(Nt)
where_max = zeros(Nt)

uc = copy(uc0)
uf = copy(uf0)
for t in tqdm(1:Nt)
    #Intermediate Velocity
    u_star_c = uc - dt * Convection(uf,uc) + dt/Re * L*uc - dt*fc 
    # u_star_c = uc - dt * Convection(uf,uc) + dt/Re * L*uc - dt*fc 
    # u_star_c = zeros(nc,2)

    #Solve the Poisson pressure problem 
    u_star_f = CellToFaceInterpolation(u_star_c)
    RHS = 1/dt * D*u_star_f

    prob = LS.LinearProblem(L,RHS)
    sol = LS.solve(prob)
    global pc = sol.u 
    
    norm_p[t] = norm(pc-pc0)/sqrt(nc)

    #Update velocity 
    # global uc = u_star_c - dt * FaceToCellInterpolation(G*pc)
    global uf = u_star_f - dt* G*pc

    global uc = FaceToCellInterpolation(uf)

    norm_div[t] = norm(D*uf)
    norm_u[t] = norm(uf-uf0)/sqrt(nc)

    if isnan(norm_div[t])
        for i = t:Nt 
            norm_div[i] = NaN
            norm_p[i] = NaN
            norm_u[i] = NaN 
        end
        break
    end
end
save("data/data_$(N).jld","pc",pc,"uc",uc,"uf",uf,"pc0",pc0,"uf0",uf0,"uc0",uc0,"norm_u",norm_u,"norm_p",norm_p)


println("At last time step:")
println("||u-u0||   = ",norm_u[end])
println("||∇⋅u||    = ",norm_div[end])
println("||p-p0||   = ",norm_p[end])

CairoMakie.activate!()
set_theme!(theme_latexfonts())

xlims = nothing
# xlims = (10,50)

fig = Figure(fontsize = 12,size = (900, 700))

Label(fig[0,1:2],fontsize=14,latexstring("Re = $(Re),\\ N = $(N),\\ dt = $(round(dt,digits=5)),\\ dx = $(round(dx,digits=5))"))

ax1 = Axis(fig[1,1],title=latexstring("\\text{2D triangulated mesh with $(length(points)) vertices}"))
Makie.wireframe!(ax1,mesh,transparency = true)
Makie.scatter!(ax1,points)

ax2 = Axis(fig[1,2],
    limits = (xlims, nothing),
    title=latexstring("\\text{Velocity divergence norm}"), 
    # yscale=log10,
    xlabel = "Time iterates",
    ylabel = L"||Du_f||_2",
    )
Makie.lines!(ax2,norm_div,linewidth=2)

ax3 = Axis(fig[2,1],
    limits = (xlims, nothing),
    title=latexstring("\\text{Velocity norm}"),
    # yscale=log10,
    xlabel = "Time iterates",
    ylabel = L"||u_f-u_0||_2")
Makie.lines!(ax3,norm_u,linewidth=2)

ax4 = Axis(fig[2,2],
    limits = (xlims, nothing),
    title=latexstring("\\text{Pressure norm}"),
    yscale=log10,
    xlabel = "Time iterates",
    ylabel = L"||p_c-p_0||_2")
Makie.lines!(ax4,norm_p,linewidth=2)

display(fig)
save("figures/N=$(N).pdf", fig, px_per_unit = 1)

#%%
#For the regularised convection 
# d1 = 0.375
# d2 = 0.0375 
# L_tilde = spzeros(nc,nc) 
# for i = 1:nc 
#     V = Volume(cell_list[i])
#     for j = 1:nc 
#         L_tilde[i,j] = - L[i,j] * V
#     end
# end
# Filter = 1I + d1 * L_tilde + d2 * L_tilde^2
# conv_order = 0
# println("Order of convection regularisation: ",conv_order)
