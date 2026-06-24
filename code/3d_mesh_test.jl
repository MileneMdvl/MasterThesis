#File to build 3d triangulation and test with divergence and gradient


using Meshes
using Delaunay, GeometryBasics
using CairoMakie, GLMakie
using Random, Distributions

include("divgrad.jl")
include("build_mesh.jl")
include("mesh_functions.jl")
include("sparse_operations.jl")

#Testing if the duality condition is satisfied using random points 
#First, 3D
Ns = 15:20:315
err3D = zeros(length(Ns))
for iter in eachindex(Ns)
    N = Ns[iter]
    err = zeros(10)
    for count = 1:10
        points = Random3DPoints(20)

        points, mesh, vertex_list, face_list, boundary_list, cell_list = BuildTetrahedralisation(points)


        global nf = length(face_list)
        global nc = length(cell_list)
        global nv = length(vertex_list)

        vc_info = Dict{Int,Array{Int,1}}()
        for i in eachindex(cell_list)
            local K = cell_list[i]
            for j in K 
                if j ∉ keys(vc_info) 
                    vc_info[j] = [i]
                else
                    push!(vc_info[j],i)
                end
            end
        end

        cf_info = Dict{Int,Array{Int,1}}()
        for i in eachindex(face_list)
            local e = face_list[i]
            cell_inds = AdjInds(e)
            for j in cell_inds
                if j ∉ keys(cf_info) 
                    cf_info[j] = [i]
                else
                    push!(cf_info[j],i)
                end
            end
        end 


        global D = Divergence(cell_list,face_list)
        global G = Gradient(cell_list,face_list)

        #Laplacian 
        global L = D*G

        u = randn(nf)
        p = randn(nc)

        err[count] = InnerProdCell(D*u,p)+InnerProdFace(u,G*p)
    end

    err3D[iter] = mean(err)
end


#Then 2D case 
err2D = zeros(length(Ns))
for iter in eachindex(Ns)
    N = Ns[iter]
    err = zeros(10)
    for count = 1:10
        points = RandomPoints(20)

        points, mesh, vertex_list, face_list, boundary_list, cell_list = BuildTriangulation(points)

        global nf = length(face_list)
        global nc = length(cell_list)
        global nv = length(vertex_list)

        vc_info = Dict{Int,Array{Int,1}}()
        for i in eachindex(cell_list)
            local K = cell_list[i]
            for j in K 
                if j ∉ keys(vc_info) 
                    vc_info[j] = [i]
                else
                    push!(vc_info[j],i)
                end
            end
        end

        cf_info = Dict{Int,Array{Int,1}}()
        for i in eachindex(face_list)
            local e = face_list[i]
            cell_inds = AdjInds(e)
            for j in cell_inds
                if j ∉ keys(cf_info) 
                    cf_info[j] = [i]
                else
                    push!(cf_info[j],i)
                end
            end
        end 

        #%%

        global D = Divergence(cell_list,face_list)
        global G = Gradient(cell_list,face_list)

        #Laplacian 
        global L = D*G

        #%%
        u = randn(nf)
        p = randn(nc)

        err[count] = InnerProdCell(D*u,p)+InnerProdFace(u,G*p)
    end

    err2D[iter] = mean(err)
end


CairoMakie.activate!()
set_theme!(theme_latexfonts())


fig = Figure(fontsize = 11,size = (700,300))

Label(fig[0,1:2],fontsize=14,latexstring("\\text{Sum of the inner products } ⟨u,∇p⟩+⟨∇⋅u,p⟩"))

colgap!(fig.layout, 1, Relative(0.15))

ax1 = Axis(fig[1,1],
    title=latexstring("\\text{2D}"),
    xlabel = L"N",
    ylabel =L"⟨u,∇p⟩+⟨∇⋅u,p⟩",
    )
Makie.lines!(ax1,Ns,err2D,linewidth=2)
# Makie.scatter!(ax1,Ns,err2D)

ax2 = Axis(fig[1,2],
    title=latexstring("\\text{3D}"),
    xlabel = L"N",
    ylabel =L"⟨u,∇p⟩+⟨∇⋅u,p⟩",
    )
Makie.lines!(ax2,Ns,err3D,linewidth=2)
# Makie.scatter!(ax2,Ns,times_N)

display(fig)
save("figures/duality condition.pdf", fig, px_per_unit = 1)