using DelaunayTriangulation
using CairoMakie
using LinearAlgebra
using SparseArrays
using SparseArrayKit
using NDimensionalSparseArrays
using StableRNGs
using Random
using Plots

include("functions_div_grad.jl")

#Note the first three cells are to create three different meshes  to test on

#%% Make triangulation of a rectangle

function TriangulateRectangle(nx,ny,a=0.0,b=1.0,c=0.0,d=1.0,)
    tri = triangulate_rectangle(a, b, c, d, nx, ny ,delete_ghosts = true,single_boundary = true)
    return tri 
end

nx, ny = 10, 10;
tri = TriangulateRectangle(nx,ny) 
fig, ax, sc = triplot(tri)

#%% Testing also on a random unconstrained triangulation (3d)
np = 100
rand_points = rand(3,np)

points = Array{Tuple{Float64,Float64,Float64}}(undef,np)
for i in 1:np
    points[i] = (rand_points[1,i],rand_points[2,i],rand_points[3,i])
end
tri = triangulate(points,delete_ghosts = true)

tri.triangles

fig = Figure()
ax = Axis3(fig[1, 1]; aspect = :equal)
mesh!(ax, stack(tri.points)', stack(tri.triangles)')

display(fig)



#%% Testing on another kind of constrained mesh
rng = StableRNG(123)
R₁ = 1.0
R₂ = 2.0
outer_circle = CircularArc((R₂, 0.0), (R₂, 0.0), (0.0, 0.0))
inner_circle = CircularArc((R₁, 0.0), (R₁, 0.0), (0.0, 0.0), positive = false)
points = NTuple{2, Float64}[]
tri = triangulate(points; rng, boundary_nodes = [[[outer_circle]], [[inner_circle]]])
A = 2π * (R₂^2 - R₁^2)
refine!(tri; max_area = 2.0e-3A, min_angle = 33.0, rng)
fig, ax, sc = triplot(tri)
display(fig)

#%% Store the points, edges, triangles and boundary edges of the triangulation
List_Points = get_points(tri)
List_Edges = collect(each_solid_edge(tri)) 
List_Triangles = collect(tri.triangles) 

#Make a list of the boundary edges 
#Use built in function get_adjacent2vertex(tri,i), which for i≤0 gives all the
#boundary edges 
#Then also need to have (j,i) for (i,j) boundary edge
#This is needed when defining the velocity at the boundary 

List_Boundary_Edges = []
for i in each_ghost_vertex(tri)
    e = collect(get_adjacent2vertex(tri,i))
    append!(List_Boundary_Edges,e)
end
for e in List_Boundary_Edges
    if reverse(e) ∉ List_Boundary_Edges
        push!(List_Boundary_Edges,reverse(e))
    end
end


#%% Compute the gradient of p and divergence of u 

#Make initial values of velocity on edges 
# u of size np x np, where np is the number of vertices 
# u[i,j] = 0 if there is no edge between vertices i and j 
#if there is an edge then set initial value
#sparse array 
np = length(tri.points)

u = NDSparseArray{Float64}(np, np) 
for i in 1:np
    for j in 1:np 
        if (i,j) ∈ List_Edges 
            u[i,j] = rand(Float16)
            # u[i,j] = 1
            if (i,j) in List_Boundary_Edges
                u[i,j] = 0
            end
        end
    end
end

#Make initial value of pressure on cells 
#Build it as a tensor where entry (i,j,k) is nonzero if and only if there is a
#triangle between these points 
p = NDSparseArray{Float64}(np, np, np) 
for i in 1:np
    for j in 1:np 
        for k in 1:np 
            if (i,j,k) in List_Triangles
                p[i,j,k] = rand(Float16)
                # p[i,j,k] = i
            end
        end
    end
end

#Compute the divergence of u  
divu = NDSparseArray{Float64}(np, np, np) 
for i in 1:np 
    for j in 1:np 
        for k in 1:np 
            if (i,j,k) in List_Triangles
                divu[i,j,k] = Divergence(u,(i,j,k)) 
            end
        end
    end
end

#Compute the gradient of p 
gradp = NDSparseArray{Float64}(np, np) 
for i in 1:np
    for j in 1:np 
        if ((i,j) ∈ List_Edges) && ((i,j) ∉ List_Boundary_Edges)
            gradp[i,j] = Gradient(p,(i,j)) 
        end
    end
end

#Check if indeed the gradient and the divergence are adjoints 
println("(p,div(u))   = ",SparseInnerProduct(p,divu,"cell"))
println("-(u,grad(p)) = ",-SparseInnerProduct(u,gradp,"edge"))

#%%
#Build u on cells, by averaging over the edges 

function Average(u,(i,j,k))
    av = 0 
    l = Length((i,j))
    if hasindex(u,i,j)
        av += u[i,j] * l
    else
        av += u[j,i] * l
    end

    l = Length((j,k))
    if hasindex(u,j,k)
        av += u[j,k] * l
    else
        av += u[k,j] * l
    end

    l = Length((k,i))
    if hasindex(u,k,i)
        av += u[k,i] * l
    else
        av += u[i,k] * l
    end
    
    return av/Area((i,j,k))
end

u_cell = NDSparseArray{Float64}(np, np, np) 
for i in 1:np 
    for j in 1:np 
        for k in 1:np 
            if (i,j,k) ∈ List_Triangles
                u_cell[i,j,k] = Average(u,(i,j,k))
            end
        end
    end
end

#Compute the gradient of u using the cell average 
gradu =  NDSparseArray{Float64}(np, np) 
for i in 1:np
    for j in 1:np 
        if ((i,j) ∈ List_Edges) && ((i,j) ∉ List_Boundary_Edges)
            gradu[i,j] = Gradient(u_cell,(i,j)) 
        end
        if (i,j) ∈ List_Boundary_Edges
            gradu[i,j] = 0
        end
    end
end

#Compute the Laplacian of u as div(grad(u))
divgradu = NDSparseArray{Float64}(np, np, np) 
for i in 1:np 
    for j in 1:np 
        for k in 1:np 
            if (i,j,k) in List_Triangles
                divgradu[i,j,k] = Divergence(gradu,(i,j,k)) 
            end
        end
    end
end

