#This file contains all the functions needed to build the stiffness matrix and
#load vector to solve the Poisson problem for pressure 

#To do: 
#      - Finish building the load vector
#      - Generalise to 3D case
#      - Make function for Jacobian inverse 
#      - Clean up file 
#      - Implement test case 

include("interpolation.jl")
include("mesh_functions.jl")
include("divgrad.jl")
include("sparse_operations.jl")


#Build the stiffness matrix A
#Function StiffnessMatrix
#Input: n Int: number of vertices 
#Output: A Array of size nxn, the stiffness matrix 
function StiffnessMatrix(n)
    stiff_mat = zeros(n,n)
    for i in 1:n
        for j in 1:n 
            for K in cell_list
                if i in K && j in K && i != j
                    index_i = findall(x->x==i,K)[1]
                    index_j = findall(x->x==j,K)[1]
                    stiff_mat[i,j] += BuildStiffElem(index_i,index_j,K)
                end
            end
        end
    end
    return NDSparseArray(stiff_mat)
end

#Build the stiffness element for entry i,j belonging to cell K 
#Function BuildStiffElem
#Input: i,j,K Int
#Output: A_ij the ij-th element of the stiffness matrix 
function BuildStiffElem(i,j,K)
    J = Jacobian(K)
    invJ = inv(J)
    detJ = det(J) 
    A_ij =  dot(invJ * GradPhi(i), invJ * GradPhi(j)) * detJ
    #Normalise depending on the dimension 
    d = length(vertex_list[1])
    if d == 2
        A_ij *= 1/2
    elseif d == 3
        A_ij *= 1/6
    end
    return A_ij
end


#Define the Jacobian on element K
function Jacobian(K)
    #Get d*d Jacobian matrix, for d the dimension of the domain 
    d = length(vertex_list[1])
    J = zeros(d,d)
    J[1,1] = dxdxi_1(K)
    J[1,2] = dydxi_1(K)
    J[2,1] = dxdxi_2(K)
    J[2,2] = dydxi_2(K)
    if d == 3 
        J[1,3] = dzdxi_1(K)
        J[2,3] = dzdxi_2(K)
        J[3,1] = dxdxi_3(K)
        J[3,2] = dydxi_3(K)
        J[3,3] = dzdxi_3(K)
    end
    return J 
end


#Define the partial derivatives on element K 
function dxdxi_1(K) 
    d = length(vertex_list[1])
    last_ind = d+1
    return vertex_list[K[1]][1] - vertex_list[K[last_ind]][1]
end

function dydxi_1(K) 
    d = length(vertex_list[1])
    last_ind = d+1
    return vertex_list[K[1]][2] - vertex_list[K[last_ind]][2]
end

function dzdxi_1(K)
    return vertex_list[K[1]][3] - vertex_list[K[4]][3]
end

function dxdxi_2(K) 
    d = length(vertex_list[1])
    last_ind = d+1
    return vertex_list[K[2]][1] - vertex_list[K[last_ind]][1]
end

function dydxi_2(K) 
    d = length(vertex_list[1])
    last_ind = d+1
    return vertex_list[K[2]][2] - vertex_list[K[last_ind]][2]
end

function dzdxi_2(K)
    return vertex_list[K[2]][3] - vertex_list[K[4]][3]
end

function dxdxi_3(K) 
    return vertex_list[K[3]][1] - vertex_list[K[4]][1]
end

function dydxi_3(K) 
    return vertex_list[K[3]][2] - vertex_list[K[4]][2]
end

function dzdxi_3(K)
    return vertex_list[K[3]][3] - vertex_list[K[4]][3]
end

#Define the gradient of the i-th shape function in the reference frame
function GradPhi(i)
    d = length(vertex_list[1])
    if d == 2
        return [dphi_dxi_1(i),dphi_dxi_2(i)]
    elseif d ==3   
        return [dphi_dxi_1(i),dphi_dxi_2(i),dphi_dxi_3(i)]
    end
end

#∂ϕᵢ / ∂ξ₁
function dphi_dxi_1(i)
    d = length(vertex_list[1])
    if i == 1
        return 1
    elseif i == 2
        return 0 
    elseif i == 3 && d == 2 
        return -1 
    elseif i == 3 && d == 3
        return 0 
    elseif i == 4 
        return -1 
    end
end

#∂ϕᵢ / ∂ξ₂
function dphi_dxi_2(i)
    d = length(vertex_list[1])
    if i == 1
        return 0
    elseif i == 2
        return 1 
    elseif i == 3 && d == 2 
        return -1 
    elseif i == 3 && d == 3
        return 0 
    elseif i == 4 
        return -1 
    end
end

#∂ϕᵢ / ∂ξ₃
function dphi_dxi_3(i)
    if i == 1
        return 0
    elseif i == 2
        return 0 
    elseif i == 3 
        return 1 
    elseif i == 4 
        return -1 
    end
end


#Build the load vector f 
function LoadVector(n,u_face,rho=1)
    load_vec = zeros(n)
    for i in 1:n 
        for K in cell_list 
            if i ∈ K 
                load_vec[i] += BuildLoadElem(i,K,u_face,rho)
            end
        end
    end
    return load_vec
end

#Build the load element for vertex i belonging to cell K
function BuildLoadElem(i,K,u_face,rho)
    J = Jacobian(K)
    invJ = inv(J)
    detJ = det(J) 
    adv = Advection(u_face)
    grad_phi = GradPhi(i)
    println(adv)
    println(invJ*grad_phi)
    f_i = -rho * detJ * dot(adv, invJ * grad_phi)
    d = length(vertex_list[1])
    if d == 2
        f_i *= 1/2
    elseif d == 3
        f_i *= 1/6
    end
    return f_i 
end


#Function: Advection 
#Input: u_face tensor defined at face centers 
#Output: u⋅∇u (Advection operator of u) defined at face centers
function Advection(u_face)
    #Get the cell interpolation of the face velocity
    u_cell = FaceToCellInterpolation(u_face)
    #Gradient of u defined on face centers 
    gradu = Gradient(u_cell)
    u_vert = VertexInterpolation(u_face)
    return SparseVecMat(u_vert, gradu)
end


