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


#Build the stiffness matrix A
#Function StiffnessMatrix
#Input: n Int: number of vertices 
#Output: A Array of size nxn, the stiffness matrix 
function StiffnessMatrix(n)
    stiff_mat = SparseArrays.spzeros(n,n)
    for i in 1:n
        for j in 1:n 
            for k in 1:n_cells 
                K = cell_list[k]
                if i in K && j in K 
                    index_i = findall(x->x==i,K)[1]
                    index_j = findall(x->x==j,K)[1]
                    stiff_mat[i,j] += BuildStiffElem(index_i,index_j,K)
                end
            end
        end
    end
    return stiff_mat
end

#Build the stiffness element for entry i,j belonging to cell K 
#Function BuildStiffElem
#Input: i,j,K Int
#Output: A_ij the ij-th element of the stiffness matrix 
function BuildStiffElem(i,j,K)
    grad_phi_i = [dphi_dxi_1(i),dphi_dxi_2(i)]
    grad_phi_j = [dphi_dxi_1(j),dphi_dxi_2(j)]
    M = [dydxi_2(K) -dydxi_1(K); -dxdxi_2(K) dxdxi_1(K)] 
    det = JacobianDeterminant(K) 
    A_ij = 1/2 * 1/det * dot(M * grad_phi_i, M * grad_phi_j)
    return A_ij
end


#Define the Jacobian determinant on element K
function JacobianDeterminant(K)
    p = zeros(3,2)
    for i in 1:3 
        p[i,:] = vertex_list[K[i]]
    end
    det = (p[1,1]-p[3,1])*(p[2,2]-p[3,2])-(p[2,1]-p[3,1])*(p[1,2]-p[3,2])
    return det 
end

#Define the inverse of the Jacobian on element K 
function JacobianInverse(K) 

end

#Define the partial derivatives on element K 
function dxdxi_1(K) 
    return vertex_list[K[1]][1] - vertex_list[K[3]][1]
end

function dydxi_1(K) 
    return vertex_list[K[1]][2] - vertex_list[K[3]][2]
end

function dxdxi_2(K) 
    return vertex_list[K[2]][1] - vertex_list[K[3]][1]
end

function dydxi_2(K) 
    return vertex_list[K[2]][2] - vertex_list[K[3]][2]
end

#Define the partial derivatives of the shape functions in the reference frame

#∂ϕᵢ / ∂ξ₁
function dphi_dxi_1(i)
    if i == 1
        return 1
    elseif i == 2
        return 0 
    elseif i == 3 
        return -1 
    end
end

#∂ϕᵢ / ∂ξ₂
function dphi_dxi_2(i)
    if i == 1
        return 0
    elseif i == 2
        return 1 
    elseif i == 3 
        return -1 
    end
end



#Build the load vector f 
load_vec = zeros(n)

#Function: Advection 
#Input: u_face tensor defined at face centers 
#Output: u⋅∇u (Advection operator of u) defined at face centers
function Advection(u_face)
    #Get the cell interpolation of the face velocity
    u_cell = FaceToCellInterpolation(u_face)
    #Gradient of u defined on face centers 
    gradu = Gradient(u_cell)
    return SparseInnerProduct(u_face,gradu,"face")
end

#Build the load element for vertex i belonging to cell K
function BuildLoadElem(i,K)
    return 1/2 * Advection(u_face) 
end
