#This file includes the operations for sparse matrices, namely: 
# - Evaluate
# - SparseInnerProduct
# - SparseMatVec 

#Not needed since we now use the SparseArrays package

include("mesh_functions.jl")

#Function: Evaluate
#Input: Tensor G, Array A (either face or cell)
#Output: G[A], the value of G on A 
function Evaluate(G,A)
    ind_A = CartesianIndex(Tuple(A))
    return G[ind_A]
end

#Function SparseInnerProduct
#Input A,B: sparse arrays to compute the inner product of
#Output A⋅B Float: Inner Product (A,B) 
function SparseInnerProduct(A,B, type::String)
    if size(A) != size(B)
        println("Error: dimensions of input do not match")
        println("-> ",size(A), " does not equal ", size(B))
        return 
    end

    if size(A)[1] != size(A)[2]
        println("Error: input not square")
        println("-> ",size(A)[1], " does not equal ", size(A)[2])
        return 
    end

    #For the face innerproduct, we only want to sum over edge (i,j) not also
    #(j,i) (same for 3d)

    if type == "cell"
        indices_to_sum = UniqueList(cell_list)
    elseif  type == "face"
        indices_to_sum = UniqueList(face_list)
    end

    innerprod = 0

    ind_nzA = findall(!iszero, A)
    ind_nzB = findall(!iszero, B)

    for i in eachindex(A)
            ind = collect(Tuple.(i))
            if ind in indices_to_sum 
                if i in ind_nzA && i in ind_nzB
                    entry = A[i] * B[i] * Volume(ind) 
                    if type == "face"
                        entry *= DualEdge(ind)
                    end
                    innerprod += entry
                end
            end
        end

    if type != "cell" && type != "face"
        println("Error: type must be either 'cell' or 'face'")
    end
    
    return innerprod                
end

#Function InnerProdCell 
#Input: a, b Vector defined on cell centers 
#Output: Discrete inner product a⋅b 
function InnerProdCell(a,b)
    ip = 0
    nc = length(cell_list)
    if length(a) == length(b) == nc 
        for i in 1:nc 
            ip += Volume(cell_list[i]) * a[i] * b[i]
        end
    else
        println("Error: incorrect vector lengths, should correspond to the number of cells, ", nc)
        println("Otherwise try 'InnerProdFace'")
    end
    return ip
end

#Function InnerProdFace 
#Input: a, b Vector defined on face centers  
#Output: Discrete inner product a⋅b 
function InnerProdFace(a,b)
    ip = 0
    nf = length(face_list)
    if length(a) == length(b) == nf
        for i in 1:nf 
            e = face_list[i]
            ip += Volume(e) * DualEdge(e) * a[i] * b[i]
        end
    else
        println("Error: incorrect vector lengths, should correspond to the number of unique faces, ", nf)
        println("Otherwise try 'InnerProdCell'")
    end
    return ip
end


#Function SparseMatVec
#Input: A Sparse Matrix 
#       b Sparse/Dense Vector 
#Output: Ab Vector, result of A*b
function SparseMatVec(A,b)
    Ab = zeros(size(A)[1])
    #Get indices in which A is nonzero
    ind_nzA = findall(!iszero, A)
    ind_nzb = findall(!iszero, b)
    for ij in ind_nzA
        i,j = Tuple(ij)
        if j in ind_nzb
            Ab[i] += A[ij] * b[j]
        end
    end
    return Ab
end

#Function SparseVecMat
#Input: b Dense Vector
#       A Sparse Matrix  
#Output: bA Vector, result of b*A
function SparseVecMat(b,A)
    bA = zeros(size(b))
    #Get indices in which A is nonzero
    ind_nzA = findall(!iszero, A)
    ind_nzb = findall(!iszero, b)
    for ij in ind_nzA
        i,j = Tuple(ij)
        if i in ind_nzb
            bA[j] += A[ij] * b[i]
        end
    end
    return bA
end

#Function SparseMatMat
#Input: AA (Sparse) matrix
#       BB (Sparse) Matrix
#       Note the matrices don't have to be both sparse, but if both are dense
#       then it would be better to use built in matrix multiplication  
#Output: AB Vector, result of A*B
function SparseMatMat(A,B)
    local n,p,m = size(A)[1], size(A)[2], size(B)[2]
    if A isa Array 
        AA = sparse(A)
    else
        AA = copy(A)
    end 
    if B isa Array 
        BB = sparse(B)
    else 
        BB = copy(B)
    end
    if size(B)[1] != p
        println("Error: matrix dimensions do not match!")
        return 
    end
    AB = zeros(n,m)
    ind_nzA = findall(!iszero, A)
    ind_nzB = findall(!iszero, B)
    for i in 1:n 
        for j in 1:m 
            for k in 1:p 
                if CartesianIndex(i,k) in ind_nzA && CartesianIndex(k,j) in ind_nzB
                    AB[i,j] += AA[i,k] * BB[k,j]
                end
            end
        end
    end
    return AB
end
