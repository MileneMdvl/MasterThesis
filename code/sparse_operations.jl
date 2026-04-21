#This file includes the operations for sparse matrices, namely: 
# - SparseInnerProduct
# - SparseMatVec 

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

    for i in eachindex(A)
            ind = collect(Tuple.(i))
            if ind in indices_to_sum 
                if hasindex(A,i) && hasindex(B,i)
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


#Function SparseMatVec
#Input: A Sparse Matrix 
#       b Sparse/Dense Vector 
#Output: Ab Vector, result of A*b
function SparseMatVec(A,b)
    Ab = zeros(size(b))
    #Get indices in which A is nonzero
    ind_nzA = findnz(A)[1]
    #Store b as a sparse vector 
    bb = NDSparseArray(b)
    ind_nzb = findnz(bb)[1] 
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
    ind_nzA = findnz(A)[1]
    #Store b as a sparse array for the sake of dimensions 
    bb = NDSparseArray(b)
    ind_nzb = findnz(bb)[1] 
    for ij in ind_nzA
        i,j = Tuple(ij)
        if i in ind_nzb
            bA[j] += A[ij] * b[i]
        end
    end
    return bA
end