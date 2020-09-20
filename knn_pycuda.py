import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule
import skcuda.cublas as cublas 
import pycuda.gpuarray as gpuarray
import sys



'''
/**
 * Computes the squared Euclidean distance matrix between the query points and the reference points.
 *
 * @param ref          refence points stored in the global memory
 * @param ref_width    number of reference points
 * @param ref_pitch    pitch of the reference points array in number of column
 * @param query        query points stored in the global memory
 * @param query_width  number of query points
 * @param query_pitch  pitch of the query points array in number of columns
 * @param height       dimension of points = height of texture `ref` and of the array `query`
 * @param dist         array containing the query_width x ref_width computed distances
'''
compute_distances_mod = SourceModule("""
__global__ void compute_distances(float * ref,
                                  int     ref_width,
                                  int     ref_pitch,
                                  float * query,
                                  int     query_width,
                                  int     query_pitch,
                                  int     height,
                                  float * dist) {
    const int BLOCK_DIM = 16;
    // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
    __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

    // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
    __shared__ int begin_A;
    __shared__ int begin_B;
    __shared__ int step_A;
    __shared__ int step_B;
    __shared__ int end_A;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    

    // Initializarion of the SSD for the current thread
    float ssd = 0.f;

    // Loop parameters
    begin_A = BLOCK_DIM * blockIdx.y;
    begin_B = BLOCK_DIM * blockIdx.x;
    step_A  = BLOCK_DIM * ref_pitch;
    step_B  = BLOCK_DIM * query_pitch;
    end_A   = begin_A + (height-1) * ref_pitch;

    // Conditions
    int cond0 = (begin_A + tx < ref_width); // used to write in shared memory
    int cond1 = (begin_B + tx < query_width); // used to write in shared memory & to computations and to write in output array 
    int cond2 = (begin_A + ty < ref_width); // used to computations and to write in output matrix

    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {

        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        if (a/ref_pitch + ty < height) {
            shared_A[ty][tx] = (cond0)? ref[a + ref_pitch * ty + tx] : 0;
            shared_B[ty][tx] = (cond1)? query[b + query_pitch * ty + tx] : 0;
        }
        else {
            shared_A[ty][tx] = 0;
            shared_B[ty][tx] = 0;
        }

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
        if (cond2 && cond1) {
            for (int k = 0; k < BLOCK_DIM; ++k){
                float tmp = shared_A[k][ty] - shared_B[k][tx];
                ssd += tmp*tmp;
            }
        }

        // Synchronize to make sure that the preceeding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory; each thread writes one element
    if (cond2 && cond1) {
        dist[ (begin_A + ty) * query_pitch + begin_B + tx ] = ssd;
    }
}

""")



'''
 * For each reference point (i.e. each column) finds the k-th smallest distances
 * of the distance matrix and their respective indexes and gathers them at the top
 * of the 2 arrays.
 *
 * Since we only need to locate the k smallest distances, sorting the entire array
 * would not be very efficient if k is relatively small. Instead, we perform a
 * simple insertion sort by eventually inserting a given distance in the first
 * k values.
 *
 * @param dist         distance matrix
 * @param dist_pitch   pitch of the distance matrix given in number of columns
 * @param index        index matrix
 * @param index_pitch  pitch of the index matrix given in number of columns
 * @param width        width of the distance matrix and of the index matrix
 * @param height       height of the distance matrix
 * @param k            number of values to find 
'''


modified_insertion_sort_mod = SourceModule("""
__global__ void modified_insertion_sort(float * dist,
                                        int     dist_pitch,
                                        int *   index,
                                        int     index_pitch,
                                        int     width,
                                        int     height,
                                        int     k){

    // Column position
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Do nothing if we are out of bounds
    if (xIndex < width) {

        // Pointer shift
        float * p_dist  = dist  + xIndex;
        int *   p_index = index + xIndex;

        // Initialise the first index
        p_index[0] = 0;

        // Go through all points
        for (int i=1; i<height; ++i) {

            // Store current distance and associated index
            float curr_dist = p_dist[i*dist_pitch];
            int   curr_index  = i;

            // Skip the current value if its index is >= k and if it's higher the k-th slready sorted mallest value
            if (i >= k && curr_dist >= p_dist[(k-1)*dist_pitch]) {
                continue;
            }

            // Shift values (and indexes) higher that the current distance to the right
            int j = min(i, k-1);
            while (j > 0 && p_dist[(j-1)*dist_pitch] > curr_dist) {
                p_dist[j*dist_pitch]   = p_dist[(j-1)*dist_pitch];
                p_index[j*index_pitch] = p_index[(j-1)*index_pitch];
                --j;
            }

            // Write the current distance and index at their position
            p_dist[j*dist_pitch]   = curr_dist;
            p_index[j*index_pitch] = curr_index; 
        }
    }
}
""")


'''
 * Computes the square root of the first k lines of the distance matrix.
 *
 * @param dist   distance matrix
 * @param width  width of the distance matrix
 * @param pitch  pitch of the distance matrix given in number of columns
 * @param k      number of values to consider
'''

compute_sqrt_mod = SourceModule(""" 
__global__ void compute_sqrt(float * dist, int width, int pitch, int k){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (xIndex<width && yIndex<k)
        dist[yIndex*pitch + xIndex] = sqrt(dist[yIndex*pitch + xIndex]);
}
""")



def copy2D_array_to_device(dst, src, width_in_bytes, height):
    copy = drv.Memcpy2D()
    copy.set_src_array(src)
    copy.set_dst_device(dst)
    copy.height = height
    copy.dst_pitch = copy.src_pitch = copy.width_in_bytes = width_in_bytes
    copy(aligned=True)


"""
 * For each reference point (i.e. each column) finds the k-th smallest distances
 * of the distance matrix and their respective indexes and gathers them at the top
 * of the 2 arrays.
 *
 * Since we only need to locate the k smallest distances, sorting the entire array
 * would not be very efficient if k is relatively small. Instead, we perform a
 * simple insertion sort by eventually inserting a given distance in the first
 * k values.
 *
 * @param dist         distance matrix
 * @param dist_pitch   pitch of the distance matrix given in number of columns
 * @param index        index matrix
 * @param index_pitch  pitch of the index matrix given in number of columns
 * @param width        width of the distance matrix and of the index matrix
 * @param height       height of the distance matrix
 * @param k            number of values to find
""" 

modified_insertion_sort_mod = SourceModule("""
__global__ void modified_insertion_sort(float * dist,
                                        int     dist_pitch,
                                        int *   index,
                                        int     index_pitch,
                                        int     width,
                                        int     height,
                                        int     k){

    // Column position
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Do nothing if we are out of bounds
    if (xIndex < width) {

        // Pointer shift
        float * p_dist  = dist  + xIndex;
        int *   p_index = index + xIndex;

        // Initialise the first index
        p_index[0] = 0;

        // Go through all points
        for (int i=1; i<height; ++i) {

            // Store current distance and associated index
            float curr_dist = p_dist[i*dist_pitch];
            int   curr_index  = i;

            // Skip the current value if its index is >= k and if it's higher the k-th slready sorted mallest value
            if (i >= k && curr_dist >= p_dist[(k-1)*dist_pitch]) {
                continue;
            }

            // Shift values (and indexes) higher that the current distance to the right
            int j = min(i, k-1);
            while (j > 0 && p_dist[(j-1)*dist_pitch] > curr_dist) {
                p_dist[j*dist_pitch]   = p_dist[(j-1)*dist_pitch];
                p_index[j*index_pitch] = p_index[(j-1)*index_pitch];
                --j;
            }

            // Write the current distance and index at their position
            p_dist[j*dist_pitch]   = curr_dist;
            p_index[j*index_pitch] = curr_index; 
        }
    }
}
""")






"""
 * Computes the squared norm of each column of the input array.
 *
 * @param array   input array
 * @param width   number of columns of `array` = number of points
 * @param pitch   pitch of `array` in number of columns
 * @param height  number of rows of `array` = dimension of the points
 * @param norm    output array containing the squared norm values
""" 

compute_squared_norm_mod = SourceModule(""" 
    __global__ void compute_squared_norm(float * array, int width, int pitch, int height, float * norm){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (xIndex<width){
        float sum = 0.f;
        for (int i=0; i<height; i++){
            float val = array[i*pitch+xIndex];
            sum += val*val;
        }
        norm[xIndex] = sum;
    }
}
""")




"""
 * Add the reference points norm (column vector) to each colum of the input array.
 *
 * @param array   input array
 * @param width   number of columns of `array` = number of points
 * @param pitch   pitch of `array` in number of columns
 * @param height  number of rows of `array` = dimension of the points
 * @param norm    reference points norm stored as a column vector
"""

add_reference_points_norm_mod = SourceModule(""" 
__global__ void add_reference_points_norm(float * array, int width, int pitch, int height, float * norm){
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int xIndex = blockIdx.x * blockDim.x + tx;
    unsigned int yIndex = blockIdx.y * blockDim.y + ty;
    __shared__ float shared_vec[16];
    if (tx==0 && yIndex<height)
        shared_vec[ty] = norm[yIndex];
    __syncthreads();
    if (xIndex<width && yIndex<height)
        array[yIndex*pitch+xIndex] += shared_vec[ty];
}
""")

"""/**
 * Adds the query points norm (row vector) to the k first lines of the input
 * array and computes the square root of the resulting values.
 *
 * @param array   input array
 * @param width   number of columns of `array` = number of points
 * @param pitch   pitch of `array` in number of columns
 * @param k       number of neighbors to consider
 * @param norm     query points norm stored as a row vector
"""

add_query_points_norm_and_sqrt_mod = SourceModule("""  
__global__ void add_query_points_norm_and_sqrt(float * array, int width, int pitch, int k, float * norm){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (xIndex<width && yIndex<k)
        array[yIndex*pitch + xIndex] = sqrt(array[yIndex*pitch + xIndex] + norm[xIndex]);
}
""")



def knn_cuda_global(ref,
                    ref_nb,
                    query,
                    query_nb,
                    dim,
                    k,
                    knn_dist,
                    knn_index):

                         
    BLOCK_DIM = 16

    #Most nVidia devices only support single precision:
    ref = ref.astype(np.float32)
    query = query.astype(np.float32)
    knn_dist = knn_dist.astype(np.float32)
    knn_index = knn_index.astype(np.int32) 
    
    #Allocate memory on device
    ref_dev = drv.mem_alloc(ref.nbytes)
    query_dev = drv.mem_alloc(query.nbytes)
    dist_dev = drv.mem_alloc(knn_dist.nbytes)
    index_dev = drv.mem_alloc(knn_index.nbytes)

    #Copy data from the host to the device
    drv.memcpy_htod(ref_dev, ref)
    drv.memcpy_htod(query_dev, query)
    drv.memcpy_htod(dist_dev, knn_dist)
    drv.memcpy_htod(index_dev, knn_index)

    #Deduce pitch values (not in bytes)
    ref_pitch = np.int32(ref.strides[0]/ref.dtype.itemsize)
    query_pitch = np.int32(query.strides[0]/query.dtype.itemsize)
    dist_pitch = np.int32(knn_dist.strides[0]/knn_dist.dtype.itemsize)
    index_pitch = np.int32(knn_index.strides[0]/knn_index.dtype.itemsize)
   
    #Check pitch values
    if (query_pitch != dist_pitch) or (query_pitch != index_pitch):
        print('ERROR: Invalid pitch value')
        sys.exit(1)


    grid_x = query_nb / BLOCK_DIM
    grid_y = ref_nb / BLOCK_DIM
    if (query_nb % BLOCK_DIM != 0):
        grid_x += 1
    if (ref_nb   % BLOCK_DIM != 0):
        grid_y += 1


    
    #Compute the squared Euclidean distances
    compute_distances = compute_distances_mod.get_function("compute_distances")
    compute_distances(ref_dev, ref_nb, ref_pitch, query_dev, query_nb, query_pitch, dim, dist_dev, block=(16,16,1), grid = (int(grid_x), int(grid_y),1)) #grid=(32,32,1)) #(np.int32(grid_x),np.int32(grid_y)))
    drv.memcpy_dtoh(knn_dist, dist_dev)

    #Sort the distances with their respective indexes
    modified_insertion_sort = modified_insertion_sort_mod.get_function("modified_insertion_sort")

    grid_x = query_nb / 256
    if (query_nb % 256 != 0):
        grid_x += 1
        
    modified_insertion_sort(dist_dev, dist_pitch, index_dev, index_pitch, query_nb, ref_nb, k, block=(256,1,1), grid=(int(grid_x),1,1))



    #Compute the square root of the k smallest distances
    compute_sqrt = compute_sqrt_mod.get_function("compute_sqrt")

    grid_x = query_nb / 16
    grid_y = k / 16
    if (query_nb % 16 != 0):
        grid_x += 1
    if (k % 16 != 0):
        grid_y += 1
        
    compute_sqrt(dist_dev, query_nb, query_pitch, k, block=(16,16,1), grid=(int(grid_x),int(grid_y),1))


    #Copy k smallest distances / indexes from the device to the host
    drv.memcpy_dtoh(knn_index, index_dev)
    drv.memcpy_dtoh(knn_dist, dist_dev)
    
    return True





def knn_cublas(ref, 
               ref_nb, 
               query, 
               query_nb, 
               dim, 
               k, 
               knn_dist,
               knn_index): 

    #Initialize CUBLAS
    context = cublas.cublasCreate()
  

    BLOCK_DIM = 16

    #Most nVidia devices only support single precision:
    ref = ref.astype(np.float32)
    query = query.astype(np.float32)
    knn_dist = knn_dist.astype(np.float32)
    knn_index = knn_index.astype(np.int32) 
    
    #Allocate memory on device
    ref_dev = drv.mem_alloc(ref.nbytes)
    query_dev = drv.mem_alloc(query.nbytes)
    dist_dev = drv.mem_alloc(knn_dist.nbytes)
    index_dev = drv.mem_alloc(knn_index.nbytes)
    ref_norm_dev =  drv.mem_alloc(ref.nbytes)
    query_norm_dev = drv.mem_alloc(query.nbytes)
    
    #Copy data from host to device
    drv.memcpy_htod(ref_dev, ref)
    drv.memcpy_htod(query_dev, query)
    drv.memcpy_htod(dist_dev, knn_dist)
    drv.memcpy_htod(index_dev, knn_index)

    
    #Deduce pitch values (not in bytes)
    ref_pitch = np.int32(ref.strides[0]/ref.dtype.itemsize)
    query_pitch = np.int32(query.strides[0]/query.dtype.itemsize)
    dist_pitch = np.int32(knn_dist.strides[0]/knn_dist.dtype.itemsize)
    index_pitch = np.int32(knn_index.strides[0]/knn_index.dtype.itemsize)

    #Check pitch values
    if (query_pitch != dist_pitch or query_pitch != index_pitch): 
        print("ERROR: Invalid pitch value")
        sys.exit(1)
    
    
    
    

    # Compute the squared norm of the reference points
    compute_squared_norm = compute_squared_norm_mod.get_function("compute_squared_norm")

    grid_x = ref_nb / 256
    if (ref_nb % 256 != 0):
        grid_x += 1

    compute_squared_norm(ref_dev, ref_nb, ref_pitch, dim, ref_norm_dev,
        block=(256,1,1), grid=(int(grid_x),1,1))


    # Compute the squared norm of the query points
    compute_squared_norm = compute_squared_norm_mod.get_function("compute_squared_norm")

    grid_x = query_nb / 256
    if (query_nb % 256 != 0):
        grid_x += 1

    compute_squared_norm(query_dev, query_nb, query_pitch, dim, query_norm_dev,
        block=(256,1,1), grid=(int(grid_x),1,1))

    
    # Computation of query*transpose(reference)  
    cublas.cublasSgemm(context, 'n', 't', query_pitch, ref_pitch, dim, np.float32(-2.0), query_dev, query_pitch, ref_dev, ref_pitch, np.float32(0.0), dist_dev, query_pitch)
    drv.memcpy_dtoh(knn_dist, dist_dev)

    cublas.cublasDestroy(context)



    # Add reference points norm
    add_reference_points_norm = add_reference_points_norm_mod.get_function("add_reference_points_norm")

    grid_x = query_nb / 16
    grid_y = ref_nb / 16
    
    if (query_nb % 16 != 0):
        grid_x += 1
      
    if (ref_nb % 16 != 0):
        grid_y += 1  

    add_reference_points_norm(dist_dev, query_nb, dist_pitch, ref_nb, ref_norm_dev,
        block=(16,16,1), grid=(int(grid_x),int(grid_y),1))
    

    # Sort each column
    modified_insertion_sort = modified_insertion_sort_mod.get_function("modified_insertion_sort")

    
    grid_x = query_nb / 256
    if (query_nb % 256 != 0):
        grid_x += 1

    modified_insertion_sort(dist_dev, dist_pitch, index_dev, index_pitch, query_nb, ref_nb, k,
        block=(256,1,1), grid=(int(grid_x),1,1))
    



    # Add query norm and compute the square root of the of the k first elements
    add_query_points_norm_and_sqrt = add_query_points_norm_and_sqrt_mod.get_function("add_query_points_norm_and_sqrt")

    grid_x = query_nb / 16
    grid_y = k / 16
    if (query_nb % 16 != 0):
        grid_x += 1
    if (k % 16 != 0):
        grid_y += 1
    
    add_query_points_norm_and_sqrt(dist_dev, query_nb, dist_pitch, k, query_norm_dev,
        block=(16, 16, 1), grid=(int(grid_x),int(grid_y),1))
    
    #Copy k smallest distances / indexes from the device to the host    
    drv.memcpy_dtoh(knn_index, index_dev)
    drv.memcpy_dtoh(knn_dist, dist_dev)

    return True

