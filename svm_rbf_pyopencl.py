import pyopencl as cl
import numpy as np
import skcuda.cublas as cublas 
import random
import math
import time
import pycuda.gpuarray as gpuarray

#Obtain an OpenCL platform.
platform = cl.get_platforms()[0]
     
#Obtain a device id for at least one device
device = platform.get_devices()[0]
     
#Create a context for the selected device.
ctx = cl.Context([device])
     
#OpenCL kernels
predict_array_mod = cl.Program(ctx,"""
__kernel void predict_array(__global float * kArray, __global float *w, __global int * d_y_pred, unsigned int feature_size) {
    int i=get_group_id(0)*get_local_size(0)+get_local_id(0);
    
    float dot_product = 0;
    
    for(int j = 0; j < feature_size; j++) {
        dot_product += w[j]*kArray[i*feature_size + j];
    }

    if (dot_product >= 0) {
        d_y_pred[i] = 1;
    }

    else d_y_pred[i] = -1;
}

""").build()




copy_batch_mod = cl.Program(ctx, """
__kernel void copy_batch(__global float * kArray, __global float *xi, unsigned int idx, unsigned int feature_size, unsigned int batch_size) {
    int i=get_group_id(0)*get_local_size(0)+get_local_id(0);
    if(i < feature_size*batch_size) {
        xi[i] = kArray[idx*feature_size+i];
    }
}


""").build()

select_samples_mod = cl.Program(ctx, """ 
__kernel void select_samples(__global float * xi,__global float * w, unsigned int feature_size, unsigned int idx, __global int * label, unsigned int batch_size) {
    int i=get_group_id(0)*get_local_size(0)+get_local_id(0);
    if(i < batch_size) {
        float dot_product = 0;
    
        for(int j = 0; j < feature_size; j++) {
            dot_product += w[j]*xi[i*feature_size + j];
        }

        if(dot_product*label[idx+i] >= 1) {
            for(int j = 0; j < feature_size; j++) {
                xi[i*feature_size + j] = 0;
            }
        }
    }   
}
""").build()


reduce_by_samples_mod = cl.Program(ctx, """ 
__kernel void reduce_by_samples(__global float *yi_xi, __global float *xi, unsigned int batch_size, unsigned int feature_size, unsigned int idx, __global int * label) {
    int i=get_group_id(0)*get_local_size(0)+get_local_id(0);
    
    if(i < feature_size) {
        float sum = 0;
    
        for(int j = 0; j < batch_size; j++) {
            sum += xi[j*feature_size + i]*label[idx+j]; 
        }

        yi_xi[i] = sum;
    }    
}
""").build()



update_w_mod = cl.Program(ctx,""" 

__kernel void update_w(__global float *yi_xi, __global float *w, __global float *next_w, float nt, float c, int batch_size, unsigned int feature_size) {

    int i=get_group_id(0)*get_local_size(0)+get_local_id(0);
    if (i < feature_size) {
        next_w[i] = w[i] - nt*c*w[i] + (nt/batch_size)*yi_xi[i];
    } 
}
""").build()


copy_array_mod = cl.Program(ctx, """ 
__kernel void copy_array(__global float * a, __global float * b, int feature_size) {
    int i=get_group_id(0)*get_local_size(0)+get_local_id(0);
    if(i < feature_size) {
        a[i] = b[i];
    }    
}
""").build()



#rbf kernel 
mod = cl.Program(ctx,"""
__kernel void rbf_kernel(__global float * data,
                           int     query_nb,
                           int     ref_nb,
                           float sigma)

{
    
int idx = get_group_id(0)*get_local_size(0)+get_local_id(0);

if (idx<query_nb*ref_nb){    
data[idx]  /= -2.0*sigma*sigma;
data[idx] = exp(data[idx]);
}    
}
""").build()




"""
 * Computes the squared norm of each column of the input array.
 *
 * @param array   input array
 * @param width   number of columns of `array` = number of points
 * @param pitch   pitch of `array` in number of columns
 * @param height  number of rows of `array` = dimension of the points
 * @param norm    output array containing the squared norm values
""" 

compute_squared_norm_mod = cl.Program(ctx,""" 
    __kernel void compute_squared_norm(__global float * array, int width, int pitch, int height,__global float * norm){
    unsigned int xIndex = get_group_id(0)*get_local_size(0)+get_local_id(0);
    if (xIndex<width){
        float sum = 0.f;
        for (int i=0; i<height; i++){
            float val = array[i*pitch+xIndex];
            sum += val*val;
        }
        norm[xIndex] = sum;
    }
}
""").build()




"""
 * Add the reference points norm (column vector) to each colum of the input array.
 *
 * @param array   input array
 * @param width   number of columns of `array` = number of points
 * @param pitch   pitch of `array` in number of columns
 * @param height  number of rows of `array` = dimension of the points
 * @param norm    reference points norm stored as a column vector
"""

add_reference_points_norm_mod = cl.Program(ctx,""" 
__kernel void add_reference_points_norm(__global float * array, int width, int pitch, int height,__global float * norm){
    unsigned int tx = get_local_id(0);
    unsigned int ty = get_local_id(1);
    unsigned int xIndex = get_group_id(0)*get_local_size(0) + tx;
    unsigned int yIndex = get_group_id(1)*get_local_size(1) + ty;
    __local float shared_vec[16];
    if (tx==0 && yIndex<height)
        shared_vec[ty] = norm[yIndex];
    __syncthreads();
    if (xIndex<width && yIndex<height)
        array[yIndex*pitch+xIndex] += shared_vec[ty];
}
""").build()

"""/**
 * Adds the query points norm (row vector) to the input
 * array and computes the square root of the resulting values.
 *
 * @param array   input array
 * @param width   number of columns of `array` = number of points
 * @param pitch   pitch of `array` in number of columns
 * @param k       number of training examples
 * @param norm    query points norm stored as a row vector
"""

add_query_points_norm_mod = cl.Program(ctx,"""  
__kernel void add_query_points_norm(__global float * array, int width, int pitch, int k, __global float * norm){
    unsigned int xIndex = get_group_id(0)*get_local_size(0)+get_local_id(0);
    unsigned int yIndex = get_group_id(1)*get_local_size(1)+get_local_id(1);
    if (xIndex<width && yIndex<k)
        array[yIndex*pitch + xIndex] = array[yIndex*pitch + xIndex] + norm[xIndex];
}
""").build()




def rbf_kernel(ref, query, sigma, queue, mf):
    #transfer data from host to device
    ref_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ref)
    query_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=query)  
    
    #data parameters
    ref_nb = np.int32(ref.shape[1]) #the number of reference points
    query_nb = np.int32(query.shape[1]) #the number of query points
    dim = np.int32(query.shape[0]) #dimension of data = the number of features
    
    #matrix to store distances
    knn_dist = np.zeros([ref_nb, query_nb]).astype(np.float32)
          
    BLOCK_DIM = 16 #dimension of blocks
    
    #Allocate memory on device
    dist_dev =  cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=knn_dist) 
    ref_norm_dev =  cl.Buffer(ctx, mf.READ_WRITE, ref.nbytes) 
    query_norm_dev = cl.Buffer(ctx, mf.READ_WRITE, query.nbytes) 

    
    #Deduce pitch values (not in bytes)
    ref_pitch = np.int32(ref.strides[0]/ref.dtype.itemsize) 
    query_pitch = np.int32(query.strides[0]/query.dtype.itemsize) 
    dist_pitch = np.int32(knn_dist.strides[0]/knn_dist.dtype.itemsize)

    
    #Check pitch values
    if (query_pitch != dist_pitch):
        print('ERROR: Invalid pitch value')
        
    #Initialize CUBLAS
    context = cublas.cublasCreate() 

    BLOCK_DIM = 16 

    
        
    # Compute the squared norm of the reference points
    grid_x = ref_nb / 256
    if (ref_nb % 256 != 0):
        grid_x += 1
        
    compute_squared_norm_mod.compute_squared_norm(queue, (int(grid_x)*256,1),(256,1), ref_g, ref_nb, ref_pitch, dim, ref_norm_dev)
                                              
    grid_x = query_nb / 256
    if (query_nb % 256 != 0):
        grid_x += 1
        
    compute_squared_norm_mod.compute_squared_norm(queue, (int(grid_x)*256,1),(256,1), query_g, query_nb, query_pitch, dim, query_norm_dev)

    dist=gpuarray.to_gpu(knn_dist)
    #Computation of query*transpose(reference)  
    cublas.cublasSgemm(context, 'n', 't', query_pitch, ref_pitch, dim, np.float32(-2.0), gpuarray.to_gpu(query).gpudata, query_pitch, gpuarray.to_gpu(ref).gpudata, ref_pitch, np.float32(0.0), dist.gpudata, query_pitch)
    dist_back = dist.get()
    cublas.cublasDestroy(context)

    dist_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=dist_back) 

    # Add reference points norm
    grid_x = query_nb / 16
    grid_y = ref_nb / 16
    
    if (query_nb % 16 != 0):
        grid_x += 1
      
    if (ref_nb % 16 != 0):
        grid_y += 1  
    
    add_reference_points_norm_mod.add_reference_points_norm(queue, (int(grid_x)*16,int(grid_y)*16), (16,16), dist_dev, query_nb, dist_pitch, ref_nb, ref_norm_dev)    
        
    # Add query norm and compute
    grid_x = query_nb / 16
    grid_y = ref_nb / 16
    if (query_nb % 16 != 0):
        grid_x += 1
    if (ref_nb % 16 != 0):
        grid_y += 1
    
    add_query_points_norm_mod.add_query_points_norm(queue, (int(grid_x)*16,int(grid_y)*16), (16,16), dist_dev, query_nb, dist_pitch, ref_nb, query_norm_dev) 

    #calculate rbf kernel matrix 
    grid_x = query_nb*ref_nb / 32

    if (query_nb*ref_nb % 32 != 0):
        grid_x += 1

    mod.rbf_kernel(queue, (int(grid_x)*32,1), (32,1),  dist_dev, query_nb, ref_nb, sigma)
 
    return dist_dev


#training function
def svm_fit(data_g, label_g, feature_size, batch_size, epochs, c, label_size, queue, mf):
   
    #allocate memory
    arr = np.ones([batch_size,feature_size]).astype(np.float32)    
    xi = cl.Buffer(ctx, mf.WRITE_ONLY, arr.nbytes)

    w_np = np.zeros([1,feature_size]).astype(np.float32)
    w = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=w_np)
    
    yi_xi_np = np.ones([1,feature_size]).astype(np.float32)
    yi_xi = cl.Buffer(ctx, mf.WRITE_ONLY, yi_xi_np.nbytes)
    
    next_w_np = np.zeros([1,feature_size]).astype(np.float32)

    start = time.time()
    #training
    for t in range(1, epochs) :

        next_w = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=next_w_np) 
        
        nt = 1/(c*t)
        
        idx = random.randint(1,100000) % (label_size - batch_size)
  
        copy_batch_mod.copy_batch(queue, (int(math.ceil(batch_size*feature_size/47000.0))*47000,1), (int(math.ceil(batch_size*feature_size/47000.0)),1), data_g, xi, np.int32(idx), feature_size, batch_size)        
        select_samples_mod.select_samples(queue, (int(math.ceil(batch_size/47000.0))*47000, 1), (int(math.ceil(batch_size/47000.0)),1), xi, w, feature_size, np.int32(idx), label_g, batch_size)
        reduce_by_samples_mod.reduce_by_samples(queue, (int(math.ceil(feature_size/512.0))*512,1), (int(math.ceil(feature_size/512.0)),1), yi_xi, xi, batch_size, feature_size, np.int32(idx), label_g)
        update_w_mod.update_w(queue, (int(math.ceil(feature_size/47000.0))*47000,1), (int(math.ceil(feature_size/47000.0)),1), yi_xi, w, next_w, np.float32(nt), c, batch_size, feature_size)
        copy_array_mod.copy_array(queue, (int(math.ceil(feature_size/47000.0))*47000,1), (int(math.ceil(feature_size/47000.0)),1), w, next_w, feature_size)
   
    end = time.time()
    delta = end-start
    print('One iteration of training took %.2f sec' % delta)
    
    #copy final weights from device to host
    cl.enqueue_copy(queue, w_np, w) 

    return w_np 



#function to make predictions
def predict(data, w, num):
    pred = []
    for i in range(num):
        dotproduct = np.sum(np.multiply(data[i][:], w))
        if dotproduct>0:
            pred.append(1)
        else:
            pred.append(-1)         
    return np.array(pred)     

#function to calculate accuracyf
def accuracy (label, pred_label):
    corr = 0
    for i in range(pred_label.shape[0]):
        if (label[i] == pred_label[i]):
            corr +=1
    return corr/label.shape[0]


def load_data():
    train = np.loadtxt('train_svm1.csv', delimiter=',').astype(np.float32) 
    train_label = np.loadtxt('train_label_svm1.csv').astype(np.int32) 
    test = np.loadtxt('test_svm1.csv', delimiter=',').astype(np.float32) 
    test_label = np.loadtxt('test_label_svm1.csv').astype(np.int32)  
    
        
    
    return train, train_label, test, test_label



def main():
    #load data
    train_data, train_label, test_data, test_label = load_data()    

    #Parameters
    c = np.float32(0.0001) #the penalty parameter 
    epochs = np.int32(100)
    num_iterations = np.int32(10)    
    positive_class = np.int32(1)     
    sigma = np.float32(3.0) 
    batch_size = np.int32(200)
    feature_size = np.int32(train_data.shape[1]) #the number of distances to the reference points
    test_size = np.int32(test_data.shape[1]) #the number of test examples
    train_size = feature_size #the number of train examples, which is obviously equal to the feature_size since feature size is the number of distances to the training points     
    print ("Started training SVM: ") 

    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    start = time.time()
    rbf_matrix = rbf_kernel(train_data, train_data, sigma, queue, mf) 
    end = time.time()
    delta = end-start
    print('Computing rbf kernel took: %.2f sec' % delta)
    
    #transfer data from host to device
    label_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=train_label) 
    
    mean_accuracy=[]
    
    for i in range(num_iterations):
        #training
        w = svm_fit(rbf_matrix, label_g, feature_size, batch_size, epochs, c, train_size, queue, mf)
        #testing
        test_rbf_matrix = rbf_kernel(test_data, train_data, sigma, queue, mf) 
        rbf = np.zeros([test_size, train_size]).astype(np.float32)
        cl.enqueue_copy(queue, rbf, test_rbf_matrix) 
        predictions = predict(rbf, w, test_size) 
        acc = accuracy(test_label, predictions)
        mean_accuracy.append(acc)
        print('Accuracy in the given iteration: %.4f' % acc)
    print('Mean accuracy: %.4f' % np.mean(np.array(mean_accuracy)))
    
    return 0

if __name__ == "__main__":
    main() 

