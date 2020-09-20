import pyopencl as cl
import numpy as np
import pycuda.gpuarray as gpuarray
import random
import math
import time
import math
import skcuda

#Obtain an OpenCL platform.
platform = cl.get_platforms()[0]
     
#Obtain a device id for at least one device
device = platform.get_devices()[0]
     
#Create a context for the selected device.
ctx = cl.Context([device])
     
#OpenCL kernels

power_mod = cl.Program(ctx,""" 
__kernel void power(__global float * array, int order, int num){
    int i=get_group_id(0)*get_local_size(0)+get_local_id(0);
    if (i<num){
    array[i]= pow(array[i], order);
}
}


""").build()

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

#polynomial kernel matrix computation function
def polynomial_kernel(x,y, gamma, coeff, order, queue, mf):
    culinalg.init()
    
    temp_gpu = gamma*skcuda.linalg.dot(x, skcuda.linalg.transpose(y)) + coeff
    kernel = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=temp_gpu.get()) 
    x,y = temp_gpu.shape
    power_mod.power(queue, (int(math.ceil(x*y/184000.0))*184000,1), (int(math.ceil(x*y/184000.0)),1), kernel, order,  np.int32(x*y)) 
    return kernel

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

    train = np.loadtxt('train_poly.csv', delimiter=',').astype(np.float32) 
    train_label = np.loadtxt('train_label_poly.csv').astype(np.int32) 
    test = np.loadtxt('test_poly.csv', delimiter=',').astype(np.float32) 
    test_label = np.loadtxt('test_label_poly.csv').astype(np.int32) 
        
    return train, train_label, test, test_label



def main():
    #load data
    train_data, train_label, test_data, test_label = load_data()    

    #Parameters
    c = np.float32(0.0001) #the penalty parameter 
    epochs = np.int32(500)
    num_iterations = np.int32(10)    
    positive_class = np.int32(1)     
    batch_size = np.int32(200)
    feature_size = np.int32(train_data.shape[0]) #the number of distances to the reference points
    test_size = np.int32(test_data.shape[0]) #the number of test examples
    train_size = feature_size #the number of train examples, which is obviously equal to the feature_size since feature size is the number of distances to the training points     
    print ("Started training SVM: ") 

    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    start = time.time()
    kernel_matrix = polynomial_kernel(gpuarray.to_gpu(train_data), gpuarray.to_gpu(train_data), np.int32(1), np.int32(1), np.int32(3), queue, mf) 
    end = time.time()
    delta = end-start
    print('Computing polynomial kernel took: %.2f sec' % delta)
    
    #transfer data from host to device
    label_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=train_label) 
    
    mean_accuracy=[]
    
    for i in range(num_iterations):
        #training
        w = svm_fit(kernel_matrix, label_g, feature_size, batch_size, epochs, c, train_size, queue, mf)
        #testing
        test_kernel_matrix = polynomial_kernel(gpuarray.to_gpu(test_data), gpuarray.to_gpu(train_data), np.int32(1), np.int32(1), np.int32(3), queue, mf) 
        k = np.zeros([test_data.shape[0], train_data.shape[0]]).astype(np.float32)
        cl.enqueue_copy(queue, k, test_kernel_matrix) 
        predictions = predict(k, w, test_size) 
        acc = accuracy(test_label, predictions)
        mean_accuracy.append(acc)
        print('Accuracy in the given iteration: %.4f' % acc)
    print('Mean accuracy: %.4f' % np.mean(np.array(mean_accuracy)))
    
    return 0

if __name__ == "__main__":
    main() 

