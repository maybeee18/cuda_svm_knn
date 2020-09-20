import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import random
import math
import time
import skcuda
import skcuda.linalg as culinalg

#CUDA KERNEL TO MAKE PREDICTIONS
predict_array_mod = SourceModule("""
__global__ void predict_array(float * kArray, float *w, int * d_y_pred, unsigned int feature_size, unsigned int num) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    
    if (i<num){
    float dot_product = 0;
    
    for(int j = 0; j < feature_size; j++) {
        dot_product += w[j]*kArray[i*feature_size + j];
    }

    if (dot_product >= 0) {
        d_y_pred[i] = 1;
    }

    else d_y_pred[i] = -1;
}
}

""")



#CUDA KERNELS FOR TRAINING SVM
copy_batch_mod = SourceModule("""
__global__ void copy_batch(float * kArray, float *xi, unsigned int idx, unsigned int feature_size, unsigned int batch_size) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i < feature_size*batch_size) {
        xi[i] = kArray[idx*feature_size+i];
    }
}


""")



copy_array_mod = SourceModule(""" 
__global__ void copy_array(float * a, float * b, int feature_size) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i < feature_size) {
        a[i] = b[i];
    }    
}
""")


select_samples_mod = SourceModule(""" 
__global__ void select_samples(float * xi, float * w, unsigned int feature_size, unsigned int idx, int * label, unsigned int batch_size) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
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
""")



reduce_by_samples_mod = SourceModule(""" 
__global__ void reduce_by_samples(float *yi_xi, float *xi, unsigned int batch_size, unsigned int feature_size, unsigned int idx, int * label) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
 //   yi_xi[i] = 100;
    
    if(i < feature_size) {
        float sum = 0;
    
        for(int j = 0; j < batch_size; j++) {
            sum += xi[j*feature_size + i]*label[idx+j]; 
        }
        printf("thread id %d", i);
        printf("sum %d", sum);
        yi_xi[i] = sum;
    }    
}
""")


update_w_mod = SourceModule(""" 

__global__ void update_w(float *yi_xi, float *w, float *next_w, float nt, float c, int batch_size, unsigned int feature_size) {

    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i < feature_size) {
        next_w[i] = w[i] - nt*c*w[i] + (nt/batch_size)*yi_xi[i];
    } 
}
""")


#CUDA KERNEL USED IN POLYNOMIAL KERNEL CALCULATION
power_mod = SourceModule("""
__global__ void power(float * array, int order, int num){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<num){
    array[i]= pow(array[i], order);
}
}


""")


#POLYNOMIAL KERNEL MATRIX COMPUTATION FUNCTION
def polynomial_kernel(x,y, gamma, coeff, order):
    culinalg.init()
    temp_gpu = gamma*skcuda.linalg.dot(x, skcuda.linalg.transpose(y)) + coeff

    p = power_mod.get_function("power")
    x,y = temp_gpu.shape
    grid_x = x*y / 32
    if (x*y % 32 != 0):
        grid_x += 1    
    p(temp_gpu, np.int32(order), np.int32(x*y), block=(32,1,1), grid=(int(grid_x),1,1))
    return temp_gpu



#FIT FUNCTION
def svm_fit(data, label, feature_size, batch_size, epochs, c, train_size):
   
    #ALLOCATE MEMORY FOR WEIGHTS AND PREDICTIONS
    w_np = np.zeros([1,feature_size]).astype(np.float32)
    w = drv.mem_alloc(w_np.nbytes) 
    drv.memcpy_htod(w, w_np)
    
    arr = np.ones([batch_size,feature_size]).astype(np.float32)
    xi = drv.mem_alloc(arr.nbytes)

    yi_xi_np = np.ones([1,feature_size]).astype(np.float32)
    yi_xi = drv.mem_alloc(yi_xi_np.nbytes)
   
    next_w_np = np.zeros([1,feature_size]).astype(np.float32)
    next_w = drv.mem_alloc(next_w_np.nbytes)
    

   
    #TRAINING
    for t in range(1, epochs) :

        drv.memcpy_htod(next_w, next_w_np)
        
        nt = 1/(c*t)
        
        idx = random.randint(1,100000) % (train_size - batch_size)
  
        copy_batch = copy_batch_mod.get_function("copy_batch")
        copy_batch(data, xi, np.int32(idx), feature_size, batch_size, block=(int(math.ceil(batch_size*feature_size/63000.0)),1,1), grid=(63000,1,1))
        drv.memcpy_dtoh(arr, xi)

        
        select_samples = select_samples_mod.get_function("select_samples")
        select_samples(xi, w, feature_size, np.int32(idx), label.gpudata, batch_size, block=(int(math.ceil(batch_size/512.0)),1,1), grid=(512,1,1))
        drv.memcpy_dtoh(arr, xi)

        
        reduce_by_samples = reduce_by_samples_mod.get_function("reduce_by_samples")
        reduce_by_samples(yi_xi, xi, batch_size, feature_size, np.int32(idx), label.gpudata, block=(int(math.ceil(feature_size/512.0)),1,1), grid=(512,1,1))
        drv.memcpy_dtoh(yi_xi_np, yi_xi)

        
        update_w = update_w_mod.get_function("update_w")
        update_w(yi_xi, w, next_w, np.float32(nt), c, batch_size, feature_size, block=(int(math.ceil(feature_size/512.0)),1,1), grid = (512,1,1))

        
        copy_array = copy_array_mod.get_function("copy_array")
        copy_array(w, next_w, feature_size, block=(int(math.ceil(feature_size/512.0)),1,1), grid = (512,1,1))


    return w 


#PREDICT LABELS
def predict(data,feature_size, w, num):

    y_pred = np.zeros([1, num]).astype(np.int32) 
    d_predicted_labels = drv.mem_alloc(y_pred.nbytes)

    predict_array = predict_array_mod.get_function("predict_array")
    predict_array(data, w, d_predicted_labels, feature_size, num, block = (math.ceil(num/1875.0),1,1), grid = (1875,1,1))
        
    drv.memcpy_dtoh(y_pred, d_predicted_labels)
    
    return y_pred   


#CALCULATE ACCURACY
def accuracy (label, pred_label):
    corr = 0
    for i in range(pred_label.shape[1]):
        if (label[i] == pred_label[0][i]):
            corr +=1
    return corr/label.shape[0]


#LOAD DATA
def load_data():

    train = np.loadtxt('train_poly.csv', delimiter=',').astype(np.float32) 
    train_label = np.loadtxt('train_label_poly.csv').astype(np.int32) 
    test = np.loadtxt('test_poly.csv', delimiter=',').astype(np.float32) 
    test_label = np.loadtxt('test_label_poly.csv').astype(np.int32)  
      
    return train, train_label, test, test_label


#MAIN FUNCTION
def main():
       
    #load data
    train_data, train_label, test_data, test_label = load_data()
   

    #Parameters
    c = np.float32(0.0001) #the penalty parameter 
    epochs = np.int32(500)
    num_iterations = np.int32(5)    
    positive_class = np.int32(1)     
    batch_size = np.int32(200)
    feature_size = np.int32(train_data.shape[0]) #the number of distances to the reference points
    test_size = np.int32(test_data.shape[0]) #the number of test examples
    train_size = feature_size #the number of train examples, which is obviously equal to the feature_size since feature size is the number of distances to the training points     
    print ("Started training SVM: ")

    start = time.time()
    
    #calculate polynomial kernel matrix
    kernel_matrix = polynomial_kernel(gpuarray.to_gpu(train_data), gpuarray.to_gpu(train_data), np.int32(1), np.int32(1), np.int32(1))

    end = time.time()
    delta = end-start
    print('Computing polynomial kernel took: %.2f sec' % delta)
    
    mean_accuracy = [] #list to store accuracy values 
    
    for i in range(num_iterations):
        start = time.time()
        w = svm_fit(kernel_matrix, gpuarray.to_gpu(train_label), feature_size, batch_size, epochs, c, train_size)
        end = time.time()
        delta = end-start
        print('Training took: %.2f sec' % delta)
        
        #calculate polynomial kernel matrix
        test_kernel_matrix = polynomial_kernel(gpuarray.to_gpu(test_data), gpuarray.to_gpu(train_data), np.int32(1), np.int32(1), np.int32(1))
            
        #make predictions
        pred_label = predict(test_kernel_matrix, feature_size, w, np.int32(test_data.shape[0]))
        acc = accuracy(test_label, pred_label)
        mean_accuracy.append(acc)
        print('Accuracy in the given iteration: %.4f' % acc)
    print('Mean accuracy: %.4f' % np.mean(np.array(mean_accuracy))) 
    
    return 0



if __name__ == "__main__":
    main()
