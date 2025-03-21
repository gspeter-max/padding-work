'''
🔥 THURSDAY (March 21) – DEEP LEARNING SPEED BOOST

📌 Task: Train a ResNet-like model with 2x speed improvement using:
🎯 Subtasks:

    Implement XLA Compilation in TensorFlow.
    Use Mixed Precision Training (FP16).
    Optimize data pipeline (TFRecord + prefetching).
    Test improvement on CIFAR-10.
    🛑 Hard Mode: Implement from scratch in NumPy (no TF/PyTorch).
''' 
import numpy as np

class restnet_classificaiton:

    def __init__(self,input_shape, kernel: (np.ndarray, list) = None):
        if kernel is None:
            self.kernel = np.array([[
                [-1,0,1],
                [-1,0,1],
                [-1,0,1]
            ],
               [[-1,0,1],
                [-1,0,1],
                [-1,0,1]],
            
            [[-1,0,1],
                [-1,0,1],
                [-1,0,1]]]
    )
        if isinstance(self.kernel ,list):
            temp_kernel = iter(self.kernel)
            length = len(next(temp_kernel))
            self.kernel_size = length

        else :
            self.kernel_size = self.kernel.shape[1]

        self.input_shape = input_shape
    def Batchnormalization(self, input): 

        ''' 
        1st compute the mean and  std ( for each filter  becuase batchnormalization mean each filter and layer means each sample )
        2st use normalization formula ( z_score, min-max)
        and  do that 
        ''' 
        batch_mean = np.mean(input , axis = 0 )

    def conv2D(self,input = x_train , filter= 30 , stride = 1, padding = 'valid'):

        final_features = []
        if padding == "same":
            if stride != 1 :
                raise ValueError('make sure stride 1 for padding = "same"')

            padding = ((self.kernel_size - 1 )// 2 )
            output_size = (self.input_shape[1] - 2 * padding - self.kernel_size ) // stride + 1 

        else:
            
            output_size = int(((self.input_shape[1] - self.kernel_size + 1 )) // stride)

        for _ in range(filter):

            each_filter_featuers = np.zeros((self.input_shape[0],output_size, output_size))

            for i in range(output_size ):

                for j in range(output_size):
                    # i am mulitply that stride make sure you know 
                    # do that for all shapes 
                    array = input[:,j * stride : j * stride + self.kernel_size, i * stride : i* stride + self.kernel_size, :]
                    values = array * self.kernel 
                    # compute the sum for axis = -1 ( last axis )\
                    v = np.sum(values,-1)
                    each_filter_featuers[:,j,i] = np.sum(v, axis = (1,2)) 


            final_features.append(each_filter_featuers)
        print(final_features)
        '''
        # that each filter take a unique information about that image
        # you need to call function
        self.Batchnormalization()
        self.relu()
          '''
    # i that is restnet-30
    def residual_block(self, filter_size, kernel_size, stride, padding ):

        '''        self.conv2D(filter_size, kernel_size, stride, padding)
        self.Batchnormalization()
        self.relu()

        self.conv2D(filter_size, kernel_size, stride, padding)
        self.Batchnormalization()
        # make sure you do that skip connection

        self.relu()
            '''
        pass

    def final_layer(self,ouput_size):

        '''self.globals_avg_pooling()
        self.Dense(ouput_size)
        self.softmax()
        '''
        pass

y = restnet_classificaiton()
y.conv2D()
