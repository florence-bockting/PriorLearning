import tensorflow as tf

from functools import partial


class energy_loss:
    def __init__(self):
        super(energy_loss, self).__init__()   
    
    def __call__(self, x, y, B, M, N, **kwargs):
        """
        Parameters
        ----------
        x : tensor, shape = (B,M)
            expert sample distribution
        y : tensor, shape = (B,N)
            model sample distribution
        B : int
            batch size
        M : int
            number of dimensions
        N : int
            number of dimensions
        """
        x = tf.expand_dims(tf.reshape(x, (B, M)), -1)
        y = tf.expand_dims(tf.reshape(y, (B, N)), -1)
        
        a = self.generate_weights(x, M)
        b = self.generate_weights(y, N)   
        
        return self.kernel_loss(x, y, a, b)
    
    def generate_weights(self, x, N):

        B, N, _ = x.shape
        weights = tf.divide(tf.ones(shape = (B, N), dtype = x.dtype), N) 
        return weights

    # (x*x - 2xy + y*y)
    def squared_distances(self, x, y):
        """
        Computes the squared distance between expert-elicited and model-implied
        statistics.
        
        .. math::
            \|x-y\|^2 = x^2 -2xy +y^2

        Parameters
        ----------
        x : tensor of shape (B,M)
            Expert elicited statistics.
        y : tensor of shape (B,N)
            Model-implied elicited statistics.

        Returns
        -------
        squared_distance : tensor of shape (B,N,M)
            Squared distance between expert elicited and model-implied statistics.

        """
        # (B, N, 1)
        D_xx = tf.expand_dims(tf.math.reduce_sum(tf.multiply(x, x), axis = -1), axis = 2)
        # (B, 1, M)
        D_yy = tf.expand_dims(tf.math.reduce_sum(tf.multiply(y, y), axis = -1), axis = 1)
        # (B, N, M)
        D_xy = tf.matmul(x, tf.transpose(y, perm = (0, 2, 1)))
        # compute sq. distance
        squared_distance = D_xx - 2*D_xy + D_yy
        
        return squared_distance

    # sqrt[(x*x - 2xy + y*y)]
    def distances(self, x, y):
        """
        Computes the distances (by taking the square-root of the squared 
        distances) and clip values to prevent under- or overflow.
        
        .. math::
            \|x-y\| = \\sqrt{\|x-y\|^2}
            
        Parameters
        ----------
        x : tensor of shape (B,M)
            Expert elicited statistics.
        y : tensor of shape (B,N)
            Model-implied elicited statistics.

        Returns
        -------
        distance : tensor of shape (B,N,M)
            distance between expert elicited and model-implied statistics.

        """
        distance = tf.math.sqrt(
            tf.clip_by_value(
                self.squared_distances(x,y), 
                clip_value_min = 1e-8, clip_value_max =  int(1e10)
                )
            )   
        
        return distance

    # - ||x-y|| = - sqrt[(x*x - 2xy + y*y)]
    def energy_kernel(self, x, y):
        """ 
        Computes the energy distance between expert-elicited and model-implied
        statistics by using the negative distance.
        
        .. math::
            k(x,y) = -\|x-y\|
            
        """
        energy_distance = - self.distances(x, y)    
        
        return energy_distance

    ## helper function
    def scal(self, a, f):
        B = a.shape[0]
        return tf.math.reduce_sum(tf.reshape(a, (B, -1)) * tf.reshape(f, (B, -1)), axis = 1)
      
    # k(x,y)=0.5*sum_i sum_j a_i a_j k(x_i,x_j) - sum_i sum_j a_i b_j k(x_i,y_j) + 0.5*sum_i sum_j b_i b_j k(y_i, y_j)
    def kernel_loss(self, x, y, a, b):
        """
        Computes the maximum mean discrepancy with energy kernel
        
        .. math::
            MMD^2(X,Y) = \\frac{1}{m(m-1)} k(x, x) - \\frac{2}{m^2} k(x, y) + \\frac{1}{m(m-1)} k(y, y)

        Parameters
        ----------
        x : tensor of shape (B,M)
            Expert elicited statistics.
        y : tensor of shape (B,N)
            Model-implied elicited statistics.
        a : TYPE
            DESCRIPTION.
        b : TYPE
            DESCRIPTION.

        Returns
        -------
        mean_loss : float
            Maximum mean discrepancy loss averaged over batches.

        """
        B, N, _ = x.shape
        
        K_xx = self.energy_kernel(x,x) # (B,N,N)
        K_yy = self.energy_kernel(y,y) # (B,M,M)
        K_xy = self.energy_kernel(x,y) # (B,N,M)
        
        # (B,N)
        a_x = tf.squeeze(tf.matmul(K_xx, tf.expand_dims(a, axis = -1)))
        # (B,M)
        b_y = tf.squeeze(tf.matmul(K_yy, tf.expand_dims(b, axis = -1)))
        # (B,N)   
        b_x = tf.squeeze(tf.matmul(K_xy, tf.expand_dims(b, axis = -1)))
        
        loss = 0.5 * self.scal(a, a_x) + 0.5 * self.scal(b, b_y) - self.scal(a, b_x)
        # average over batches
        mean_loss =  tf.reduce_mean(loss)
        
        return mean_loss
    
    
# class gaussian_loss: 
#     "Input has form (B, N, D)"
#     def __init__(self):
#         super(gaussian_loss, self).__init__()
        
#     def __call__(self, x, y, B, N, M):
#         x = tf.expand_dims(tf.reshape(x, (B, M)), -1)
#         y = tf.expand_dims(tf.reshape(y, (B, N)), -1)
                 
#         return(self.maximum_mean_discrepancy(x, y))

#     # (x*x - 2xy + y*y)
#     def squared_distances(self, x, y):
#         # (B, N, 1)
#         D_xx = tf.expand_dims(tf.math.reduce_sum(tf.multiply(x, x), axis = -1), axis = 2)
#         # (B, 1, M)
#         D_yy = tf.expand_dims(tf.math.reduce_sum(tf.multiply(y, y), axis = -1), axis = 1)
#         # (B, N, M)
#         D_xy = tf.matmul(x, tf.transpose(y, perm = (0, 2, 1)))
#         return D_xx - 2*D_xy + D_yy
    
#     def gaussian_kernel_matrix(self, x, y, sigmas):
#         # b = 1/2*sigma
#         beta = 1. / (2. * (tf.expand_dims(sigmas, axis=1)))
#         dist = self.squared_distances(x,y)
#         # 1/(2*sigma) * || x-y ||2
#         s = tf.matmul(beta, tf.reshape(dist, (1,-1)))
#         # exp(- 1/(2*sigma) * || x-y ||2 )
#         return tf.reshape(tf.reduce_sum(tf.exp(-s), axis=0), tf.shape(dist))
    
#     def mmd_kernel(self, x, y, kernel=gaussian_kernel_matrix):
#         # || x-y ||2 = xTx -2*xTy + yTy
#         loss = tf.reduce_mean(input_tensor = kernel(x, x), axis=(1,2))
#         loss += tf.reduce_mean(input_tensor = kernel(y, y), axis=(1,2))
#         loss -= 2 * tf.reduce_mean(input_tensor = kernel(x, y), axis=(1,2))
#         return loss
      
#     def maximum_mean_discrepancy(self,x, y, weight=1., minimum=0., **args):
#         sigmas = [
#             1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
#             1e3, 1e4, 1e5, 1e6
#         ]
#         gaussian_kernel = partial(self.gaussian_kernel_matrix, sigmas=sigmas)
#         loss_value = self.mmd_kernel(x, y, kernel=gaussian_kernel)
#         loss_value = tf.maximum(minimum, loss_value) * weight
#         return tf.reduce_mean(loss_value)