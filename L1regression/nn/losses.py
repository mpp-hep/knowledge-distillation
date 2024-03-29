import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def mse(y_true,y_pred):
    return K.mean( K.square(y_true[:,0] - y_pred[:,0]) )

def msle(y_true,y_pred):
    def mse_local(y_true,y_pred):
        return K.mean( K.square(y_true - y_pred) )
    return K.sqrt(mse_local(K.log(y_pred[:,0] + 1), K.log(y_true[:,0] + 1)))

def mae(y_true,y_pred):
    return K.mean( K.abs(y_true[:,0] - y_pred[:,0]))

def mape(y_true, y_pred):
    return K.mean( K.abs(y_true[:,0] - y_pred[:,0]) / K.clip(K.abs(y_true[:,0]),K.epsilon(),None))





class QuantileLoss(object):
    'Quantile Loss '
    def __init__(self,taus=[0.5],weights=[1.]):
        if isinstance(taus, float):
            taus = np.array([taus]) 
            weights = np.array([1.]) 
        self.taus = np.array(taus).reshape(1,-1)
        self.weights = np.array(weights).reshape(1,-1)           
        self.__name__ = 'QuantileLoss'
        
    def __call__(self,y_true,y_pred):
        e = y_true[:,0] - y_pred[:,0]
        return K.mean(self.weights*( self.taus*e + K.clip( -e, K.epsilon(), np.inf ) ))

class HuberLoss(object):
    'Huber Loss '
    def __init__(self,delta=1.):
        self.delta = delta
        self.n_params = 1
        self.__name__ = 'HuberLoss'
        
    def __call__(self,y_true,y_pred):
        z = K.abs(y_true[:,0] - y_pred[:,0])
        mask = K.cast(K.less(z,self.delta),K.floatx())
        return K.mean( 0.5*mask*K.square(z) + (1.-mask)*(self.delta*z - 0.5*self.delta**2) )


class DiceLoss(object):
    'Dice Loss '
    def __init__(self,epsilon=1.):
        self.__name__ = 'DiceLoss'
        self.epsilon = epsilon

    def __call__(self,y_true,y_pred):
        sum_p = tf.reduce_sum(tf.square(y_pred[:,0]))
        sum_t = tf.reduce_sum(tf.square(y_true[:,0]))

        union = sum_p + sum_t
        intersection = tf.reduce_sum(y_pred[:,0] * y_true[:,0])

        loss = union / (2.0 * intersection+self.epsilon)
        return loss    




class MseThesholdMetric(tf.keras.metrics.Metric):
    'MSE Theshold Metric '
    def __init__(self, threshold=0., **kwargs):
        super().__init__(**kwargs)
        self.res_mse = self.add_weight(name='res_mse', initializer='zeros')
        self.total_count = self.add_weight(name='total_count', initializer='zeros')
        self.threshold = threshold           

    def update_state(self, y_true, y_pred, sample_weight=None):
        where = K.cast(K.greater(y_true[:,1],self.threshold),K.floatx())
        res_mse = K.sum( K.square( where*(y_true[:,0] - y_pred[:,0])) )
        self.res_mse.assign_add(res_mse)
        self.total_count.assign_add(K.sum(where)+K.epsilon())

    def result(self):
        return self.res_mse/self.total_count

    def reset_state(self):
        self.res_mse.assign(0)
        self.total_count.assign(0)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold" : self.threshold})
        return config

    #def from_config(cls, config):
    #    return cls(**config)


def get_loss_func(str_name):
    loss_dictionary = {
        'mse':mse,
        'mae':mae,
        'mape':mape,
        'msle':msle
    }
    if str_name in loss_dictionary.keys():
        return loss_dictionary[str_name]
    else:
        parsed_name = str_name.split('_')
        if len(parsed_name)==1:
            return loss_dictionary[parsed_name[0]]
        else:
            if 'huber' in  parsed_name[0]:
                return HuberLoss(delta=float(parsed_name[1]))
            elif 'quantile' in  parsed_name[0]:
                return QuantileLoss(taus=list(float(parsed_name[1])))
            elif 'dice' in  parsed_name[0]:
                return DiceLoss(epsilon=float(parsed_name[1]))
            else :
                print('Loss function name is not recognized.')
                exit()



            