from src.get_data import inject_label_noise

class Trainer:

    def __init__(self,model,epoch_nb) -> None:
        self.model=model
        self.epoch_nb=epoch_nb
        self.train_sizes=[int(10**i) for i in range(2,5)]
        self.noise_ratios=[0.0,0.01,0.1]
        self.model_type=model.model_type

        pass

    def set_train_data(self,X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train
        self.model.set_train_data(X_train,y_train)
        pass

    def set_test_data(self,X_test,y_test):
        self.X_test=X_test
        self.y_test=y_test
        pass

    
    def train_epochs(self,epoch_nb=None):

        """
        Train the model for different number of epochs.
        """

        if epoch_nb is None:
            epoch_nb=self.epoch_nb

        if self.model_type!='optimizer':
            raise ValueError("train_epochs is only available for 'optimizer' model type.")

        metrics_dict={'rkhs_norm':[],
                      'test_mse':[],
                      'train_mse':[],
                      'accuracy':[],
                      'classification_error':[],
                      'var':[]

        }
        
        for _ in range(epoch_nb//5):
            self.model.fit_step(step_epochs=5)
            self._log_metrics(metrics_dict)
            metrics_dict['var'].append(len(metrics_dict['var'])*5)


        return metrics_dict

    def train_size(self,train_sizes=None):
        """
        Train the model with different training sizes.
        """
        if train_sizes is None:
            train_sizes=self.train_sizes

        metrics_dict={'rkhs_norm':[],
                      'test_mse':[],
                      'train_mse':[],
                      'accuracy':[],
                      'classification_error':[],
                      'var': train_sizes

        }

        for train_sz in train_sizes:
            new_train_X=self.X_train[:train_sz]
            new_train_y=self.y_train[:train_sz]
            self.model.set_train_data(new_train_X,new_train_y)
            self.model.fit()
            self._log_metrics(metrics_dict)

        return metrics_dict
    
    def train_noise(self,noise_ratios=None):
        """
        Train the model with different noise ratios.
        """
        if noise_ratios is None:
            noise_ratios=self.noise_ratios

        metrics_dict={'rkhs_norm':[],
                      'test_mse':[],
                      'train_mse':[],
                      'accuracy':[],
                      'classification_error':[],
                      'var':noise_ratios
        }

        for noise_rt in noise_ratios:

            noisy_y=inject_label_noise(self.y_train,noise_rt)
            self.model.set_train_data(self.X_train,noisy_y)
            self.model.fit()
            self._log_metrics(metrics_dict)

        return metrics_dict
    
    def _log_metrics(self,metrics_dict):
        """
        Log various metrics during training.

        """
        y_pred=self.model.predict(self.X_test)
        y_true=self.y_test

        rkhs_norm=self.model.compute_rkhs_norm()
        train_mse=self.model.compute_mse_loss(self.model.y_train,self.model.predict(self.model.X_train))
        test_mse=self.model.compute_mse_loss(y_true,y_pred)
        accuracy=self.model.compute_accuracy(y_true,y_pred)
        classification_error=self.model.compute_classification_error(y_true,y_pred)

        metrics_dict['rkhs_norm'].append(rkhs_norm)
        metrics_dict['test_mse'].append(test_mse)
        metrics_dict['train_mse'].append(train_mse)

        metrics_dict['accuracy'].append(accuracy)
        metrics_dict['classification_error'].append(classification_error)

        pass


    
