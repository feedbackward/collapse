'''Setup: Street View House Numbers (SVHN) dataset.'''

# External modules.
import numpy as np
import torchvision

# Internal modules.
from setup.directories import data_path
from setup.utils import do_normalization


###############################################################################


class SVHN:
    '''
    Prepare data from SVHN dataset.
    '''

    # Label dictionary.
    label_dict = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9"
    }
    
    def __init__(self, rg, tr_frac=0.8,
                 noise_frac=0.0, clean_test=True,
                 imbalance_factor=1.0, num_minority_classes=0,
                 label_dict=label_dict,
                 data_path=data_path, download=True):
        '''
        - rg : a Numpy random generator.
        '''
        
        print("--Preparing benchmark data (SVHN)--")

        # Get number of classes based on label dictionary.
        num_classes = len(label_dict)
        
        # Hang on to the generator.
        self.rg = rg
        
        # Hang on to the fraction to be used for training.
        self.tr_frac = tr_frac

        # Prepare the raw data (download if needed/desired).
        data_raw_tr = torchvision.datasets.SVHN(root=data_path,
                                                split="train",
                                                download=download,
                                                transform=None)
        data_raw_te = torchvision.datasets.SVHN(root=data_path,
                                                split="test",
                                                download=download,
                                                transform=None)
        
        # Extract raw data into a more convenient form.
        self.X = np.copy(data_raw_tr.data.astype(np.float32))
        self.Y = np.copy(data_raw_tr.labels.astype(np.uint8))
        self.X_te = np.copy(data_raw_te.data.astype(np.float32))
        self.Y_te = np.copy(data_raw_te.labels.astype(np.uint8))
        del data_raw_tr, data_raw_te

        # If desired, set up indices and sub-sample to get class imbalance.
        if imbalance_factor == 1.0:
            pass
        
        elif imbalance_factor > 1.0:
            
            class_counts = []
            for j in range(num_classes):
                class_counts += [int(np.sum(np.where(self.Y == j, 1.0, 0.0)))]
            class_counts = np.array(class_counts)
            n_max = np.max(class_counts)
            n_min = np.min(class_counts)
            if n_max != n_min:
                raise ValueError("We are assuming original data is balanced.")

            # First treat the simple case of "step imbalance".
            if num_minority_classes > 0:
                step_condition_1 = num_minority_classes >= 1
                step_condition_2 = num_minority_classes < num_classes
                step_condition = step_condition_1 and step_condition_2
                if not step_condition:
                    raise ValueError("The num_minority_classes is wrong.")
                
                # Discount factor (assuming original data is balanced).
                d_factor = 1.0 / imbalance_factor

                # New per-class sizes (step imbalance).
                class_counts_revised = []
                for j in range(num_minority_classes):
                    class_counts_revised += [int(n_max*d_factor)]
                for j in range(num_classes-num_minority_classes):
                    class_counts_revised += [n_max]
                class_counts_revised = np.array(class_counts_revised)
                
            else:
                # Discount factor (assuming original data is balanced).
                d_factor = (1.0/imbalance_factor)**(1.0/(num_classes-1.0))
                
                # New per-class sizes (long-tailed imbalance).
                class_counts_revised = []
                for j in range(num_classes):
                    class_counts_revised += [int(class_counts[j]*d_factor**j)]
                class_counts_revised = np.array(class_counts_revised)
            
            # Decision of "which class is which?" under imbalance is random.
            self.rg.shuffle(class_counts_revised) # in-place shuffling
            #print("class_counts_revised:", class_counts_revised)
            
            # Down-sample the dataset to make it imbalanced.
            X_list = []
            Y_list = []
            for j in range(num_classes):
                idx_class = self.Y == j
                if class_counts_revised[j] == n_max:
                    X_list += [np.copy(self.X[idx_class])]
                    Y_list += [np.copy(self.Y[idx_class])]
                elif class_counts_revised[j] > 1:
                    idx_sub = self.rg.choice(class_counts[j],
                                             size=class_counts_revised[j],
                                             replace=False)
                    X_list += [np.copy(self.X[idx_class][idx_sub])]
                    Y_list += [np.copy(self.Y[idx_class][idx_sub])]
                else:
                    raise ValueError("Bad value in class_counts_revised.")
            #print("shapes...")
            #print("X_list:")
            #for _X in X_list:
            #    print(_X.shape)
            #print("Y_list:")
            #for _Y in Y_list:
            #    print(_Y.shape)
            self.X = np.copy(np.vstack(X_list))
            self.Y = np.copy(np.concatenate(Y_list))
            del X_list, Y_list
        
        else:
            raise ValueError("The imbalance_factor is not valid.")
        
        # If desired, randomly flip training labels.
        num_noisy = int(len(self.Y)*noise_frac)
        flip_condition = num_noisy > 0 and num_noisy <= len(self.Y)
        if flip_condition:
            noisy_idx = self.rg.choice(len(self.Y), size=num_noisy,
                                       replace=False)
            noise = self.rg.integers(low=1, high=num_classes,
                                     size=(num_noisy,),
                                     dtype=np.uint8)
            self.Y[noisy_idx] = (self.Y[noisy_idx] + noise) % num_classes
        elif num_noisy == 0:
            pass
        else:
            raise ValueError("The noise_frac value is not valid.")

        # If desired, randomly flip test labels as well.
        num_noisy_te = 0 if clean_test else int(len(self.Y_te)*noise_frac)
        flip_condition_te = num_noisy_te > 0 and num_noisy <= len(self.Y_te)
        if flip_condition_te:
            noisy_idx = self.rg.choice(len(self.Y_te), size=num_noisy_te,
                                       replace=False)
            noise = self.rg.integers(low=1, high=num_classes,
                                     size=(num_noisy_te,),
                                     dtype=np.uint8)
            self.Y_te[noisy_idx] = (self.Y_te[noisy_idx] + noise) % num_classes

        # Normalize test inputs.
        self.X_te = do_normalization(X=self.X_te)
        
        return None
        
    
    def __call__(self):
        '''
        Each call gives us a chance to shuffle up the tr/va data.
        '''

        # Set the number of points to be used for training.
        n = len(self.X)
        self.n_tr = int(self.tr_frac*n)
        
        # Original index for the tr/va data.
        self.idx = np.arange(n)
        
        # Shuffle the tr/va data.
        self.rg.shuffle(self.idx)
        self.X = self.X[self.idx]
        self.Y = self.Y[self.idx]
        
        # Do the tr-va split.
        X_tr = self.X[0:self.n_tr]
        Y_tr = self.Y[0:self.n_tr]
        X_va = self.X[self.n_tr:]
        Y_va = self.Y[self.n_tr:]

        # Normalize the tr/va inputs.
        X_tr = np.copy(do_normalization(X=X_tr))
        X_va = np.copy(do_normalization(X=X_va))
        
        return (X_tr, Y_tr, X_va, Y_va, self.X_te, self.Y_te)
        

###############################################################################
