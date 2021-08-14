import h5py
import numpy as np

# read the training set
with h5py.File('datasets/train_catvnoncat.h5', "r") as hdf_t :
    keys = list(hdf_t.keys())
    # keys are list_classes , train_set_x, train_set_y
    classes = np.array(hdf_t.get('list_classes')[:])
    train_x = np.array(hdf_t.get('train_set_x')[:])
    train_y = np.array(hdf_t.get('train_set_y')[:])

# read test set
with h5py.File('datasets/test_catvnoncat.h5', "r") as hdf_test:
    keys_t = list(hdf_test.keys())
    # keys are list_classes , test_set_x , test_set_y
    test_x = np.array(hdf_test.get('test_set_x')[:])
    test_y = np.array(hdf_test.get('test_set_y')[:])

def get_raw():
    return train_x,train_y,test_x,test_y


def load_data():
    #Transform training set
    x_train = train_x.reshape((train_x.shape[0],-1)).T
    y_train = train_y.reshape((1,train_y.shape[0]))
    
    #Transform test set
    x_test = test_x.reshape((test_x.shape[0],-1)).T
    y_test = test_y.reshape((1,test_y.shape[0]))

    # assertion : check if 1st 8 pixels of 2nd image are in correct place
    assert np.all([[196,192,190,193,186,182,188,179] , x_train[:8,1]]),"Wrong transformation for train_x"
    assert np.all([[115,110,111,137,129,129,155,146] , x_test[:8,1]]),"Wrong transformation for test_x"

    return x_train,y_train,x_test,y_test,classes

