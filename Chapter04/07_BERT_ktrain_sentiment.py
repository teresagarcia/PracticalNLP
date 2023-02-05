import tarfile
import ktrain
from ktrain import text

import os
if not os.path.exists( os.getcwd() + "/data/Chapter04/aclImdb") :
    import tensorflow as tf
    dataset_path = os.getcwd() + "/data/Chapter04/aclImdb.tar.gz"
    dataset = tf.keras.utils.get_file(
        fname= dataset_path, 
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
        extract=True,
    )
    with tarfile.open(dataset_path) as file:
        file.extractall(os.getcwd() + "/data/Chapter04/")
    # set path to dataset
IMDB_DATADIR= os.getcwd() + "/data/Chapter04/aclImdb"

  
(x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder(IMDB_DATADIR, 
                                                                       maxlen=500, 
                                                                       preprocess_mode='bert',
                                                                       train_test_names=['train', 
                                                                                         'test'],
                                                                       classes=['pos', 'neg'])

model = text.text_classifier('bert', (x_train, y_train), preproc=preproc)
learner = ktrain.get_learner(model,train_data=(x_train, y_train), val_data=(x_test, y_test), batch_size=6)
learner.fit_onecycle(2e-5, 4)
                                                                      