# Shamelessly copied (and maybe retouched) from
# http://learn.h2o.ai/content/hands-on_training/deep_learning.html

#Classification and Regression with H2O Deep Learning
#This tutorial shows how a Deep Learning model can be used to do supervised classification and regression. 

# The h2o.deeplearning function fits H2O's Deep Learning models from within R.

library(h2o)
args(h2o.deeplearning)
?"h2o.deeplearning"

#there are a lot of parameters! 
#Luckily, as you'll see later, you only need to know a few to get the most out of Deep Learning. 
#More information can be found in the H2O Deep Learning booklet and in our slides.

# You can run the example from the man page using the example function:
  
example(h2o.deeplearning)
# With the always present iris dataset

# Start H2O and load the MNIST data

#For the rest of this tutorial, we will use the well-known MNIST dataset of hand-written digits, 
#where each row contains the 28^2=784 raw gray-scale pixel values 
#from 0 to 255 of the digitized digits (0 to 9).

#Initialize the H2O server and import the MNIST training/testing datasets.

library(h2o)

# Remember: from installation we defined a particular cluster

localH2O = h2o.init(ip = "localhost", 
                    port = 54321, 
                    startH2O = TRUE, 
                    max_mem_size = '2g',
                    nthreads = 3)

homedir <- "d:/H2O/"   # Set you own working folder
setwd(homedir)
list.files()
TRAIN = "train.csv.gz"
TEST = "test.csv.gz"
train_hex <- h2o.importFile(localH2O, path = paste0(homedir,TRAIN), header = F, sep = ',') #, key = 'train.hex')
test_hex <- h2o.importFile(localH2O, path = paste0(homedir,TEST), header = F, sep = ',') #, key = 'test.hex')

class(train_hex)
summary(train_hex)

class(test_hex)
summary(test_hex)

# You can access individual columns of the h2o object as with any other R object

quantile(train_hex$C785, probs = seq(0, 1, by = 0.01))

#  data consists of 784 (=28^2) pixel values per row, with (gray-scale) values from 0 to 255. 
# The last column is the response (a label in 0,1,2,...,9).

# and since this is a classification problem we must convert C785 the class into a factor

train_hex$C785 <- as.factor(train_hex$C785)
summary(train_hex)

#same with test
test_hex$C785 <- as.factor(test_hex$C785)
summary(test_hex)


#While H2O Deep Learning has many parameters, it was designed to be just as easy to use 
#as the other supervised training methods in H2O. 
#Automatic data standardization and handling of categorical variables and missing values 
#and per-neuron adaptive learning rates reduce the amount of parameters the user has to specify. 
#Often, it's just the number and sizes of hidden layers, 
#the number of epochs and the activation function and maybe some regularization techniques.

dlmodel <- h2o.deeplearning(x = 1:784, 
                            y = 785, 
                            training_frame = train_hex, 
                            validation = test_hex,
                            hidden = c(50, 50), 
                            epochs = 0.1, 
                            activation = "Tanh")

# There are constant columns in this dataset and you get a warning accordingly 
# but it does not stop
# Warning message:
#  In .h2o.validateModelParameters(conn, algo, param_values, h2oRestApiVersion) :
#  Dropping constant columns: C1, (...)

summary(dlmodel)

# Let's look at the model summary, and the confusion matrix and classification error 
# (on the validation set, since it was provided) in particular:

# dlmodel is a very specific object of h2o library

class(dlmodel)
# Correctly declaring a "multinomial model"
str(dlmodel)

dlmodel@model

dlmodel@model$training_metrics    #confusion matrix training dataset
dlmodel@model$validation_metrics  #confusion matrix validation dataset

#To confirm that the reported confusion matrix on the validation set (here, the test set) 
# was correct, we make a prediction on the test set and compare the confusion matrices explicitly:

# This is old code but still useful
# For generating predictions from our model
pred_labels <- h2o.predict(dlmodel, test_hex)[,1]
head(pred_labels)
# To compare with real numbers
actual_labels <- test_hex[, 785]
head(actual_labels)

# But modern h2o uses the model directly to extract the confusion matrix with test data
cm <- h2o.confusionMatrix(dlmodel, test_hex)
cm

# To see the model parameters DOES NOT WORK AND CANNOT FIND HOW THIS CHANGED

dlmodel@model$params

# Follow this guide to find changed syntax
# https://github.com/h2oai/h2o-3/blob/master/h2o-docs/src/product/upgrade/H2ODevPortingRScripts.md

dlmodel@allparameters

# Hyper-parameter Tuning with Grid Search
# Since there are a lot of parameters that can impact model accuracy, 
# hyper-parameter tuning is especially important for Deep Learning:

# grid_search is *NOT* supported yet in h2o 3
# search for grid_search in
# https://github.com/h2oai/h2o-3/blob/master/h2o-docs/src/product/upgrade/H2ODevPortingRScripts.md

