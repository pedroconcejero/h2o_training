
# NOW LET'S GO TO MNIST DATA
# MNIST train set in csv from here
# http://pjreddie.com/projects/mnist-in-csv/

# BTW csv.gz versions of these files can be found here
# https://github.com/h2oai/h2o-2/tree/master/smalldata/mnist

# Following closely anomaly detection tutorial here
# http://learn.h2o.ai/content/hands-on_training/anomaly_detection.html

# CAVEAT I am using the localH2O cluster defined before *NOT* the one in the tutorial
# CAVEAT key is not defined in this h2o version
# see ?"h2o.importFile"


homedir <- "d:/H2O/"
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

#  data consists of 784 (=28^2) pixel values per row, with (gray-scale) values from 0 to 255. 
# The last column is the response (a label in 0,1,2,...,9).

predictors = c(1:784)
resp = 785

# We  do unsupervised training, so we can drop the response column.

train_hex <- train_hex[,-resp]
test_hex <- test_hex[,-resp]

# Finding outliers - ugly hand-written digits
#We train a Deep Learning Auto-Encoder to learn a compressed 
#(low-dimensional) non-linear representation of the dataset, 
#hence learning the intrinsic structure of the training dataset. 
#The auto-encoder model is then used to transform all test set images to their reconstructed images, 
#by passing through the lower-dimensional neural network. 
#We then find outliers in a test dataset by comparing the reconstruction of 
#each scanned digit with its original pixel values. 
#The idea is that a high reconstruction error of a digit 
#indicates that the test set point doesn't conform to the structure of the training data 
#and can hence be called an outlier.

#1  Learn what's normal from the training data

#Train unsupervised Deep Learning autoencoder model on the training dataset. 
#For simplicity, we train a model with 1 hidden layer of 50 Tanh neurons to create 
#50 non-linear features with which to reconstruct the original dataset. 
#We learned from the Dimensionality Reduction tutorial that 50 is a reasonable choice. 
#For simplicity, we train the auto-encoder for only 1 epoch (one pass over the data). 
#We explicitly include constant columns (all white background) for the visualization to be easier.

ae_model <- h2o.deeplearning(x = predictors,
                             #y = 42, #response (ignored - pick any non-constant column)
                             training_frame = train_hex, #CAVEAT data is deprecated
                             activation = "Tanh",
                             autoencoder = T,
                             hidden = c(50),
                             ignore_const_cols = F,
                             epochs = 1)

# Note that the response column is ignored 
# Actually if you keep it you'll get an error message pointint to autoencoder = T

summary(ae_model)
class(ae_model)

#2 Find outliers in the test data

# The Anomaly app computes the per-row reconstruction error for the test data set. 
# It passes it through the autoencoder model (built on the training data) and 
#computes mean square error (MSE) for each row in the test set.

?h2o.anomaly

h2o.anomaly(ae_model, test_hex) #CAVEAT THE ORDER!!! first the h2oautoencodermodel object
test_rec_error <- as.data.frame(h2o.anomaly(ae_model, test_hex))

summary(test_rec_error)

#In case you wanted to see the lower-dimensional features created by 
#the auto-encoder deep learning model, here's a way to extract them for a given dataset. 
#This a non-linear dimensionality reduction, similar to PCA, 
#but the values are capped by the activation function (in this case, they range from -1...1)

?"h2o.deepfeatures"

test_features_deep <- h2o.deepfeatures(ae_model, test_hex, layer = 1)#CAVEAT THE ORDER!!!
summary(test_features_deep)

#3. Visualize the good, the bad and the ugly

# We will need a helper function for plotting handwritten digits 
#(adapted from http://www.r-bloggers.com/the-essence-of-a-handwritten-digit/). 
#Don't worry if you don't follow this code...

plotDigit <- function(mydata, rec_error) {
  len <- nrow(mydata)
  N <- ceiling(sqrt(len))
  op <- par(mfrow = c(N,N), 
            pty = 's',
            mar = c(1,1,1,1),
            xaxt ='n',
            yaxt ='n')
  for (i in 1:nrow(mydata)) {
    colors <- c('white','black')
    cus_col <- colorRampPalette(colors = colors)
    z <- array(mydata[i,],
               dim=c(28,28))
    z <- z[,28:1]
    image(1:28, 1:28, z,
          main = paste0("rec_error: ", round(rec_error[i],4)), col = cus_col(256))
  }
  on.exit(par(op))
}

plotDigits <- function(data, rec_error, rows) {
  row_idx <- order(rec_error[,1],
                   decreasing = F)[rows]
  my_rec_error <- rec_error[row_idx,]
  my_data <- as.matrix(as.data.frame(data[row_idx,]))
  plotDigit(my_data, my_rec_error)
}


#Let's look at the test set points with low/median/high reconstruction errors. 
# We will now visualize the original test set points and their reconstructions obtained by 
#propagating them through the narrow neural net.

test_recon <- h2o.predict(ae_model, test_hex)
summary(test_recon)

#The good

# Let's plot the 25 digits with lowest reconstruction error. 
#First we plot the reconstruction, then the original scanned images.

plotDigits(test_recon, test_rec_error, c(1:25))
plotDigits(test_hex,   test_rec_error, c(1:25))

#The bad

# Now let's look at the 25 digits with median reconstruction error.

plotDigits(test_recon, test_rec_error, c(4988:5012))
plotDigits(test_hex,   test_rec_error, c(4988:5012))

# The ugly

#And here are the biggest outliers - The 25 digits with highest reconstruction error!
  
plotDigits(test_recon, test_rec_error, c(9976:10000))
plotDigits(test_hex,   test_rec_error, c(9976:10000))
