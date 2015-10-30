# H2O tutorial - anomaly detection
# 90% based on H2O 2014 2014 training book
# http://learn.h2o.ai/content/hands-on_training/anomaly_detection.html

if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

homedir <- "d:/H2O/"   # put your working directory
setwd(homedir)

download.file("http://h2o-release.s3.amazonaws.com/h2o/rel-slater/9/h2o-3.2.0.9.zip", 
              destfile = "h2o-3.2.0.9.zip")

unzip("h2o-3.2.0.9.zip")

install.packages(c("RCurl", 
                   "bitops", 
                   "rjson", 
                   "statmod", 
                   "tools"))

install.packages(paste0(homedir,
                        "/h2o-3.2.0.9/R/h2o_3.2.0.9.tar.gz"),
                 repos = NULL, type = "source")

library(h2o)
localH2O <- h2o.init(ip = "localhost", 
                     port = 54321, startH2O = TRUE)

localH2O = h2o.init(ip = "localhost", 
                    port = 54321, 
                    startH2O = TRUE, 
                    max_mem_size = '2g')

localH2O = h2o.init(ip = "localhost", 
                    port = 54321, 
                    startH2O = TRUE, 
                    max_mem_size = '2g',
                    nthreads = 3)

demo(h2o.glm)

homedir <- "d:/H2O/"   # put your working directory
setwd(homedir)

download.file("http://pjreddie.com/media/files/mnist_train.csv", 
              destfile = "mnist_train.csv")

download.file("http://pjreddie.com/media/files/mnist_test.csv", 
              destfile = "mnist_test.csv")
list.files()

TRAIN = "train.csv.gz"
TEST = "test.csv.gz"
train_hex <- h2o.importFile(localH2O, 
                            path = paste0(homedir,TRAIN), 
                            header = F, 
                            sep = ',') 
test_hex <- h2o.importFile(localH2O, path = paste0(homedir,TEST), header = F, sep = ',') 

class(train_hex)
summary(train_hex)

class(test_hex)
summary(test_hex)

quantile(train_hex$C785, probs = seq(0, 1, by = 0.01))

train_hex$C785 <- as.factor(train_hex$C785)
summary(train_hex)

test_hex$C785 <- as.factor(test_hex$C785)
summary(test_hex)

predictors = c(1:784)
resp = 785

train_hex <- train_hex[,-resp]
test_hex <- test_hex[,-resp]

ae_model <- h2o.deeplearning(x = predictors,
                             #y = 42, #response (ignored - pick any non-constant column)
                             training_frame = train_hex, #CAVEAT data is deprecated
                             activation = "Tanh",
                             autoencoder = T,
                             hidden = c(50),
                             ignore_const_cols = F,
                             epochs = 1)

summary(ae_model)
class(ae_model)

?h2o.anomaly

h2o.anomaly(ae_model, test_hex) 

test_rec_error <- as.data.frame(h2o.anomaly(ae_model, test_hex))

summary(test_rec_error)

?"h2o.deepfeatures"

test_features_deep <- h2o.deepfeatures(ae_model, 
                                       test_hex, 
                                       layer = 1)
summary(test_features_deep)
dim(test_hex)
dim(test_features_deep)

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

test_recon <- h2o.predict(ae_model, test_hex)
summary(test_recon)

plotDigits(test_recon, test_rec_error, c(1:25))
plotDigits(test_hex,   test_rec_error, c(1:25))

plotDigits(test_hex, test_rec_error, c(500:550))

plotDigits(test_recon, test_rec_error, c(4988:5012))
plotDigits(test_hex,   test_rec_error, c(4988:5012))

plotDigits(test_recon, test_rec_error, c(9976:10000))
plotDigits(test_hex,   test_rec_error, c(9976:10000))

