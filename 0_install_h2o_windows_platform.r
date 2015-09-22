
# The following two commands remove any previously installed H2O packages for R.
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

# Next, you must download last H20 version from
# http://h2o-release.s3.amazonaws.com/h2o/rel-slater/1/index.html?aliId=78076
# just in case follow
# http://h2o-release.s3.amazonaws.com/h2o/rel-lambert/5/docs-website/Ruser/Rinstall.html

# and install from source
install.packages("D:/H2O/h2o-3.2.0.1/R/h2o_3.2.0.1.tar.gz",
                 repos = NULL, type = "source")
library(h2o)
localH2O = h2o.init()

# First steps with H2O
# from http://blenditbayes.blogspot.com.es/2014/07/things-to-try-after-user-part-1-deep.html

## Start a local cluster with 1GB RAM (default)
library(h2o)
localH2O <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE)

## Start a local cluster with 2GB RAM
localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, 
                    max_mem_size = '2g')

# By default, H2O starts a cluster using all available threads (8 in my case). 
# The h2o.init(...) function has no argument for limiting the number of threads yet 
# (well, sometimes you do want to leave one thread idle for other important tasks like Facebook). 
# But it is not really a problem.

# This is what this guy says, but it is always wise to keep one free core just in case process does not end
# So I use nthreads = 3

localH2O = h2o.init(ip = "localhost", 
                    port = 54321, 
                    startH2O = TRUE, 
                    max_mem_size = '2g',
                    nthreads = 3)

# Finally, let's run a demo to see H2O at work.
demo(h2o.glm)
# demo is not particularly well described, but at least you have proof h2o is working

# Basic operation: including a dataframe in h2o

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
