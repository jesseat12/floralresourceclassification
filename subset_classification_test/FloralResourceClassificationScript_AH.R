#####################################################################################################
# title         : Machine learning exercise for Sentinel-2 data
# purpose       : Implementing a machine learning workflow in R 
# author        : Abdulhakim M. Abdi (Twitter: @HakimAbdi / www.hakimabdi.com)
# input         : A multi-temporal raster stack of Sentinel-2 data comprising scenes from four dates 
# output        : One classified land cover map from each of three machine learning algorithms  
# Note 1        : This brief tutorial assumes that you are already well-grounded in R concepts and are 
#               : familiar with image classification procedure and terminology
# Reference		  : Please cite Abdi (2020): "Land cover and land use classification performance of machine learning 
#				        : algorithms in a boreal landscape using Sentinel-2 data" in GIScience & Remote Sensing if you find this 
#               : tutorial useful in a publication. 
# Reference URL	: https://doi.org/10.1080/15481603.2019.1650447 
#####################################################################################################

rm(list = ls(all.names = TRUE)) # will clear all objects, including hidden objects
gc() # free up memory and report memory usage


# set working directory. This will be the directory where all the unzipped files are located
#setwd("F:/subset_test_data/subset_classification/")
library(here)
here()
# load required libraries (Note: if these packages are not installed, then install them first and then load)
# rgdal: a comprehansive repository for handling spatial data
# raster: for the manipulation of raster data
# caret: for the machine learning algorithms
# sp: for the manipulation of spatial objects
# nnet: Artificial Neural Network
# randomForest: Random Forest 
# kernlab: Support Vector Machines
# e1071: provides miscellaneous functions requiered by the caret package



#### Load required packages
### Pay attention to how functions are called throughout the script - because some functions have the 
### same name across different packages. Functions are superseded or masked as packages are loaded
library(rgdal)
library(raster)
library(caret)
library(sp)
library(nnet)
library(randomForest)
library(kernlab)
library(e1071)
library(pacman)
library(mapview)
library(terra)
library(sf)
library(exactextractr)
library(readxl)
library(dplyr)
library(ModelMap)
library(RColorBrewer)

######## Data Exploration and preliminary geoprocessing


# Read in the training polygons 
training.vect<- vect(here("subset_classification_test//training_samples.shp")) # as Terra object 
training.polys<- st_read(here("subset_classification_test/training_samples.shp")) # as SF object for later use with exactextract
# Assign ID from row numbers
training.polys$Id<-row.names(training.polys)

# Read in the mask polygon to be used to clip/crop the orthophoto
Mask.clip<-vect(here("subset_classification_test//mask2clip.shp")) # as Terra
Mask.poly<-st_read(here("subset_classification_test//mask2clip.shp")) # as SF

# Reading in the orthophoto as a terra object
ortho.foto.terra<-rast(here("subset_classification_test/ortho.tif")) # terra object

# Reading in the DSM  as a terra object
DSM.terra<-rast(here("subset_classification_test/dsm.tif")) # terra object

# Reading in the DTM  as a terra object
DTM.terra<-rast(here("subset_classification_test/dtm.tif")) # terra object
# Select only the RGB bands, resample the orthophoto to the DSM spatial resolution, and then clip/crop
# the orthophoto using the mask
Ortho.foto.res<- terra::mask(resample(terra::subset(ortho.foto.terra, 1:3),DSM.terra, method = "near"),Mask.clip)
names(Ortho.foto.res)<-c("Red","Green","Blue") # rename the 3 channels / bands
# Do the same with the DSM + DTM before they are merged into one multi-band object
DSM.terra.crop<-terra::mask(DSM.terra, Mask.clip)
DTM.terra.crop<-terra::mask(DTM.terra, Mask.clip)

### Plotting
#par(mfrow = c(1,2))
#plotRGB(Ortho.foto.res, axes=TRUE, stretch="lin", main="Ortho Color Composite")
#plotRGB(ortho.foto.terra, axes=TRUE, stretch="lin", main="Original Ortho Color Composite")


#######################################################
#### Prepare functions to compute VIs
# EBI

EBI <- function(img, r,g,b) {
  br <- img[[r]]
  bg <- img[[g]]
  bb <- img[[b]]
  bright<-br+bg+bb
  gss<-((bg/bb)*(br-bb+255))
  vi <- bright / gss
  return(vi)
}

# VDVI
VDVI<- function(img, r,g,b){
  br <- img[[r]]
  bg <- img[[g]]
  bb <- img[[b]]
  vi <-((2*bg-br-bb)/(2*bg+br+bb))
  return(vi)
}

# VARI
VARI<- function(img, r,g,b){
  br <- img[[r]]
  bg <- img[[g]]
  bb <- img[[b]]
  vi <-((bg-br)/(bg+br-bb))
  return(vi)
}

# MGRVI

MGRVI<- function(img, r,g,b){
  br <- img[[r]]
  bg <- img[[g]]
  bb <- img[[b]]
  vi <-((bg^2-br^2)/(bg^2+br^2))
  return(vi)
}

# CIVE
CIVE<- function(img, r,g,b){
  br <- img[[r]]
  bg <- img[[g]]
  bb <- img[[b]]
  vi <-(0.441*br - 0.881*bg + 0.385*bb + 18.787)
  return(vi)
}


#######################################################
### Apply the indices #####
#Ortho.EBI.orig<-EBI(ortho.foto.terra, 1,2,3) # on the original image
Ortho.EBI<-EBI(Ortho.foto.res, 1,2,3) # on the resampled image
names(Ortho.EBI)<-c("EBI") # rename the recently created VI 
Ortho.CIVE<-CIVE(Ortho.foto.res, 1,2,3) # on the resampled image
names(Ortho.CIVE)<-c("CIVE") # rename the recently created VI 
Ortho.MGRVI<-MGRVI(Ortho.foto.res, 1,2,3) # on the resampled image
names(Ortho.MGRVI)<-c("MGRVI") # rename the recently created VI 
Ortho.VARI<-VARI(Ortho.foto.res, 1,2,3) # on the resampled image
names(Ortho.VARI)<-c("VARI") # rename the recently created VI 
Ortho.VDVI<-VDVI(Ortho.foto.res, 1,2,3) # on the resampled image
names(Ortho.VDVI)<-c("VDVI") # rename the recently created VI 
#######################################################

plot(Ortho.EBI, col=rev(terrain.colors(10)), main = "Ortho-EBI")

# view histogram of EBI
hist(Ortho.EBI,
     main = "Distribution of EBI values",
     xlab = "EBI",
     ylab= "Frequency",
     col = "wheat",
     xlim = c(-0.1, 2.3),
     breaks = 30,
     xaxt = 'n')
## Warning in .hist1(x, maxcell = maxcell, main = main, plot = plot, ...): 54%
## of the raster cells were used.
axis(side=1, at = seq(-0.1,2.3, 0.05), labels = seq(-0.1,2.33, 0.05))


mapview(as(Ortho.EBI,"Raster"))+mapview(training.polys,zcol="class_name")

writeRaster(Ortho.EBI, filename="Ortho_EBI_temp01.tif", overwrite=TRUE)
# Adding the DSM and DTM height and Vegetation Indices to the Ortho
height<-DSM.terra.crop-DTM.terra.crop
names(height)<-c("height") # rename the recently created VI 
Composite<-c(Ortho.foto.res, DSM.terra.crop, DTM.terra.crop, height,Ortho.EBI,Ortho.CIVE,
             Ortho.MGRVI, Ortho.VARI, Ortho.VDVI)

# Extract pixel values using exact extract
prec_dfs <- exact_extract(Composite, training.polys, include_xy=TRUE, include_cols=c('Id','class_name','class'))
tbl <- do.call(rbind, prec_dfs) # convert the previous list object to a dataframe

# Get the plant / features height
#tbl$height<-abs(tbl$dsm-tbl$dtm)

# A new dataset with just the variables we want to keep or rows... (i.e. filter out partial pixels)

tbl.trim<-tbl[,c(3:6,9:14 )]
tbl.trim$class<-as.factor(tbl.trim$class)
#######################################################
# Extract pixel values only for the original EBI
prec_dfs.orig <- exact_extract(Ortho.EBI.orig, training.polys, include_xy=TRUE, include_cols=c('Id','class_name','class'))
tbl.orig <- do.call(rbind, prec_dfs.orig)
#######################################################

# EBI distributions
boxplot(value~class_name,data=tbl.orig, main="EBI distribution per class - original RGB",
        xlab="Classes", ylab="EBI values")

# Height distributions
boxplot(height~class_name,data=tbl, main="Height distribution per class",
        xlab="Classes", ylab="Height (DSM - DTM)")


######################################################
### Fit a RandomForest Classifier

set.seed(123) # Run this before the sampling so that it's reproducible or do not run it if you want different results each time
split1<- sample(c(rep(1, 0.7 * nrow(tbl.trim)), rep(0, 0.3 * nrow(tbl.trim)))) # Create a vector to split into training ==1 and test ==0
table(split1)
tbl.train <- tbl.trim[split1 == 1, ]
tbl.test <-  tbl.trim[split1 == 0, ]
summary(as.factor(tbl$class))
summary(as.factor(tbl.train$class))
summary(as.factor(tbl.test$class))

#### Optimization
mtry <- tuneRF(tbl.trim[-1],tbl.trim$class, ntreeTry=500,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m)

set.seed(71)
rf <-randomForest(class~.,data=tbl.train, mtry=best.m, importance=TRUE,ntree=500)
print(rf)
#Evaluate variable importance
importance(rf)
varImpPlot(rf)

# Evaluate other models now that we've assessed variable importance
set.seed(71)
rf.simple <-randomForest(class~height+VDVI+MGRVI+Red+EBI,data=tbl.train, mtry=best.m, importance=TRUE,ntree=500)
# VDVI+MGRVI+Red+
print(rf.simple)
#Evaluate variable importance
importance(rf.simple)
varImpPlot(rf.simple)

# Apply the fitted model to the test dataset
prediction <-predict(rf.simple, tbl.test)
# Assess the accuracy of our model on a test dataset
confusionMatrix(prediction, tbl.test$class)


# Spatial prediction
Rast.Predict <- predict(Composite, rf.simple, type='response')

# Custom palette
my_palette <- brewer.pal(n = 6, name = "Dark2")
my_palette


# Plot
mapView(as(Rast.Predict, "Raster"), col.regions = my_palette)+mapview(as(Ortho.foto.res,"Raster"))
raster::spplot(Rast.Predict)


writeRaster(Rast.Predict, filename="Prediction_Test01312022.tif", overwrite=TRUE)




















# Load the sample data
# Alternatively, you can use the supplied orthophotos to generate a new set of training and validation data 
# Your samples layer must have a column for each image in the raster stack, a column for the land cover class that point represents, an X and Y column
# You can create such a sample file using QGIS or another GIS software
samples = read.csv("subset_classification_test/training_samples.csv")

# Extract ALL pixels that intersect the training polygons
# This needs to be updated to reflect the X,Y coordinates for each pixel
pixel.val<-data.frame(training.vect$class_name, terra::extract(x=Composite, y=training.vect, cells=TRUE, xy=TRUE))






#ex.df <- as.data.frame(extract(ortho.raster,training.polys,cellnumbers=T))

# create coordinate columns using xyFromCell
#ex.df.coords <- cbind(ex.df, xyFromCell(s,ex.df[,1]))

#######
# Split the data frame into 70-30 by class
trainx = list(0)
evalx = list(0)
for (i in 1:3){ # loop through all eight classes
  cls = samples[samples$class == i,]
  smpl <- floor(0.70 * nrow(cls))
  tt <- sample(seq_len(nrow(cls)), size = smpl)
  trainx[[i]] <- cls[tt,]
  evalx[[i]] <- cls[-tt,]
}

# combine them all into training and evaluation data frames
trn = do.call(rbind, trainx) 
eva = do.call(rbind, evalx)

# Set up a resampling method in the model training process
tc <- trainControl(method = "repeatedcv", # repeated cross-validation of the training data
                   number = 10, # number of folds
                   repeats = 5, # number of repeats
                   allowParallel = TRUE, # allow use of multiple cores if specified in training
                   verboseIter = TRUE) # view the training iterations
                        
# Generate a grid search of candidate hyper-parameter values for inclusion into the models training
# These hyper-parameter values are examples. You will need a more complex tuning process to achieve high accuracies
# For example, you can play around with the parameters to see which combinations gives you the highest accuracy. 
nnet.grid = expand.grid(size = seq(from = 2, to = 10, by = 2), # number of neurons units in the hidden layer 
                        decay = seq(from = 0.1, to = 0.5, by = 0.1)) # regularization parameter to avoid over-fitting 

rf.grid <- expand.grid(mtry=1:20) # number of variables available for splitting at each tree node

#svm.grid <- expand.grid(sigma=seq(from = 0.01, to = 0.10, by = 0.02), # controls for non-linearity in the hyperplane
                        #C=seq(from = 2, to = 10, by = 2)) # controls the influence of each support vector

## Begin training the models. It took my laptop 8 minutes to train all three algorithms
# Train the neural network model
nnet_model <- caret::train(x = trn[,(5:ncol(trn)-1)], y = as.factor(as.integer(as.factor(trn$class))),
                    method = "nnet", metric="Accuracy", trainControl = tc, tuneGrid = nnet.grid)

# Train the random forest model
rf_model <- caret::train(x = trn[,(5:ncol(trn)-1)], y = as.factor(as.integer(as.factor(trn$class))),
                    method = "rf", metric="Accuracy", trainControl = tc, tuneGrid = rf.grid)

# Train the support vector machines model
#svm_model <- caret::train(x = trn[,(5:ncol(trn)-1)], y = as.factor(as.integer(as.factor(trn$class))),
                    #method = "svmRadialSigma", metric="Accuracy", trainControl = tc, tuneGrid = svm.grid)

## Apply the models to data. It took my laptop 2 minutes to apply all three models
# Apply the neural network model to the Sentinel-2 data. 
nnet_prediction = raster::predict(s2data, model=nnet_model)

# Apply the random forest model to the Sentinel-2 data
rf_prediction = raster::predict(s2data, model=rf_model)

# Apply the support vector machines model to the Sentinel-2 data
#svm_prediction = raster::predict(s2data, model=svm_model)

# Convert the evaluation data into a spatial object using the X and Y coordinates and extract predicted values
eva.sp = SpatialPointsDataFrame(coords = cbind(eva$X, eva$Y), data = eva, 
                                proj4string = crs("+proj=utm +zone=12 +datum=WGS84 +units=m +no_defs"))
#eva.sp = SpatialPointsDataFrame(coords = cbind(eva$x, eva$y), data = eva, 
                                #proj4string = crs("+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0"))

## Superimpose evaluation points on the predicted classification and extract the values
# neural network
nnet_Eval = extract(nnet_prediction, eva.sp)
# random forest
rf_Eval = extract(rf_prediction, eva.sp)
# support vector machines
#svm_Eval = extract((svm_prediction), eva.sp)

# Create an error matrix for each of the classifiers
nnet_errorM = confusionMatrix(as.factor(nnet_Eval),as.factor(eva$class)) # nnet is a poor classifier, so it will not capture all the classes
rf_errorM = confusionMatrix(as.factor(rf_Eval),as.factor(eva$class))
#svm_errorM = confusionMatrix(as.factor(svm_Eval),as.factor(eva$class))
print(nnet_errorM)
print(rf_errorM)

# Plot the results next to one another
rstack = stack(nnet_prediction, rf_prediction) # combine the layers into one stack
names(rstack) = c("Single Layer Neural Network", "Random Forest") # name the stack
plot(rstack) # plot it!

# Plot the results next to one another along with the 2018 NMD dataset for comparison
#nmd2018 = raster("NMD_S2Small.tif") # load NMD dataset (Nationella Marktaeckedata, Swedish National Land Cover Dataset)
#crs(nmd2018) <- crs(nnet_prediction) # Correct the coordinate reference system so it matches with the rest
#rstack = stack(nmd2018, nnet_prediction, rf_prediction, svm_prediction) # combine the layers into one stack
#names(rstack) = c("NMD 2018", "Single Layer Neural Network", "Random Forest", "Support Vector Machines") # name the stack
#plot(rstack) # plot it! 

# Congratulations! You conducted your first machine learning classification in R. 
# Please cite the paper referred to at the beginning if you use any part of this script in a publication. Thank you! :-)