#####################################################################################################
# USDA / ARS-FRRL-PIBR Logan Utah
# Floral resources mapping:
# Authors: Alexander Hernandez alexander.hernandez@usda.gov / Jesse Tabor
# * Random Forest model fit to classify RGB imagery into landscape features (flowers, shrubs, soil)
# * Pre-processing: Uses Geospatial Data Abstraction Library GDAL tools
# * AOI extracted from original imagery is ~ 10 acres in size and its spatial domain contains the 
# field data transect
# * RGB imagery processing (vegetation indices + texture-based covariates) uses raster::raster objects
# * Vegetations indices are calculated using a general customized function
# * Texture-based metrics are obtained using glcm package https://github.com/azvoleff/glcm
# * raster::stack objects pixel values extractions are obtained using exactextractr package 
#  https://github.com/isciences/exactextractr
# * Model is fit using 70% of observations for training - 30% for validation
# * Full RF is initially fit and then parsimonious model is identified by assessing variable importance metrics
# * Geospatial predictions are obtained using raster::predict that solely uses raster:stack objects
#####################################################################################################

rm(list = ls(all.names = TRUE)) # will clear all objects, including hidden objects
gc() # free up memory and report memory usage


# set working directory. This will be the directory where all the unzipped files are located
setwd("/media/geospastore/shared/Floral/in_data/apiary")
#setwd("~/Desktop/Floral_Resources/subset_classification_test")
library(here)
here()


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
library(imager)
library(GLCMTextures)
library(gdalUtils)
library(glcm)

################################### GDAL-based pre processing #####################
## This is only needed to crop the full mosaic using the Area of Interest AOI -
## the 10-acre AOI around the field transect location, and then to extract only the 3 bands 
## of interest and convert the RGB to a pseudo-color 8-Bit image that is the primary input
## for the texture-based metrics

# first get desired output resolution (pixel size) for the orthophoto that requires cropping
# In this case we can use the DTM or DSM
x.out<-terra::res(rast(here("34_twincreek_apiary_t1/dsm.tif")))


# This will crop an area of an image using a shapefile as cutline and resampling to a different
# pixel size -  outputs are saved to disk
# Start the clock!
# srcfile is your full orthophoto mosaic
# dstfile is the output subset area
# cutline is your shapefile AOI
# 
ptm <- proc.time()
gdalwarp(srcfile=here("34_twincreek_apiary_t1/odm_orthophoto.tif"),
        dstfile=here("34_twincreek_apiary_t1/odm_orthophoto_res3.tif"),
        cutline = here("34_twincreek_apiary_t1/10acre/10acre_extent/10acre_extent.shp"),
        srcnodata = 0, dstnodata = 0, 
        crop_to_cutline=TRUE, r="near", tr=c(x.out[1],x.out[2]),
        output_Raster=TRUE,overwrite=TRUE,verbose=TRUE,
        t_srs='+proj=utm +zone=12 +datum=WGS84')
# Stop the clock
proc.time() - ptm         

# Crop the DSM - arguments for inputs are identical to the orthophoto ones above
gdalwarp(srcfile=here("34_twincreek_apiary_t1/dsm.tif"),
         dstfile=here("34_twincreek_apiary_t1/dsm_crop.tif"),
         cutline = here("34_twincreek_apiary_t1/10acre/10acre_extent/10acre_extent.shp"),
         crop_to_cutline=TRUE, r="bilinear", tr=c(x.out[1],x.out[2]),
         output_Raster=TRUE,overwrite=TRUE,verbose=TRUE,
         t_srs='+proj=utm +zone=12 +datum=WGS84')

# Crop the DTM - arguments for inputs are identical to the orthophoto ones above
gdalwarp(srcfile=here("34_twincreek_apiary_t1/dtm.tif"),
         dstfile=here("34_twincreek_apiary_t1/dtm_crop.tif"),
         cutline = here("34_twincreek_apiary_t1/10acre/10acre_extent/10acre_extent.shp"),
         crop_to_cutline=TRUE, r="bilinear", tr=c(x.out[1],x.out[2]),
         output_Raster=TRUE,overwrite=TRUE,verbose=TRUE,
         t_srs='+proj=utm +zone=12 +datum=WGS84')


# This will subset the first 3 bands of the subset orthophoto and save it to disk
# Start the clock!
#src_dataset is the output tiff file obtained after running gdalwarp
#dst_dataset is the output image with only the 3 initial bands RGB
# b is a vector with the band numbers that are desired
ptm <- proc.time()
gdal_translate(src_dataset=here("34_twincreek_apiary_t1//odm_orthophoto_res3.tif"),
               dst_dataset=here("34_twincreek_apiary_t1//odm_orthophoto_res_3b.tif"),
                                b=c(1,2,3),overwrite=TRUE,verbose=TRUE)
# Stop the clock
proc.time() - ptm  

# This will generate a pseudo color image (1 band) from the previous output to be used
# as input for the texture-based covariates

# Start the clock!
# First input is the output TIFF with only 3 bands
# second argument is the output pseudo color image (Only one 8-Bit band)
ptm <- proc.time()
system(paste0("rgb2pct.py ",here("34_twincreek_apiary_t1/input_3b.tif "),
              here("34_twincreek_apiary_t1/pseudo-colored3.tif")))
# Stop the clock
proc.time() - ptm  


#############################################################################################
######## Data Exploration and preliminary geoprocessing
## Here we use the training polygons to extract covariates of
## interest that can be used for model fitting
## We are mainly using the raster package and raster package objects + simple features sf objects

# Read in the training polygons 
training.sf<- st_read(here("34_10acre_test/training_samples/training_samples.shp")) # as SF object for later use with exactextract

# Reading in the orthophoto 
ortho.foto.raster<-raster::brick(here("34_twincreek_apiary_t1/odm_orthophoto_res_3b.tif")) # 
# Reading in the DSM  
DSM.raster<-raster(here("34_twincreek_apiary_t1//dsm_crop.tif")) # terra object
# Reading in the DTM  
DTM.raster<-raster(here("34_twincreek_apiary_t1//dtm_crop.tif")) # terra object


######################################### Stack the orthophoto, DSM and DTM to calculate heights
Stack01<-raster::stack(ortho.foto.raster, DTM.raster, DSM.raster)

Stack01$height<- abs(Stack01$dtm_crop - Stack01$dsm_crop)

#######################################################
#### Prepare functions to compute VIs
# This function requires a raster object to compute the requested indices


redgreenblue_ind<-function(red, green, blue, rgbindex= c("EBI", "VARI",
                                                         "VDVI", "MGVRI","CIVE"))
  {
  indices<-lapply(rgbindex, function(item){
    if (item == "EBI"){
      message(" Computing EBI")
      bright<-(red+green+blue)
      gss<-((green/blue)*(red-blue+255))
      EBI<-bright / gss
      names(EBI)<-"EBI"
      return(EBI)
    }
    else if (item == "VARI"){
      message(" Computing VARI")
      VARI<-(green - red) / (green + red - blue)
      names(VARI)<-"VARI"
      return(VARI)
    }
    else if (item == "VDVI"){
      message(" Computing VDVI")
      VDVI<-((2*green-red-blue)/(2*green+red+blue))
      names(VDVI)<-"VDVI"
      return(VDVI)
  }
    else if (item == "MGVRI"){
      message(" Computing MGVRI")
      MGVRI<-((green^2-red^2)/(green^2+red^2))
      names(MGVRI)<-"MGVRI"
      return(MGVRI)
    }
    else if (item == "CIVE"){
      message(" Computing CIVE")
      CIVE<-(0.441*red - 0.881*green + 0.385*blue + 18.787)
      names(CIVE)<-"CIVE"
      return(CIVE)
    }
  })
  return(raster::stack(indices))
}

######################################
###################################### Apply the general function "redgreenblue_ind"
# To get a raster::stack object where each band is one of the requested VI's
# The arguments require to define which band index is the red, the green or the blue
# Notice that you can use the stack that already has the ortho, dsm, dtm, and height bands
# the rgbindex argument can be customized to only extract one VI or multiple VI's
# Start the clock!
ptm <- proc.time()
rgbInd<- redgreenblue_ind( red = ortho.foto.raster[[1]],
                           green = ortho.foto.raster[[2]],
                           blue = ortho.foto.raster[[3]],
                           rgbindex = c("EBI","VARI","MGVRI"))
# Stop the clock
proc.time() - ptm  


##################  rescale data to 8-Bit format
######## You can do this with the different raster::stacks that are being generated - i.e. the 
# raster::stack that has the orthophoto, dtm, dsm and height, the raster::stack that contains the
# vegetation indices, and the texture-based metrics
mnv <- cellStats(ortho.foto.raster,'min') # minimum value of the raster
mxv <- cellStats(ortho.foto.raster,'max') # maximum value of the raster
x <- ((ortho.foto.raster - mnv)*255) / (mxv - mnv) # substitute "x" for a name for a particular 
# raster::stack and substitute "ortho.foto.raster" with the name of the particular raster::stack 
# that is being rescaled


###############################################################################################

# Quantization and derivation of texture-based metrics using the GLCM package - this packages does NOT
# use Terra objects but raster objects
# Arguments
# x = raster object - here you can use the pseudo color TIFF or the 3-band RGB TIFF, if you use the latter
# then you need to provide which band to use, in general band or layer = 2 (green band) works fine
# using values greater than 16 for the n_grey may be too computationally expensive
# You can try different sizes for the moving window, and the shifts

#####
# Start the clock!
ptm <- proc.time()
x.glcm<- glcm(x=raster(here("34_twincreek_apiary_t1/input_3b.tif"),layer=2),
     n_grey = 16, window = c(3, 3), shift = list(c(0,1), c(1,1), c(1,0), c(1,-1)),
     statistics = 
       c("mean", "correlation"),  na_opt="ignore", 
     na_val=NA,  asinteger=TRUE)
# Stop the clock
proc.time() - ptm

# This ouput raster::stack object has as many bands as metrics were requested in the argument "statistics"

Composite<-raster::stack() # here we can create one big object of all the rescaled stack objects (i.e. orthophoto,
# texture-based metrics and VIs)

###############################################################################################
################ Drill through all the raster covariate layers and extract values for each pixel
################ that intersects the training polygons

# Extract pixel values using exactextract functions
prec_dfs <- exact_extract(Composite, training.sf, include_xy=TRUE, include_cols=c('Id','class_name','class'))
tbl <- do.call(rbind, prec_dfs) # convert the previous list object to a dataframe

# A new dataset with just the variables we want to keep or rows... (i.e. filter out partial pixels).
# You should only keep the response variable (numeric variable that has been converted to a factor)

tbl.trim<-tbl[,c(3:6,9:22 )]
tbl.trim$class<-as.factor(tbl.trim$class) # in this case the column "class" has the values that identify
# the classes that we are trying to map

tbl.trim <- tbl.trim[complete.cases(tbl.trim), ] # Eliminate all rows that have NA or empty values for 
# any variable

#######################################################
# Boxplots to assess if classess can be separated using a particular variable
#######################################################

# EBI distributions
boxplot(EBI~class_name,data=tbl, main="EBI distribution per class - original RGB",
        xlab="Classes", ylab="EBI values")

# Height distributions
boxplot(height~class_name,data=tbl, main="Height distribution per class",
        xlab="Classes", ylab="Height (DSM - DTM)")


#############################################################################################################
#############################################################################################################
#############################################################################################################

### Fit a RandomForest Classifier

set.seed(123) # Run this before the sampling so that it's reproducible or do not run it if you want different results each time
split1<- sample(c(rep(1, 0.7 * nrow(tbl.trim)), rep(0, 0.3 * nrow(tbl.trim)))) # Create a vector to split into training ==1 and test ==0
table(split1)
tbl.train <- tbl.trim[split1 == 1, ] # subset dataset used for initial model fit
tbl.test <-  tbl.trim[split1 == 0, ] # subset dataset used for independent model validation

summary(as.factor(tbl.train$class))
summary(as.factor(tbl.test$class))

#### Optimization
mtry <- tuneRF(tbl.trim[-1],tbl.trim$class, ntreeTry=500,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m)

set.seed(71)
########### fit a full RF model - uses ALL available covariates
rf <-randomForest(class~.,data=tbl.train, mtry=best.m, importance=TRUE,ntree=500)
print(rf)
#Evaluate variable importance
importance(rf)
varImpPlot(rf)

# Fit simpler or reduced models now that we've assessed variable importance
set.seed(71)
rf.simple <-randomForest(class~height+VDVI+MGRVI+Red+EBI,data=tbl.train, mtry=best.m, importance=TRUE,ntree=500)

print(rf.simple)
#Evaluate variable importance
importance(rf.simple)
varImpPlot(rf.simple)

# Apply the fitted model to the test dataset
prediction <-predict(rf.simple, tbl.test)
prediction <-predict(rf, tbl.test)
# Assess the accuracy of our model on a test dataset
confusionMatrix(prediction, tbl.test$class)


# Spatial prediction - to get a spatially-explicit response variable grid
Rast.Predict <- predict(Composite, rf, type='response')

# Custom palette
my_palette <- brewer.pal(n = 6, name = "Dark2")
my_palette


# Plot
mapView(as(Rast.Predict, "Raster"), col.regions = my_palette)+mapview(as(Ortho.foto.res,"Raster"))
raster::spplot(Rast.Predict)


writeRaster(Rast.Predict, filename="Prediction_Test02022022.tif", overwrite=TRUE)

