# USDA / ARS-FRRL-PIBR Logan Utah
# Floral resources mapping:
# Authors: Alexander Hernandez alexander.hernandez@usda.gov / Jesse Tabor Jessetabor12@gmail.com
# * Random Forest model fit to classify RGB imagery into landscape features (flowers, shrubs, soil)
# * Pre-processing: Uses Geospatial Data Abstraction Library GDAL tools
# * AOI extracted from original imagery is ~ 10 acres in size and its spatial domain contains the field data transect
# * RGB imagery processing (vegetation indices + texture-based covariates) uses raster::raster objects
# * Vegetations indices are calculated using a general customized function
# * Texture-based metrics are obtained using glcm package https://github.com/azvoleff/glcm
# * raster::stack objects pixel values extractions are obtained using exactextractr package (https://github.com/isciences/exactextractr)
# * Model is fit using 70% of observations for training - 30% for validation
# * Full RF is initially fit and then parsimonious model is identified by assessing variable importance metrics
# * Geospatial predictions are obtained using raster::predict that solely uses raster:stack objects





rm(list = ls(all.names = TRUE)) # will clear all objects, including hidden objects
gc() # free up memory and report memory usage

# set working directory. This will be the directory where all the unzipped files are located
setwd("/media/geospastore/shared/Floral/in_data/apiary")
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
library(GLCMTextures)

# Correlation test
library(Hmisc)
library(corrplot)
library(caret)

dir.create("42_frank_apiary_t2/output")
dir.create("42_frank_apiary_t2/intermediate")

######################################### Crop AOI from full mosaic and resample using GDAL-based pre processing ###############################################

# This will crop the full mosaic using the 10-acre Area of Interest AOI around the field transect location, and then extract only the 3 bands of interest

# First get desired output resolution (pixel size) for the orthophoto that requires cropping, in this case we can use the DTM or DSM
x.out<-terra::res(rast(here("42_frank_apiary_t2/dsm.tif")))

# This will crop an area of an image using a shapefile as cutline and resampling to a different pixel size -  outputs are saved to disk
# srcfile is your full orthophoto mosaic
# dstfile is the output subset area
# cutline is your shapefile AOI

# Crop the orthophoto
# Start the clock!
ptm <- proc.time()
gdalwarp(srcfile=here("42_frank_apiary_t2/odm_orthophoto.tif"),
         dstfile=here("42_frank_apiary_t2/intermediate/odm_orthophoto_resample.tif"),
         cutline = here("42_frank_apiary_t2/extent/extent.shp"),
         srcnodata = 0, dstnodata = 0, 
         crop_to_cutline=TRUE, r="near", tr=c(x.out[1],x.out[2]),
         output_Raster=TRUE,overwrite=TRUE,verbose=TRUE,
         t_srs='+proj=utm +zone=12 +datum=WGS84')
# Stop the clock
proc.time() - ptm         


# Crop the DSM - arguments for inputs are identical to the orthophoto ones above
# Start the clock!
ptm <- proc.time()
gdalwarp(srcfile=here("42_frank_apiary_t2/dsm.tif"),
         dstfile=here("42_frank_apiary_t2/intermediate/dsm_crop.tif"),
         cutline = here("42_frank_apiary_t2/extent/extent.shp"),
         crop_to_cutline=TRUE, r="bilinear", tr=c(x.out[1],x.out[2]),
         output_Raster=TRUE,overwrite=TRUE,verbose=TRUE,
         t_srs='+proj=utm +zone=12 +datum=WGS84')
# Stop the clock
proc.time() - ptm


# Crop the DTM - arguments for inputs are identical to the orthophoto ones above
# Start the clock!
ptm <- proc.time()
gdalwarp(srcfile=here("42_frank_apiary_t2/dtm.tif"),
         dstfile=here("42_frank_apiary_t2/intermediate/dtm_crop.tif"),
         cutline = here("42_frank_apiary_t2/extent/extent.shp"),
         crop_to_cutline=TRUE, r="bilinear", tr=c(x.out[1],x.out[2]),
         output_Raster=TRUE,overwrite=TRUE,verbose=TRUE,
         t_srs='+proj=utm +zone=12 +datum=WGS84')
# Stop the clock
proc.time() - ptm


# This will subset the first 3 bands of the orthophoto and save it to disk
#src_dataset is the output tiff file obtained after running gdalwarp
#dst_dataset is the output image with only the 3 initial bands RGB
# b is a vector with the band numbers that are desired

# Start the clock!
ptm <- proc.time()
gdal_translate(src_dataset=here("42_frank_apiary_t2/intermediate/odm_orthophoto_resample.tif"),
               dst_dataset=here("42_frank_apiary_t2/intermediate/odm_orthophoto_resample_3b.tif"),
               b=c(1,2,3),overwrite=TRUE,verbose=TRUE)
# Stop the clock
proc.time() - ptm





############################################################### Read in AOI data and training polygons #########################################################

# We are mainly using the raster package and raster package objects + simple features sf objects

# Read in the training polygons 
training.sf<- st_read(here("42_frank_apiary_t2/reduced_samples/training_samples.shp")) # as SF object for later use with exactextract
# Reading in the orthophoto 
ortho.foto.raster<-raster::brick(here("42_frank_apiary_t2/intermediate/odm_orthophoto_resample_3b.tif"))
# Reading in the DSM  
DSM.raster<-raster(here("42_frank_apiary_t2/intermediate/dsm_crop.tif")) # terra object
# Reading in the DTM  
DTM.raster<-raster(here("42_frank_apiary_t2/intermediate/dtm_crop.tif")) # terra object

# Create a raster stack from the orthophoto, DSM, and DTM
orthoEle<-raster::stack(ortho.foto.raster, DTM.raster, DSM.raster)
# Calculate height
orthoEle$height<- abs(orthoEle$dtm_crop - orthoEle$dsm_crop)

# Save the output orthoEle raster
writeRaster(orthoEle, filename="42_frank_apiary_t2/intermediate/orthoEle.tif", overwrite=TRUE)





############################################################## Create pseudo color image using rgb2pct.py ######################################################

# This will generate a pseudo color image (1 band) from the previous output to be used as input for the texture-based covariates 
# (NO NEED TO USE THIS LINE IF USING ODM_ORTHOPHOTO_RESAMPLE_3B FOR CREATING TEXTURE COVARIATES)
# First input is the output TIFF with only 3 bands
# Second argument is the output pseudo color image (Only one 8-Bit band)

# Start the clock!
ptm <- proc.time()
system(paste("rgb2pct.py ",here("42_frank_apiary_t2/intermediate/odm_orthophoto_resample_3b.tif"),here("42_frank_apiary_t2/intermediate/pseudo-color.tif")))
# Stop the clock
proc.time() - ptm  






############################################################### Create texture-based covariates ################################################################

# Quantization and derivation of texture-based metrics using the GLCM package - this packages does NOT use Terra objects but raster objects
# x = raster object - here you can use the pseudo color TIFF or the 3-band RGB TIFF, if you use the latter
# then you need to provide which band to use, in general band or layer = 2 (green band) works fine
# using values greater than 16 for the n_grey may be too computationally expensive
# You can try different sizes for the moving window, and the shifts


# Start the clock!
#ptm <- proc.time()
#x.glcm<- glcm(x=raster(here("42_frank_apiary_t2/odm_orthophoto_resample_3b.tif"),layer=2),
              #n_grey = 16, window = c(3, 3), shift = list(c(0,1), c(1,1), c(1,0), c(1,-1)),
              #statistics = c("mean", "correlation", "contrast", "variance"),  na_opt="ignore", na_val=NA,  asinteger=TRUE)
#names(x.glcm)<-c("mean", "correlation", "contrast", "variance")
# Stop the clock
#proc.time() - ptm

# Start the clock!
ptm <- proc.time()
x.glcm<- glcm_textures(rast(here("42_frank_apiary_t2/intermediate/pseudo-color.tif")), w = c(3,5), n_levels = 16, quantization = "equal prob",shift=c(1, 0), 
                       metrics = c("glcm_mean", "glcm_variance", "glcm_correlation"))
names(x.glcm)<-c("mean", "variance", "correlation")
# Stop the clock
proc.time() - ptm

# Create a raster starck from the textures output
texture<-raster::stack(x.glcm)





############################################################### Prepare function to compute VIs ################################################################

# This function requires a raster object to compute the requested indices
redgreenblue_ind<-function(red, green, blue, rgbindex= c("EBI","VDVI","MGVRI",
                                                         "CIVE","NGRDI","ExG",
                                                         "RGBVI","VEG","NGBDI"))
{
  indices<-lapply(rgbindex, function(item){
    if (item == "EBI"){ # (Enhanced bloom index)
      message("Computing EBI")
      bright<-(red+green+blue)
      gss<-((green/blue)*(red-blue+255))
      EBI<-bright / gss
      names(EBI)<-"EBI"
      return(EBI)
    }
    else if (item == "VDVI"){ # (Visible band difference vegetation index)
      message("Computing VDVI")
      VDVI<-((2*green-red-blue)/(2*green+red+blue))
      names(VDVI)<-"VDVI"
      return(VDVI)
    }
    else if (item == "MGVRI"){ # (Modified green red vegetation index)
      message("Computing MGVRI")
      MGVRI<-((green^2-red^2)/(green^2+red^2))
      names(MGVRI)<-"MGVRI"
      return(MGVRI)
    }
    else if (item == "CIVE"){ # (Color index of vegetation)
      message("Computing CIVE")
      CIVE<-(0.441*red - 0.881*green + 0.385*blue + 18.787)
      names(CIVE)<-"CIVE"
      return(CIVE)
    }
    else if (item == "NGRDI"){ # (Normalized green-red difference index)
      message("Computing NGRDI")
      NGRDI <-((green-red)/(green+red))
      names(NGRDI)<-"NGRDI"
      return(NGRDI)
    }
    else if (item == "RGRI"){ # (Red-green ratio index)
      message("Computing RGRI")
      RGRI <-(red/green)
      names(RGRI)<-"RGRI"
      return(RGRI)
    }
    else if (item == "ExG"){ # (Excess green index)
      message("Computing ExG")
      ExG <-(2*green-red-blue)
      names(ExG)<-"ExG"
      return(ExG)
    }
    else if (item == "RGBVI"){ # (RGB-based vegetation index)
      message("Computing RGBVI")
      RGBVI <-(((green^2)-(blue*red))/((green^2)+(blue*red)))
      names(RGBVI)<-"RGBVI"
      return(RGBVI)
    }
    else if (item == "BGI"){ # (Simple blue-green ratio)
      message("Computing BGI")
      BGI <-(blue/green)
      names(BGI)<-"BGI"
      return(BGI)
    }
    else if (item == "VEG"){ # (Vegetativen)
      message("Computing VEG")
      VEG <-((green/(red^0.667))*(blue^(1-0.667)))
      names(VEG)<-"VEG"
      return(VEG)
    }
    else if (item == "NGBDI"){ # (Normalized green-blue difference index)
      message("Computing NGBDI")
      NGBDI <-((green-blue)/(green+blue))
      names(NGBDI)<-"NGBDI"
      return(NGBDI)
    }
    else if (item == "RGBVI2"){ # (RGB-based vegetation index 2)
      message("Computing RGBVI2")
      RGBVI2 <-((green-red)/blue)
      names(RGBVI2)<-"RGBVI2"
      return(RGBVI2)
    }
    else if (item == "RGBVI3"){ # (RGB-based vegetation index 3)
      message("Computing RGBVI3")
      RGBVI3 <-((green+blue)/red)
      names(RGBVI3)<-"RGBVI3"
      return(RGBVI3)
    }
  })
  return(raster::stack(indices))
}





############################################### Apply the general function "redgreenblue_ind" to ortho.foto.raster #############################################

# To get a raster::stack object where each band is one of the requested VI's the arguments require to define which band index is the red, the green or the blue
# Notice that you can use the stack that already has the ortho, dsm, dtm, and height bands
# The rgbindex argument can be customized to only extract one VI or multiple VI's

# Start the clock!
ptm <- proc.time()
rgbInd<- redgreenblue_ind( red = ortho.foto.raster[[1]],
                           green = ortho.foto.raster[[2]],
                           blue = ortho.foto.raster[[3]],
                           rgbindex = c("EBI","VDVI","MGVRI",
                                        "CIVE","NGRDI","ExG",
                                        "RGBVI","VEG","NGBDI"))
# Stop the clock
proc.time() - ptm  





################################################################## Rescale data to 8-bit format ################################################################

# You can do this with the different raster::stacks that are being generated - i.e. the 
# raster::stack that has the orthophoto, dtm, dsm and height, the raster::stack that contains the
# vegetation indices, and the texture-based metrics

# Check input data for unusual values
texture
rgbInd

# Start the clock!
ptm <- proc.time()
mnv <- cellStats(texture,'min') # minimum value of the raster
mxv <- cellStats(texture,'max') # maximum value of the raster
texture8 <- round(((texture - mnv)*255) / (mxv - mnv)) # substitute "x" for a name for a particular
names(texture8)<-c("mean", "variance", "correlation")
# Stop the clock
proc.time() - ptm

# Save the output textures raster
writeRaster(texture8, filename="42_frank_apiary_t2/intermediate/texture8.tif", overwrite=TRUE)

# Start the clock!
ptm <- proc.time()
mnv <- cellStats(rgbInd,'min') # minimum value of the raster
mxv <- cellStats(rgbInd,'max') # maximum value of the raster
rgbInd8 <- round(((rgbInd - mnv)*255) / (mxv - mnv)) # substitute "x" for a name for a particular
names(rgbInd8)<-c("EBI","VDVI","MGVRI","CIVE","NGRDI","ExG","RGBVI","VEG","NGBDI")
# Stop the clock
proc.time() - ptm

# Save the output rgbInd8 raster
writeRaster(rgbInd8, filename="42_frank_apiary_t2/intermediate/rgbInd8.tif", overwrite=TRUE)

# Check output data is 0-255
texture8
rgbInd8





################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
################################################################## Create composite ############################################################################

# Create composite with 8-bit data
composite<-raster::stack(orthoEle,texture8,rgbInd8) # here we can create one big object of all the rescaled stack objects (i.e. orthophoto,
# texture-based metrics and VIs)





########################################################### Extract pixel values from training data ############################################################

# Drill through all the raster covariate layers and extract values for each pixel that intersects the training polygons

# Extract pixel values using exactextract functions
prec_dfs <- exact_extract(composite, training.sf, include_xy=TRUE, include_cols=c('Id','class_name','class'))
tbl <- do.call(rbind, prec_dfs) # convert the previous list object to a dataframe
write.csv(tbl, here("42_frank_apiary_t2/intermediate/training_data.csv"))

# A new dataset with just the variables we want to keep or rows... (i.e. filter out partial pixels).
# You should only keep the response variable (numeric variable that has been converted to a factor)
tbl.trim<-tbl[,c(3:6,9:21)] # This may change depending on what classes are decided to go into the model
tbl.trim$class<-as.factor(tbl.trim$class) # In this case the column "class" has the values that identify the classes that we are trying to map

tbl.trim <- tbl.trim[complete.cases(tbl.trim), ] # Eliminate all rows that have NA or empty values for any variable
write.csv(tbl.trim, here("42_frank_apiary_t2/intermediate/trim_training_data.csv"))





########################################################################## Correlation test #################################################################

#################################################### Correlation test on indices

con <- file("42_frank_apiary_t2/output/correlation_output.log")
sink(con, append=TRUE)
sink(con, append=TRUE, type="message")

tbl_cor <- tbl.trim[,c(9:17)]

mydata.cor = cor(tbl_cor, method = c("pearson"))

mydata.rcorr = rcorr(as.matrix(tbl_cor))
mydata.rcorr

mydata.coeff = mydata.rcorr$r
mydata.p = mydata.rcorr$P

jpeg("42_frank_apiary_t2/output/correlation_indices_test.jpeg", width = 1100, height = 1100, type="cairo")
corrplot(mydata.cor, method = 'number')
dev.off()

# Remove the highly correlated variables
df2 = cor(tbl_cor)
hc = findCorrelation(df2, cutoff=0.6) # put any value as a "cutoff" 
hc = sort(hc)
reduced_Data = tbl_cor[,-c(hc)]
print (reduced_Data)

# Plot the "non-correlated data

non_cor = cor(reduced_Data, method = c("pearson"))

mydata.non_corr = rcorr(as.matrix(reduced_Data))
mydata.non_corr

non_cor.coeff = mydata.non_corr$r
non_cor.coeff
non_cor.p = mydata.non_corr$P
non_cor.p

jpeg("42_frank_apiary_t2/output/correlation_reduced_indices_test.jpeg", width = 900, height = 900, type="cairo")
corrplot(non_cor, method = 'number')
dev.off()

# Restore output to console
sink() 
sink(type="message")

mydata.non_corr


################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
########################################################## Subset final table to use in randomForest ###########################################################


tbl.trim2<-tbl.trim[,c(1:8,13:14,16)]

################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
################################################################### Load saved data to re-run randomForest here ################################################

# Read in raster data
#orthoEle<-raster::brick(here("42_frank_apiary_t2/intermediate/orthoEle.tif"))
#texture8<-raster::brick(here("42_frank_apiary_t2/intermediate/texture8.tif"))
#rgbInd8<-raster::brick(here("42_frank_apiary_t2/intermediate/rgbInd8.tif"))

# Read in training data
#tbl.trim<-read.csv("42_frank_apiary_t2/intermediate/trim_training_data.csv")

# Subset final table to use in randomForest
#tbl.trim2<-tbl.trim[,c()]

# Create composite with 8-bit data
#composite<-raster::stack(orthoEle,texture8,rgbInd8) # here we can create one big object of all the rescaled stack objects (i.e. orthophoto,
# texture-based metrics and VIs)





################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
################################################################### Fit a randomForest classifier ##############################################################

# First, create a full randomForest model

con <- file("42_frank_apiary_t2/output/RF_output.log")
sink(con, append=TRUE)
sink(con, append=TRUE, type="message")

set.seed(123) # Run this before the sampling so that it's reproducible or do not run it if you want different results each time
split1<- sample(c(rep(1, 0.7 * nrow(tbl.trim2)), rep(0, 0.3 * nrow(tbl.trim2)))) # Create a vector to split into training ==1 and test ==0
table(split1)
tbl.train <- tbl.trim2[split1 == 1, ] # subset dataset used for initial model fit
tbl.test <-  tbl.trim2[split1 == 0, ] # subset dataset used for independent model validation

summary(as.factor(tbl.train$class))
summary(as.factor(tbl.test$class))

#### Optimization
jpeg("42_frank_apiary_t2/output/RF_mtry.jpeg", type="cairo")
mtry <- tuneRF(tbl.trim2[-1],tbl.trim2$class, ntreeTry=500,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
dev.off()

best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m)

########### fit a full RF model - uses ALL available covariates
set.seed(71)
rf <-randomForest(class~.,data=tbl.train, mtry=best.m, importance=TRUE,ntree=500)

print(rf)
#Evaluate variable importance
importance(rf)

jpeg("42_frank_apiary_t2/output/RFfull_variable_importance.jpeg", type="cairo")
varImpPlot(rf)
dev.off()

varImpPlot(rf)




################################################# Fit a simple or reduced randomForest classifier ##############################################################

set.seed(71)
rf.simple <-randomForest(class~NGRDI+ExG+blue+red+mean+variance+VEG+green,data=tbl.train, mtry=best.m, importance=TRUE,ntree=500)

print(rf.simple)
#Evaluate variable importance
importance(rf.simple)

jpeg("42_frank_apiary_t2/output/RFsimple_variable_importance.jpeg", type="cairo")
varImpPlot(rf.simple)
dev.off()

varImpPlot(rf.simple)




############################################################ Apply the fitted model to the test dataset ########################################################

# Compare the full model to the simple model

prediction <-predict(rf, tbl.test)
# Assess the accuracy of our model on a test dataset
confusionMatrix(prediction, tbl.test$class)

prediction <-predict(rf.simple, tbl.test)
# Assess the accuracy of our model on a test dataset
confusionMatrix(prediction, tbl.test$class)


# Restore output to console
sink() 
sink(type="message")


# Start the clock!
ptm <- proc.time()
# Spatial prediction - to get a spatially-explicit response variable grid
Rast.Predict <- predict(composite, rf.simple, type='response')
# Stop the clock
proc.time() - ptm

# Custom palette
#my_palette <- brewer.pal(n = 7, name = "Dark2") # Change n to appropriate amount of classes
#my_palette

# Plot
#mapView(as(Rast.Predict, "Raster"), col.regions = my_palette)+mapview(as(ortho.foto.raster,"Raster"))
#raster::spplot(Rast.Predict)

# Save the output prediction raster
writeRaster(Rast.Predict, filename="42_frank_apiary_t2/output/Prediction_classification.tif", overwrite=TRUE)





