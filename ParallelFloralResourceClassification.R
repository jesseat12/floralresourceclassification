# USDA / ARS-FRRL-PIBR Logan Utah
# Floral resources mapping:
# Authors: Alexander Hernandez alexander.hernandez@usda.gov / Jesse Tabor Jessetabor12@gmail.com
# * Divide imagery into chunks and run model in parallel
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

setwd("D:/Jesse/FloralResources")
library(here)
here()

library(exactextractr)
library(sf)
library(caret)
library(rgdal)
library(gdalUtils)
library(raster)
library(randomForest)
library(landmap)
library(snowfall)
library(scutr)
library(raster)
#library(dplyr)
library(terra)

dir.create("34_twincreek_apiary_t1/3class_parallel")
dir.create("34_twincreek_apiary_t1/3class_parallel/tiled_rgb")
dir.create("34_twincreek_apiary_t1/3class_parallel/tiled_VIs")
dir.create("34_twincreek_apiary_t1/3class_parallel/tiled_tex")
dir.create("34_twincreek_apiary_t1/3class_parallel/outclass")
dir.create("34_twincreek_apiary_t1/3class_output")
dir.create("34_twincreek_apiary_t1/flower_output")
dir.create("34_twincreek_apiary_t1/PredictionClassification")
dir.create("34_twincreek_apiary_t1/PredictionClassification/reclassify")

################################################################### Load saved data to re-run randomForest ################################################

# Read in raster data
ortho<-raster::brick(here("34_twincreek_apiary_t1/intermediate/odm_orthophoto_crop_3b.tif"))
names(ortho)<-c("red","green","blue")
texture8<-raster::brick(here("34_twincreek_apiary_t1/intermediate/texture8.tif"))
names(texture8)<-c("contrast","dissimilarity","homogeneity","ASM","entropy","mean","variance","correlation")
rgbInd8<-raster::brick(here("34_twincreek_apiary_t1/intermediate/rgbInd8.tif"))
names(rgbInd8)<-c("EBI","VDVI","MGVRI","CIVE","NGRDI","RGRI","ExG","RGBVI","BGI","VEG","NGBDI","RGBVI2","RGBVI3")

# Create composite with 8-bit data
composite<-raster::stack(ortho,texture8,rgbInd8) # here we can create one big object of all the rescaled stack objects (i.e. orthophoto,
#writeRaster(composite, filename="34_twincreek_apiary_t1/intermediate/composite.tif", overwrite=TRUE)

########################################################### Extract pixel values from training data ############################################################

# Read in the training polygons 
training.sf<- st_read(here("34_twincreek_apiary_t1/3class_samples/training_samples.shp")) # as SF object for later use with exactextract

# Drill through all the raster covariate layers and extract values for each pixel that intersects the training polygons

# Extract pixel values using exactextract functions
prec_dfs <- exact_extract(composite, training.sf, include_xy=TRUE, include_cols=c('Id','class_name','class'))
tbl <- do.call(rbind, prec_dfs) # convert the previous list object to a dataframe

# A new dataset with just the variables we want to keep or rows... (i.e. filter out partial pixels).
# You should only keep the response variable (numeric variable that has been converted to a factor)
tbl.trim<-tbl[,c(3:13,15:27)] # This may change depending on what classes are decided to go into the model
tbl.trim$class<-as.factor(tbl.trim$class) # In this case the column "class" has the values that identify the classes that we are trying to map

tbl.trim <- tbl.trim[complete.cases(tbl.trim), ] # Eliminate all rows that have NA or empty values for any variable
write.csv(tbl.trim, here("34_twincreek_apiary_t1/3class_output/trim_training_data.csv"))

tbl.trim$class<-as.factor(tbl.trim$class)

################################################################### Fit a randomForest classifier ##############################################################

# First, create a randomForest model
con <- file("34_twincreek_apiary_t1/3class_output/full_RF_output.log")
sink(con, append=TRUE)
sink(con, append=TRUE, type="message")

set.seed(123) # Run this before the sampling so that it's reproducible or do not run it if you want different results each time
split1<- sample(c(rep(1, 0.7 * nrow(tbl.trim)), rep(0, 0.3 * nrow(tbl.trim)))) # Create a vector to split into training ==1 and test ==0
table(split1)
tbl.train <- tbl.trim[split1 == 1, ] # subset dataset used for initial model fit
tbl.test <-  tbl.trim[split1 == 0, ] # subset dataset used for independent model validation

summary(as.factor(tbl.train$class))
summary(as.factor(tbl.test$class))

#### Optimization
jpeg("34_twincreek_apiary_t1/3class_output/RF_mtry.jpeg", type="cairo")
mtry <- tuneRF(tbl.trim[-1],tbl.trim$class, ntreeTry=500,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
dev.off()

best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m)

################################################# Fit a full randomForest classifier ##############################################################

set.seed(123)
rf_full <-randomForest(class~. ,data=tbl.train, mtry=best.m, importance=TRUE,ntree=500)

print(rf_full)
#Evaluate variable importance
importance(rf_full)

jpeg("34_twincreek_apiary_t1/3class_output/RF_variable_importance.jpeg", type="cairo")
varImpPlot(rf_full)
dev.off()

varImpPlot(rf_full)

prediction <-predict(rf_full, tbl.test)
# Assess the accuracy of our model on a test dataset
confusionMatrix(prediction, tbl.test$class)

# Restore output to console
sink() 
sink(type="message")

################################################# Fit a simple randomForest classifier ##############################################################

# First, create a randomForest model
con <- file("34_twincreek_apiary_t1/3class_output/simple_RF_output.log")
sink(con, append=TRUE)
sink(con, append=TRUE, type="message")

set.seed(123)
rf <-randomForest(class~red+green+variance+mean+MGVRI+CIVE,data=tbl.train, mtry=best.m, importance=TRUE,ntree=500)

print(rf)
#Evaluate variable importance
importance(rf)

jpeg("34_twincreek_apiary_t1/3class_output/RF_simple_variable_importance.jpeg", type="cairo")
varImpPlot(rf)
dev.off()

varImpPlot(rf)

prediction <-predict(rf, tbl.test)
# Assess the accuracy of our model on a test dataset
confusionMatrix(prediction, tbl.test$class)

# Restore output to console
sink() 
sink(type="message")


########################################################### Extrapolate rf model on 10 acre area of interest ######################################################

# Prepare an RGDAL-based tiling system
# This can be any of your intermediate TIFFs - here I am using the ortho RGB
obj <- GDALinfo(here::here("34_twincreek_apiary_t1/intermediate/odm_orthophoto_crop_3b.tif")) #extent

## tile to 10 meter blocks with 5% overlap:
tile.lst <- getSpatialTiles(obj, overlap.percent=5, block.x=10, return.SpatialPolygons=TRUE)
tile.tbl <- getSpatialTiles(obj, overlap.percent=5,block.x=10, return.SpatialPolygons=FALSE)

tile.tbl$ID <- as.character(1:nrow(tile.tbl))
head(tile.tbl)

tile.pol <- SpatialPolygonsDataFrame(tile.lst, tile.tbl, match.ID = FALSE)

#
# Prepare functions to run in parallel processing
# This will clip each of the TIFF - You will need to change the input file and output folder each
# time that run the function:: change the location of out.path and Ortho
# Required folder name for ortho == tiled_rgb
# Required folder name for VIs == tiled_VIs
# Required folder name for textires == tiled_tex
# RGDAL-based to clip tiles and write them to disk
make_LC_tiles <- function(i, tile.tbl, 
                          out.path=here::here("34_twincreek_apiary_t1/3class_parallel/tiled_VIs"),
                          Ortho=here::here("34_twincreek_apiary_t1/intermediate/rgbInd8.tif")
){
  out.tif = paste0(out.path, "/T_", tile.tbl[i,"ID"], ".tif")
  if(!file.exists(out.tif)){
    m <- readGDAL(Ortho, offset=unlist(tile.tbl[i,c("offset.y","offset.x")]),
                  region.dim=unlist(tile.tbl[i,c("region.dim.y","region.dim.x")]),
                  output.dim=unlist(tile.tbl[i,c("region.dim.y","region.dim.x")]),
                  silent = TRUE)
    m <- as(m, "SpatialPixelsDataFrame")
    writeGDAL(m, out.tif, type="Int16", 
              options="COMPRESS=DEFLATE", mvFlag=-32768)
  }
}

### Parallel function to classify
### You will need to create the required output folder
classify.floral.RF <-function(i, 
                              out.path=here::here("34_twincreek_apiary_t1/3class_parallel/outclass")
){
  nombre<-paste0("T_",i)
  out.tif = paste0(out.path, "/T_",nombre,".tif")
  if(!file.exists(out.tif)){
    print(paste0("Processing...",here::here(paste0("tiled","/",i))))
    #RGB.tile <- terra::rast(here::here(paste0("tiled","/",nombre,".tif")),lyrs=2)
    RGB.tile <- terra::rast(here::here(paste0("34_twincreek_apiary_t1/3class_parallel/tiled_rgb","/",nombre,".tif")))
    names(RGB.tile)<-c("red","green","blue")
    VI.tile <- terra::rast(here::here(paste0("34_twincreek_apiary_t1/3class_parallel/tiled_VIs","/",nombre,".tif")))
    names(VI.tile)<-c("EBI",	"VDVI",	"MGVRI",	"CIVE",	"NGRDI",	"RGRI",	"ExG",
                      "RGBVI",	"BGI",	"VEG",	"NGBDI",	"RGBVI2",	"RGBVI3")
    text.tile <- terra::rast(here::here(paste0("34_twincreek_apiary_t1/3class_parallel/tiled_tex","/",nombre,".tif")),lyrs=c(1:7))
    names(text.tile)<-c("contrast",	"dissimilarity",	"homogeneity",	"ASM",
                        "entropy",	"mean",	"variance")
    
    #writeGDAL(m, out.tif, type="Int16", 
    #          options="COMPRESS=DEFLATE", mvFlag=-32768)
    composite<-c(RGB.tile, VI.tile, text.tile) 
    Rast.Predict <- terra::predict(composite, rf, type='response')
    
    terra::writeRaster(Rast.Predict, out.tif, filetype = "GTiff",
                       overwrite = TRUE)
    
  }
}

## Get a list of tile files TIFFs
list.tiles.RGB<-list.files(path=here::here("34_twincreek_apiary_t1/3class_parallel/tiled_rgb"))

# Parallelization
sfInit(parallel=TRUE, cpus=parallel::detectCores())
# Reload this line every time that you make changes to the functions
sfExport("make_LC_tiles", "tile.tbl","list.tiles.RGB", "classify.floral.RF","rf") # don't forget to include your RF model
sfLibrary(rgdal)
sfLibrary(plyr)
sfLibrary(terra)
sfLibrary(randomForest)

# Work in parallel to extract the tiles --- You will need to run this 3 times
ptm<-proc.time()
out.lst <- sfClusterApplyLB(1:nrow(tile.tbl), 
                            function(x){ make_LC_tiles(x, tile.tbl) })
proc.time()-ptm

#
# Prepare functions to run in parallel processing
# This will clip each of the TIFF - You will need to change the input file and output folder each
# time that run the function:: change the location of out.path and Ortho
# Required folder name for ortho == tiled_rgb
# Required folder name for VIs == tiled_VIs
# Required folder name for textires == tiled_tex
# RGDAL-based to clip tiles and write them to disk
make_LC_tiles <- function(i, tile.tbl, 
                          out.path=here::here("34_twincreek_apiary_t1/3class_parallel/tiled_tex"),
                          Ortho=here::here("34_twincreek_apiary_t1/intermediate/texture8.tif")
){
  out.tif = paste0(out.path, "/T_", tile.tbl[i,"ID"], ".tif")
  if(!file.exists(out.tif)){
    m <- readGDAL(Ortho, offset=unlist(tile.tbl[i,c("offset.y","offset.x")]),
                  region.dim=unlist(tile.tbl[i,c("region.dim.y","region.dim.x")]),
                  output.dim=unlist(tile.tbl[i,c("region.dim.y","region.dim.x")]),
                  silent = TRUE)
    m <- as(m, "SpatialPixelsDataFrame")
    writeGDAL(m, out.tif, type="Int16", 
              options="COMPRESS=DEFLATE", mvFlag=-32768)
  }
}

### Parallel function to classify
### You will need to create the required output folder
classify.floral.RF <-function(i, 
                              out.path=here::here("34_twincreek_apiary_t1/3class_parallel/outclass")
){
  nombre<-paste0("T_",i)
  out.tif = paste0(out.path, "/T_",nombre,".tif")
  if(!file.exists(out.tif)){
    print(paste0("Processing...",here::here(paste0("tiled","/",i))))
    #RGB.tile <- terra::rast(here::here(paste0("tiled","/",nombre,".tif")),lyrs=2)
    RGB.tile <- terra::rast(here::here(paste0("34_twincreek_apiary_t1/3class_parallel/tiled_rgb","/",nombre,".tif")))
    names(RGB.tile)<-c("red","green","blue")
    VI.tile <- terra::rast(here::here(paste0("34_twincreek_apiary_t1/3class_parallel/tiled_VIs","/",nombre,".tif")))
    names(VI.tile)<-c("EBI",	"VDVI",	"MGVRI",	"CIVE",	"NGRDI",	"RGRI",	"ExG",
                      "RGBVI",	"BGI",	"VEG",	"NGBDI",	"RGBVI2",	"RGBVI3")
    text.tile <- terra::rast(here::here(paste0("34_twincreek_apiary_t1/3class_parallel/tiled_tex","/",nombre,".tif")),lyrs=c(1:7))
    names(text.tile)<-c("contrast",	"dissimilarity",	"homogeneity",	"ASM",
                        "entropy",	"mean",	"variance")
    
    #writeGDAL(m, out.tif, type="Int16", 
    #          options="COMPRESS=DEFLATE", mvFlag=-32768)
    composite<-c(RGB.tile, VI.tile, text.tile) 
    Rast.Predict <- terra::predict(composite, rf, type='response')
    
    terra::writeRaster(Rast.Predict, out.tif, filetype = "GTiff",
                       overwrite = TRUE)
    
  }
}

# Reload this line every time that you make changes to the functions
sfExport("make_LC_tiles", "tile.tbl","list.tiles.RGB", "classify.floral.RF","rf") # don't forget to include your RF model

# Work in parallel to extract the tiles --- You will need to run this 3 times
ptm<-proc.time()
out.lst <- sfClusterApplyLB(1:nrow(tile.tbl), 
                            function(x){ make_LC_tiles(x, tile.tbl) })
proc.time()-ptm

#
# Prepare functions to run in parallel processing
# This will clip each of the TIFF - You will need to change the input file and output folder each
# time that run the function:: change the location of out.path and Ortho
# Required folder name for ortho == tiled_rgb
# Required folder name for VIs == tiled_VIs
# Required folder name for textires == tiled_tex
# RGDAL-based to clip tiles and write them to disk
make_LC_tiles <- function(i, tile.tbl, 
                          out.path=here::here("34_twincreek_apiary_t1/3class_parallel/tiled_rgb"),
                          Ortho=here::here("34_twincreek_apiary_t1/intermediate/odm_orthophoto_crop_3b.tif")
){
  out.tif = paste0(out.path, "/T_", tile.tbl[i,"ID"], ".tif")
  if(!file.exists(out.tif)){
    m <- readGDAL(Ortho, offset=unlist(tile.tbl[i,c("offset.y","offset.x")]),
                  region.dim=unlist(tile.tbl[i,c("region.dim.y","region.dim.x")]),
                  output.dim=unlist(tile.tbl[i,c("region.dim.y","region.dim.x")]),
                  silent = TRUE)
    m <- as(m, "SpatialPixelsDataFrame")
    writeGDAL(m, out.tif, type="Int16", 
              options="COMPRESS=DEFLATE", mvFlag=-32768)
  }
}

### Parallel function to classify
### You will need to create the required output folder
classify.floral.RF <-function(i, 
                              out.path=here::here("34_twincreek_apiary_t1/3class_parallel/outclass")
){
  nombre<-paste0("T_",i)
  out.tif = paste0(out.path, "/T_",nombre,".tif")
  if(!file.exists(out.tif)){
    print(paste0("Processing...",here::here(paste0("tiled","/",i))))
    #RGB.tile <- terra::rast(here::here(paste0("tiled","/",nombre,".tif")),lyrs=2)
    RGB.tile <- terra::rast(here::here(paste0("34_twincreek_apiary_t1/3class_parallel/tiled_rgb","/",nombre,".tif")))
    names(RGB.tile)<-c("red","green","blue")
    VI.tile <- terra::rast(here::here(paste0("34_twincreek_apiary_t1/3class_parallel/tiled_VIs","/",nombre,".tif")))
    names(VI.tile)<-c("EBI",	"VDVI",	"MGVRI",	"CIVE",	"NGRDI",	"RGRI",	"ExG",
                      "RGBVI",	"BGI",	"VEG",	"NGBDI",	"RGBVI2",	"RGBVI3")
    text.tile <- terra::rast(here::here(paste0("34_twincreek_apiary_t1/3class_parallel/tiled_tex","/",nombre,".tif")),lyrs=c(1:7))
    names(text.tile)<-c("contrast",	"dissimilarity",	"homogeneity",	"ASM",
                        "entropy",	"mean",	"variance")
    
    #writeGDAL(m, out.tif, type="Int16", 
    #          options="COMPRESS=DEFLATE", mvFlag=-32768)
    composite<-c(RGB.tile, VI.tile, text.tile) 
    Rast.Predict <- terra::predict(composite, rf, type='response')
    
    terra::writeRaster(Rast.Predict, out.tif, filetype = "GTiff",
                       overwrite = TRUE)
    
  }
}

# Reload this line every time that you make changes to the functions
sfExport("make_LC_tiles", "tile.tbl","list.tiles.RGB", "classify.floral.RF","rf") # don't forget to include your RF model

# Work in parallel to extract the tiles --- You will need to run this 3 times
ptm<-proc.time()
out.lst <- sfClusterApplyLB(1:nrow(tile.tbl), 
                            function(x){ make_LC_tiles(x, tile.tbl) })
proc.time()-ptm

## Get a list of tile files TIFFs
list.tiles.RGB<-list.files(path=here::here("34_twincreek_apiary_t1/3class_parallel/tiled_rgb"))

# Work in parallel to run the classification --- You will be applying your model to each tile
ptm<-proc.time()
out.lst <- sfClusterApplyLB(1:nrow(as.data.frame(list.tiles.RGB)), 
                            function(x){ classify.floral.RF(x) })
proc.time()-ptm

# Make sure to stop the cluster when done
sfStop()

####################################################################### Create mosaic of prediction raster #####################################################

# Must run gdalbuildvrt D:\Jesse\FloralResources\34_twincreek_apiary_t1\3class_parallel\outclass\classifRF.vrt D:\Jesse\FloralResources\34_twincreek_apiary_t1\3class_parallel\outclass\*.tif in OSGeo4w
# This R code below is for converting 32bit raster data to 8bit raster data to be used with "Build Raster Attribute Table" tool
# Before you run this R code you must run this line in OSGro4W Shell to convert .vrt (virtual raster file)
# to a .tif file: gdal_translate D:\Jesse\FloralResources\34_twincreek_apiary_t1\3class_parallel\outclass\classifRF.vrt D:\Jesse\FloralResources\34_twincreek_apiary_t1\3class_output\PredictClass_32float.tif
# After you run this R code you must use the "Build Raster Attribute Table" tool in Arc Pro to populate a attribute table for the .tif file.

mosaico<-terra::rast("34_twincreek_apiary_t1/3class_output/PredictClass_32float.tif")
mosaico

mosaico<-as.factor(mosaico)
is.factor(mosaico)

writeRaster(mosaico, "34_twincreek_apiary_t1/3class_output/Prediction_classification.tif", overwrite=TRUE)

################################################################### Reclassify and mask to isolate flower class #################################################

# 1. Open "34_twincreek_apiary_t1/3class_output/Prediction_classification.tif" in Arc Pro
# 2. "Build raster attribute table" tool on Prediction_classification.tif
# 3. "Reclassify" tool to make all values NA except flower class (flower_output/flower_reclass.tif)

######################################################### Extract by mask to isolate flowers from composite ###############################################

flower_class<-terra::rast("34_twincreek_apiary_t1/flower_output/flower_reclass.tif")
flower_class<-resample(flower_class,ortho, method="near")

ortho<-rast("34_twincreek_apiary_t1/intermediate/odm_orthophoto_crop_3b.tif")
flower_ortho_terra <- terra::mask(ortho, flower_class, inverse=FALSE)
names(flower_ortho_terra)<-c("red","green","blue")
writeRaster(flower_ortho_terra, "34_twincreek_apiary_t1/flower_output/flower_rgb.tif",overwrite=TRUE)

texture8<-rast("34_twincreek_apiary_t1/intermediate/texture8.tif")
flower_texture_terra <- terra::mask(texture8, flower_class, inverse=FALSE)
names(flower_texture_terra)<-c("contrast","dissimilarity","homogeneity","ASM","entropy","mean","variance","correlation")
writeRaster(flower_texture_terra, "34_twincreek_apiary_t1/flower_output/flower_texture.tif",overwrite=TRUE)

rgbInd8<-rast("34_twincreek_apiary_t1/intermediate/rgbInd8.tif")
flower_ind_terra <- terra::mask(rgbInd8, flower_class, inverse=FALSE)
names(flower_ind_terra)<-c("EBI","VDVI","MGVRI","CIVE","NGRDI","RGRI","ExG","RGBVI","BGI","VEG","NGBDI","RGBVI2","RGBVI3")
writeRaster(flower_ind_terra, "34_twincreek_apiary_t1/flower_output/flower_ind.tif",overwrite=TRUE)

# Read in raster data
flower_ortho<-raster::brick("34_twincreek_apiary_t1/flower_output/flower_rgb.tif")
names(flower_ortho)<-c("red","green","blue")
flower_texture<-raster::brick("34_twincreek_apiary_t1/flower_output/flower_texture.tif")
names(flower_texture)<-c("contrast","dissimilarity","homogeneity","ASM","entropy","mean","variance","correlation")
flower_ind<-raster::brick("34_twincreek_apiary_t1/flower_output/flower_ind.tif")
names(flower_ind)<-c("EBI","VDVI","MGVRI","CIVE","NGRDI","RGRI","ExG","RGBVI","BGI","VEG","NGBDI","RGBVI2","RGBVI3")

# Create composite with 8-bit data
flower_composite<-raster::stack(flower_ortho,flower_texture,flower_ind) # here we can create one big object of all the rescaled stack objects (i.e. orthophoto,
names(flower_composite)<-c("red","green","blue",
                           "contrast","dissimilarity","homogeneity","ASM","entropy","mean","variance","correlation",
                           "EBI","VDVI","MGVRI","CIVE","NGRDI","RGRI","ExG","RGBVI","BGI","VEG","NGBDI","RGBVI2","RGBVI3")
################################################################## Subset flower composite for batch processing ################################################

flowers <- read.csv("34_twincreek_apiary_t1/intermediate/trim_training_data.csv")
flowers <-flowers[,c(2:25)]# This may change depending on what classes are decided to go into the model
flower_samples <- flowers[flowers$class ==1 | flowers$class ==2 | flowers$class ==3, ]
flower_samples$class<-as.factor(flower_samples$class) # In this case the column "class" has the values that identify the classes that we are trying to map
flower_samples <- flower_samples[complete.cases(flower_samples), ] # Eliminate all rows that have NA or empty values for any variable
summary(flower_samples)

################################################################### Fit a randomForest classifier ##############################################################

# First, create a randomForest model
con <- file("34_twincreek_apiary_t1/flower_output/full_flower_RF_output.log")
sink(con, append=TRUE)
sink(con, append=TRUE, type="message")

set.seed(123) # Run this before the sampling so that it's reproducible or do not run it if you want different results each time
split1<- sample(c(rep(1, 0.7 * nrow(flower_samples)), rep(0, 0.3 * nrow(flower_samples)))) # Create a vector to split into training ==1 and test ==0
table(split1)
flower.train <- flower_samples[split1 == 1, ] # subset dataset used for initial model fit
flower.test <-  flower_samples[split1 == 0, ] # subset dataset used for independent model validation

summary(as.factor(flower.train$class))
summary(as.factor(flower.test$class))

#### Optimization
jpeg("34_twincreek_apiary_t1/flower_output/RF_mtry.jpeg", type="cairo")
mtry <- tuneRF(flower_samples[-1],flower_samples$class, ntreeTry=500,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
dev.off()

best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m)

################################################# Fit a full randomForest classifier ##############################################################

set.seed(123)
flower_rf_full <-randomForest(class~. ,data=flower.train, mtry=best.m, importance=TRUE,ntree=500)

print(flower_rf_full)
#Evaluate variable importance
importance(flower_rf_full)

jpeg("34_twincreek_apiary_t1/flower_output/full_flower_rf_variable_importance.jpeg", type="cairo")
varImpPlot(flower_rf_full)
dev.off()

varImpPlot(flower_rf_full)

prediction <-predict(flower_rf_full, flower.test)
# Assess the accuracy of our model on a test dataset
confusionMatrix(prediction, flower.test$class)

# Restore output to console
sink() 
sink(type="message")

################################################# Fit a simple randomForest classifier ##############################################################

con <- file("34_twincreek_apiary_t1/flower_output/simple_flower_RF_output.log")
sink(con, append=TRUE)
sink(con, append=TRUE, type="message")

set.seed(123)
flower_rf <-randomForest(class~ASM+entropy+red+green+NGBDI+MGVRI,data=flower.train, mtry=best.m, importance=TRUE,ntree=500)

print(flower_rf)
#Evaluate variable importance
importance(flower_rf)

jpeg("34_twincreek_apiary_t1/flower_output/simple_flower_rf_variable_importance.jpeg", type="cairo")
varImpPlot(flower_rf)
dev.off()

varImpPlot(flower_rf)

prediction <-predict(flower_rf, flower.test)
# Assess the accuracy of our model on a test dataset
confusionMatrix(prediction, flower.test$class)

# Restore output to console
sink() 
sink(type="message")


########################################################### Extrapolate flower class model on 10 acre area of interest ######################################################

dir.create("34_twincreek_apiary_t1/flower_parallel")
dir.create("34_twincreek_apiary_t1/flower_parallel/tiled_rgb")
dir.create("34_twincreek_apiary_t1/flower_parallel/tiled_VIs")
dir.create("34_twincreek_apiary_t1/flower_parallel/tiled_tex")
dir.create("34_twincreek_apiary_t1/flower_parallel/outclass")

# Prepare an RGDAL-based tiling system
# This can be any of your intermediate TIFFs - here I am using the ortho RGB
obj <- GDALinfo(here::here("34_twincreek_apiary_t1/flower_output/flower_rgb.tif")) #extent

## tile to 10 meter blocks with 5% overlap:
tile.lst <- getSpatialTiles(obj, overlap.percent=5, block.x=10, return.SpatialPolygons=TRUE)
tile.tbl <- getSpatialTiles(obj, overlap.percent=5,block.x=10, return.SpatialPolygons=FALSE)

tile.tbl$ID <- as.character(1:nrow(tile.tbl))
head(tile.tbl)

tile.pol <- SpatialPolygonsDataFrame(tile.lst, tile.tbl, match.ID = FALSE)

#
# Prepare functions to run in parallel processing
# This will clip each of the TIFF - You will need to change the input file and output folder each
# time that run the function:: change the location of out.path and Ortho
# Required folder name for ortho == tiled_rgb
# Required folder name for VIs == tiled_VIs
# Required folder name for textires == tiled_tex
# RGDAL-based to clip tiles and write them to disk
make_LC_tiles <- function(i, tile.tbl, 
                          out.path=here::here("34_twincreek_apiary_t1/flower_parallel/tiled_VIs"),
                          Ortho=here::here("34_twincreek_apiary_t1/flower_output/flower_ind.tif")
){
  out.tif = paste0(out.path, "/T_", tile.tbl[i,"ID"], ".tif")
  if(!file.exists(out.tif)){
    m <- readGDAL(Ortho, offset=unlist(tile.tbl[i,c("offset.y","offset.x")]),
                  region.dim=unlist(tile.tbl[i,c("region.dim.y","region.dim.x")]),
                  output.dim=unlist(tile.tbl[i,c("region.dim.y","region.dim.x")]),
                  silent = TRUE)
    m <- as(m, "SpatialPixelsDataFrame")
    writeGDAL(m, out.tif, type="Int16", 
              options="COMPRESS=DEFLATE", mvFlag=-32768)
  }
}

### Parallel function to classify
### You will need to create the required output folder
classify.floral.rf <-function(i, 
                              out.path=here::here("34_twincreek_apiary_t1/flower_parallel/outclass")
){
  nombre<-paste0("T_",i)
  out.tif = paste0(out.path, "/T_",nombre,".tif")
  if(!file.exists(out.tif)){
    print(paste0("Processing...",here::here(paste0("tiled","/",i))))
    #RGB.tile <- terra::rast(here::here(paste0("tiled","/",nombre,".tif")),lyrs=2)
    RGB.tile <- terra::rast(here::here(paste0("34_twincreek_apiary_t1/flower_parallel/tiled_rgb","/",nombre,".tif")))
    names(RGB.tile)<-c("red","green","blue")
    VI.tile <- terra::rast(here::here(paste0("34_twincreek_apiary_t1/flower_parallel/tiled_VIs","/",nombre,".tif")))
    names(VI.tile)<-c("EBI",	"VDVI",	"MGVRI",	"CIVE",	"NGRDI",	"RGRI",	"ExG",
                      "RGBVI",	"BGI",	"VEG",	"NGBDI",	"RGBVI2",	"RGBVI3")
    text.tile <- terra::rast(here::here(paste0("34_twincreek_apiary_t1/flower_parallel/tiled_tex","/",nombre,".tif")),lyrs=c(1:7))
    names(text.tile)<-c("contrast",	"dissimilarity",	"homogeneity",	"ASM",
                        "entropy",	"mean",	"variance")
    
    #writeGDAL(m, out.tif, type="Int16", 
    #          options="COMPRESS=DEFLATE", mvFlag=-32768)
    composite<-c(RGB.tile, VI.tile, text.tile) 
    Rast.Predict <- terra::predict(composite, flower_rf, type='response')
    
    terra::writeRaster(Rast.Predict, out.tif, filetype = "GTiff",
                       overwrite = TRUE)
    
  }
}

## Get a list of tile files TIFFs
list.tiles.RGB<-list.files(path=here::here("34_twincreek_apiary_t1/flower_parallel/tiled_rgb"))

# Parallelization
sfInit(parallel=TRUE, cpus=parallel::detectCores())
# Reload this line every time that you make changes to the functions
sfExport("make_LC_tiles", "tile.tbl","list.tiles.RGB", "classify.floral.rf","flower_rf") # don't forget to include your flower_rf model
sfLibrary(rgdal)
sfLibrary(plyr)
sfLibrary(terra)
sfLibrary(randomForest)

# Work in parallel to extract the tiles --- You will need to run this 3 times
ptm<-proc.time()
out.lst <- sfClusterApplyLB(1:nrow(tile.tbl), 
                            function(x){ make_LC_tiles(x, tile.tbl) })
proc.time()-ptm

#
# Prepare functions to run in parallel processing
# This will clip each of the TIFF - You will need to change the input file and output folder each
# time that run the function:: change the location of out.path and Ortho
# Required folder name for ortho == tiled_rgb
# Required folder name for VIs == tiled_VIs
# Required folder name for textires == tiled_tex
# RGDAL-based to clip tiles and write them to disk
make_LC_tiles <- function(i, tile.tbl, 
                          out.path=here::here("34_twincreek_apiary_t1/flower_parallel/tiled_tex"),
                          Ortho=here::here("34_twincreek_apiary_t1/flower_output/flower_texture.tif")
){
  out.tif = paste0(out.path, "/T_", tile.tbl[i,"ID"], ".tif")
  if(!file.exists(out.tif)){
    m <- readGDAL(Ortho, offset=unlist(tile.tbl[i,c("offset.y","offset.x")]),
                  region.dim=unlist(tile.tbl[i,c("region.dim.y","region.dim.x")]),
                  output.dim=unlist(tile.tbl[i,c("region.dim.y","region.dim.x")]),
                  silent = TRUE)
    m <- as(m, "SpatialPixelsDataFrame")
    writeGDAL(m, out.tif, type="Int16", 
              options="COMPRESS=DEFLATE", mvFlag=-32768)
  }
}

### Parallel function to classify
### You will need to create the required output folder
classify.floral.rf <-function(i, 
                              out.path=here::here("34_twincreek_apiary_t1/flower_parallel/outclass")
){
  nombre<-paste0("T_",i)
  out.tif = paste0(out.path, "/T_",nombre,".tif")
  if(!file.exists(out.tif)){
    print(paste0("Processing...",here::here(paste0("tiled","/",i))))
    #RGB.tile <- terra::rast(here::here(paste0("tiled","/",nombre,".tif")),lyrs=2)
    RGB.tile <- terra::rast(here::here(paste0("34_twincreek_apiary_t1/flower_parallel/tiled_rgb","/",nombre,".tif")))
    names(RGB.tile)<-c("red","green","blue")
    VI.tile <- terra::rast(here::here(paste0("34_twincreek_apiary_t1/flower_parallel/tiled_VIs","/",nombre,".tif")))
    names(VI.tile)<-c("EBI",	"VDVI",	"MGVRI",	"CIVE",	"NGRDI",	"RGRI",	"ExG",
                      "RGBVI",	"BGI",	"VEG",	"NGBDI",	"RGBVI2",	"RGBVI3")
    text.tile <- terra::rast(here::here(paste0("34_twincreek_apiary_t1/flower_parallel/tiled_tex","/",nombre,".tif")),lyrs=c(1:7))
    names(text.tile)<-c("contrast",	"dissimilarity",	"homogeneity",	"ASM",
                        "entropy",	"mean",	"variance")
    
    #writeGDAL(m, out.tif, type="Int16", 
    #          options="COMPRESS=DEFLATE", mvFlag=-32768)
    composite<-c(RGB.tile, VI.tile, text.tile) 
    Rast.Predict <- terra::predict(composite, flower_rf, type='response')
    
    terra::writeRaster(Rast.Predict, out.tif, filetype = "GTiff",
                       overwrite = TRUE)
    
  }
}

# Reload this line every time that you make changes to the functions
sfExport("make_LC_tiles", "tile.tbl","list.tiles.RGB", "classify.floral.rf","flower_rf") # don't forget to include your flower_rf model

# Work in parallel to extract the tiles --- You will need to run this 3 times
ptm<-proc.time()
out.lst <- sfClusterApplyLB(1:nrow(tile.tbl), 
                            function(x){ make_LC_tiles(x, tile.tbl) })
proc.time()-ptm

#
# Prepare functions to run in parallel processing
# This will clip each of the TIFF - You will need to change the input file and output folder each
# time that run the function:: change the location of out.path and Ortho
# Required folder name for ortho == tiled_rgb
# Required folder name for VIs == tiled_VIs
# Required folder name for textires == tiled_tex
# RGDAL-based to clip tiles and write them to disk
make_LC_tiles <- function(i, tile.tbl, 
                          out.path=here::here("34_twincreek_apiary_t1/flower_parallel/tiled_rgb"),
                          Ortho=here::here("34_twincreek_apiary_t1/flower_output/flower_rgb.tif")
){
  out.tif = paste0(out.path, "/T_", tile.tbl[i,"ID"], ".tif")
  if(!file.exists(out.tif)){
    m <- readGDAL(Ortho, offset=unlist(tile.tbl[i,c("offset.y","offset.x")]),
                  region.dim=unlist(tile.tbl[i,c("region.dim.y","region.dim.x")]),
                  output.dim=unlist(tile.tbl[i,c("region.dim.y","region.dim.x")]),
                  silent = TRUE)
    m <- as(m, "SpatialPixelsDataFrame")
    writeGDAL(m, out.tif, type="Int16", 
              options="COMPRESS=DEFLATE", mvFlag=-32768)
  }
}

### Parallel function to classify
### You will need to create the required output folder
classify.floral.rf <-function(i, 
                              out.path=here::here("34_twincreek_apiary_t1/flower_parallel/outclass")
){
  nombre<-paste0("T_",i)
  out.tif = paste0(out.path, "/T_",nombre,".tif")
  if(!file.exists(out.tif)){
    print(paste0("Processing...",here::here(paste0("tiled","/",i))))
    #RGB.tile <- terra::rast(here::here(paste0("tiled","/",nombre,".tif")),lyrs=2)
    RGB.tile <- terra::rast(here::here(paste0("34_twincreek_apiary_t1/flower_parallel/tiled_rgb","/",nombre,".tif")))
    names(RGB.tile)<-c("red","green","blue")
    VI.tile <- terra::rast(here::here(paste0("34_twincreek_apiary_t1/flower_parallel/tiled_VIs","/",nombre,".tif")))
    names(VI.tile)<-c("EBI",	"VDVI",	"MGVRI",	"CIVE",	"NGRDI",	"RGRI",	"ExG",
                      "RGBVI",	"BGI",	"VEG",	"NGBDI",	"RGBVI2",	"RGBVI3")
    text.tile <- terra::rast(here::here(paste0("34_twincreek_apiary_t1/flower_parallel/tiled_tex","/",nombre,".tif")),lyrs=c(1:7))
    names(text.tile)<-c("contrast",	"dissimilarity",	"homogeneity",	"ASM",
                        "entropy",	"mean",	"variance")
    
    #writeGDAL(m, out.tif, type="Int16", 
    #          options="COMPRESS=DEFLATE", mvFlag=-32768)
    composite<-c(RGB.tile, VI.tile, text.tile) 
    Rast.Predict <- terra::predict(composite, flower_rf, type='response')
    
    terra::writeRaster(Rast.Predict, out.tif, filetype = "GTiff",
                       overwrite = TRUE)
    
  }
}

# Reload this line every time that you make changes to the functions
sfExport("make_LC_tiles", "tile.tbl","list.tiles.RGB", "classify.floral.rf","flower_rf") # don't forget to include your flower_rf model

# Work in parallel to extract the tiles --- You will need to run this 3 times
ptm<-proc.time()
out.lst <- sfClusterApplyLB(1:nrow(tile.tbl), 
                            function(x){ make_LC_tiles(x, tile.tbl) })
proc.time()-ptm

## Get a list of tile files TIFFs
list.tiles.RGB<-list.files(path=here::here("34_twincreek_apiary_t1/flower_parallel/tiled_rgb"))

# Work in parallel to run the classification --- You will be applying your model to each tile
ptm<-proc.time()
out.lst <- sfClusterApplyLB(1:nrow(as.data.frame(list.tiles.RGB)), 
                            function(x){ classify.floral.rf(x) })
proc.time()-ptm

# Make sure to stop the cluster when done
sfStop()

####################################################################### Create mosaic of prediction raster #####################################################

# Must run gdalbuildvrt D:\Jesse\FloralResources\34_twincreek_apiary_t1\flower_parallel\outclass\classifRF.vrt D:\Jesse\FloralResources\34_twincreek_apiary_t1\flower_parallel\outclass\*.tif in OSGeo4w
# This R code below is for converting 32bit raster data to 8bit raster data to be used with "Build Raster Attribute Table" tool
# Before you run this R code you must run this line in OSGro4W Shell to convert .vrt (virtual raster file)
# to a .tif file: gdal_translate D:\Jesse\FloralResources\34_twincreek_apiary_t1\flower_parallel\outclass\classifRF.vrt D:\Jesse\FloralResources\34_twincreek_apiary_t1\flower_output\PredictClass_32float.tif

# After you run this R code you must use the "Build Raster Attribute Table" tool in Arc Pro to populate a attribute table for the .tif file.
mosaico<-terra::rast("34_twincreek_apiary_t1/flower_output/PredictClass_32float.tif")
mosaico

mosaico<-as.factor(mosaico)
is.factor(mosaico)

writeRaster(mosaico, "34_twincreek_apiary_t1/flower_output/Flower_prediction_classification.tif", overwrite=TRUE)

################################################################### Build Attribute table and combine prediction rasters #######################################

# 1. Open "34_twincreek_apiary_t1/flower_output/Flower_Prediction_classification.tif" and "34_twincreek_apiary_t1/3class_output/Prediction_classification.tif"in Arc Pro
# 2. "Build raster attribute table" tool on Flower_Prediction_classification.tif
# 3. "Reclassify" tool to turn NODATA into 0 in Flower_Prediction_Classification.tif and 1,2,3 to 10,20,30 in Prediction_classification.tif
# 4. Use raster calculator to combine prediction rasters








# gdal_calc -A D:\Jesse\FloralResources\34_twincreek_apiary_t1\3class_output\Prediction_classification.tif 
# -B D:\Jesse\FloralResources\34_twincreek_apiary_t1\flower_output\Flower_prediction_classification.tif 
# --outfile=D:\Jesse\FloralResources\34_twincreek_apiary_t1\PredictionClassification\FinalPredictionClassification.tif --calc="A+B"

