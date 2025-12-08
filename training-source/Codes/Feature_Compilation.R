tempPath <- "/storage1/aayushim/EvOlf_3.3/01_Dry_Lab/15_Colab/Sample_Run_01/Temp_Folder"
outPath <- "/storage1/aayushim/EvOlf_3.3/01_Dry_Lab/15_Colab/Sample_Run_01"

# Feature Processing ####

## Ligands ####

ligPaths <- c("Signaturizer" = "Raw_Signaturizer.csv",
              "ChemBERTa" = "Raw_ChemBERTa.csv",
              "Mol2Vec" = "Raw_Mol2Vec.csv",
              "Graph2Vec" = "Raw_Graph2Vec.csv",
              "Mordred" = "Raw_Mordred.csv")

# Load the Ligand Features, and Handle Missing Values

LigandFeatures <- list()
badLigs <- list()

for (i in names(ligPaths)) {
  rawFeatFile <- read.csv(file = file.path(tempPath, ligPaths[[i]]))

  # identify rows that have missing values in all feature columns excluding ligand id and smiles column, and remove them
  badLigs[[i]] <- rowSums(is.na(rawFeatFile)) >= ncol(rawFeatFile)-2 
  rawFeatFile <- rawFeatFile[!badLigs[[i]],]
  
  # Identify columns with all NAs, and replace NAs with 0
  colsMissingValueAll <- sapply(rawFeatFile, function(col) all(is.na(col)))
  rawFeatFile[, colsMissingValueAll] <- 0
  
  # Identify columns with some missing values, and replace with column mean
  colsMissingValue <- sapply(rawFeatFile, function(col) any(is.na(col)))
  rawFeatFile[, colsMissingValue] <- lapply(rawFeatFile[, colsMissingValue], function(col) {
    col[is.na(col)] <- mean(col, na.rm = TRUE)
    return(col)
  })
  
  # Mordred specific pre-processing
  if (i == "Mordred") {
    rawFeatFile$Lipinski <- ifelse(rawFeatFile$Lipinski == "True", 1, 0)
    rawFeatFile$GhoseFilter <- ifelse(rawFeatFile$GhoseFilter == "True", 1, 0)
  }
  
  LigandFeatures[[i]] <- rawFeatFile
  rm(rawFeatFile, i)
}

# Combine all the files into one
ligsCommon <- purrr::reduce(LigandFeatures, dplyr::inner_join, by = c("Ligand_ID", "SMILES"))
ligsCommon <- ligsCommon[,1:2]


for (i in names(LigandFeatures)) {
  procFeatFile <- LigandFeatures[[i]]
  procFeatFile <- procFeatFile[procFeatFile$Ligand_ID %in% ligsCommon$Ligand_ID,]
  write.csv(procFeatFile, file = paste0(tempPath, "/", i, "_Final.csv"), row.names = FALSE)
  rm(i, procFeatFile)
}




## Receptors ####

recsInput <- read.csv(file = file.path(tempPath, "recsData.csv"))

#### Math Feature ####

# Compile all features into one

mfFiles <- c("MF_02.csv", "MF_04.csv", "MF_06.csv", "MF_08.csv", "MF_09.csv", "MF_10.csv", "MF_11.csv")

mfDesc <- data.frame(matrix(nrow = nrow(recsInput), ncol = 0))

for (i in mfFiles) {
  # read the file
  a <- read.csv(file = paste0(tempPath,"/", i), header = TRUE)
  
  # check if file has at least one row
  if (nrow(a) == 0) {
    stop("MathFeature files are empty! Re-run MathFeature. Stopping execution.")
  }
  
  # make the Receptor IDs row headers
  row.names(a) <- a[,1]
  
  # remove the last column (labels) from each file and the first column (receptor ids)
  a <- a[,c(2:(ncol(a)-1))]
  
  # assign them column headers
  newName <- stringr::str_replace_all(i, ".csv", "")
  names(a) <- paste0(newName, "_", 1:ncol(a))
  
  # combine them into a single file
  mfDesc <- cbind(mfDesc, a)
  
  rm(a, i)
}

mfDesc <- cbind("Receptor_ID" = row.names(mfDesc), mfDesc)



# Load the Ligand Features, and Handle Missing Values

recPaths <- c("ProtR" = "Raw_ProtR.csv",
              "ProtT5" = "Raw_ProtT5.csv",
              "ProtBERT" = "Raw_ProtBERT.csv",
              "MathFeature" = "")

ReceptorFeatures <- list()
badRecs <- list()

for (i in names(recPaths)) {
  # Mordred specific pre-processing
  if (i == "MathFeature") {
    rawFeatFile <- mfDesc
  } else {
    rawFeatFile <- read.csv(file = file.path(tempPath, recPaths[[i]]))
  }
  
  # identify rows that have missing values in all feature columns excluding Receptor id column, and remove them
  badRecs[[i]] <- rowSums(is.na(rawFeatFile)) >= ncol(rawFeatFile)-1 
  rawFeatFile <- rawFeatFile[!badRecs[[i]],]
  
  # Identify columns with all NAs, and replace NAs with 0
  colsMissingValueAll <- sapply(rawFeatFile, function(col) all(is.na(col)))
  rawFeatFile[, colsMissingValueAll] <- 0
  
  # Identify columns with some missing values, and replace with column mean
  colsMissingValue <- sapply(rawFeatFile, function(col) any(is.na(col)))
  rawFeatFile[, colsMissingValue] <- lapply(rawFeatFile[, colsMissingValue], function(col) {
    col[is.na(col)] <- mean(col, na.rm = TRUE)
    return(col)
  })
  
  ReceptorFeatures[[i]] <- rawFeatFile
  rm(rawFeatFile, i)
}

# Combine all the files into one
recsCommon <- purrr::reduce(ReceptorFeatures, dplyr::inner_join, by = "Receptor_ID")
recsCommon <- recsCommon[,1:2]


for (i in names(ReceptorFeatures)) {
  procFeatFile <- ReceptorFeatures[[i]]
  procFeatFile <- procFeatFile[procFeatFile$Receptor_ID %in% recsCommon$Receptor_ID,]
  write.csv(procFeatFile, file = paste0(tempPath, "/", i, "_Final.csv"), row.names = FALSE)
  rm(i, procFeatFile)
}


# Main Data ####
mainData <- read.csv(file = file.path(tempPath, "mainData_01.csv"))
mainData_user <- mainData

# merge this main data with ligs Common and recsCommon to remove datapoints that are not being processed
mainData <- merge(mainData, ligsCommon, by = c("Ligand_ID", "SMILES"), sort = FALSE)
mainData <- merge(mainData, recsCommon, by = "Receptor_ID", sort = FALSE)

# remove the unwanted column
mainData$A <- NULL

# sort main data file back to original order
mainData <- mainData[order(mainData$SrNum),]

# rearrange columns
mainData <- mainData[,c("IDs", "Ligand_ID", "SMILES", "Receptor_ID", "Sequence")]

# give the user information about 
mainData_user$ProcessingStatus <- ifelse(mainData_user$IDs %in% mainData$IDs, "Processed", "Not Processed")

write.csv(mainData_user, file = file.path(outPath, "Input_ID_Information.csv"), row.names = FALSE)
write.csv(mainData, file = file.path(tempPath, "mainData.csv"), row.names = FALSE)

print("Code ran successfully")