run_compilation <- function(receptor_csv, main_data_csv,  signaturizer, chemberta, mol2vec, graph2vec, mordred, protr, prott5, protbert, mf02, mf04, mf06, mf08, mf09,  mf10, mf11, output_prefix) {

    # Feature Processing ####

    ## Ligands ####

    ligPaths <- c("Signaturizer" = signaturizer,
                "ChemBERTa" = chemberta,
                "Mol2Vec" = mol2vec,
                "Graph2Vec" = graph2vec,
                "Mordred" = mordred)

    # Load the Ligand Features, and Handle Missing Values

    LigandFeatures <- list()
    badLigs <- list()

    for (i in names(ligPaths)) {
    rawFeatFile <- read.csv(file = file.path(ligPaths[[i]]))

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

    print(paste("Processed ligand features for:", i))

    rm(rawFeatFile, i)

    }

    # Combine all the files into one
    ligsCommon <- purrr::reduce(LigandFeatures, dplyr::inner_join, by = c("Ligand_ID", "SMILES"))
    ligsCommon <- ligsCommon[,1:2]


    for (i in names(LigandFeatures)) {
    procFeatFile <- LigandFeatures[[i]]
    procFeatFile <- procFeatFile[procFeatFile$Ligand_ID %in% ligsCommon$Ligand_ID,]
    write.csv(procFeatFile, file = paste0(output_prefix, "_", i, "_Final.csv"), row.names = FALSE)
    rm(i, procFeatFile)
    }

    ## Receptors ####

    recsInput <- read.csv(file = receptor_csv)

    #### Math Feature ####

    # Compile all features into one

    mfFiles <- c(mf02, mf04, mf06, mf08, mf09, mf10, mf11)
        mfNames <- c("MF_02", "MF_04", "MF_06", "MF_08", "MF_09", "MF_10", "MF_11")

    mfDesc <- data.frame(matrix(nrow = nrow(recsInput), ncol = 0))

    for (idx in 1:length(mfFiles)) {
          i_file <- mfFiles[idx]
          i_name <- mfNames[idx]
          
          # Read the file path variable
          a <- read.csv(file = i_file, header = TRUE)
    
          if (nrow(a) == 0) {
            stop("MathFeature files are empty! Re-run MathFeature. Stopping execution.")
    }
    
    # make the Receptor IDs row headers
    row.names(a) <- a[,1]
    
    # remove the last column (labels) from each file and the first column (receptor ids)
    a <- a[,c(2:(ncol(a)-1))]
    
    # assign them column headers
    names(a) <- paste0(i_name, "_", 1:ncol(a))
    mfDesc <- cbind(mfDesc, a)
          
    rm(a, i_file, i_name)
   }

    mfDesc <- cbind("Receptor_ID" = row.names(mfDesc), mfDesc)
    
    # 6. Read files *directly* from arguments
    recPaths <- c("ProtR" = protr,
                  "ProtT5" = prott5,
                  "ProtBERT" = protbert,
                  "MathFeature" = "") # This is handled below

    ReceptorFeatures <- list()
    badRecs <- list()

    for (i in names(recPaths)) {
      if (i == "MathFeature") {
          rawFeatFile <- mfDesc
      } else {
          # Read the file path variable
          rawFeatFile <- read.csv(file = recPaths[[i]])
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
       # Write output to the local directory
       write.csv(procFeatFile, file = paste0(output_prefix, "_", i, "_Final.csv"), row.names = FALSE)
       rm(i, procFeatFile)
     }


    # Main Data ####
	mainData <- read.csv(file = main_data_csv)
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

	write.csv(mainData_user, file = paste0(output_prefix, "_Input_ID_Information.csv"), row.names = FALSE)
    write.csv(mainData, file = paste0(output_prefix, "_mainData.csv"), row.names = FALSE)

    print("Code ran successfully")
}

# It reads all the file paths from the Nextflow script
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 18) {
  stop("This script requires 18 file path arguments.", call.=FALSE)
}

# Call the main function with all the arguments
run_compilation(
  receptor_csv = args[1],
  main_data_csv = args[2],
  signaturizer = args[3],
  chemberta = args[4],
  mol2vec = args[5],
  graph2vec = args[6],
  mordred = args[7],
  protr = args[8],
  prott5 = args[9],
  protbert = args[10],
  mf02 = args[11],
  mf04 = args[12],
  mf06 = args[13],
  mf08 = args[14],
  mf09 = args[15],
  mf10 = args[16],
  mf11 = args[17],
  output_prefix = args[18]
)
