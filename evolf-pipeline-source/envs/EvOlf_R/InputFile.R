library(readr)
library(dplyr)
library(purrr)

run_input_prep <- function(inputFile, 
                           ligSmilesColumn, recSeqColumn, 
                           ligID, recID, lrID) {

    # take input file from user
    mainData <- read.csv(inputFile)
    # add serial numbers to help in sorting
    mainData$SrNum <- 1:nrow(mainData)



    # Column Checks ####

    # check if ligand smiles and receptor smiles are present in the main data file or not
    if(ligSmilesColumn %in% names(mainData)) {
    # rename the column to SMILES
    names(mainData)[names(mainData) == ligSmilesColumn] <- 'SMILES'
    # print("already present")
    } else {
    stop("Ligand SMILES column absent!\n")
    }

    if(recSeqColumn %in% names(mainData)) {
    # rename the column to Sequence
    names(mainData)[names(mainData) == recSeqColumn] <- 'Sequence'
    # print("already present")
    } else {
    stop("Sequence column absent!\n")
    }

    # check if ligID column is provided or not
    if (!(is.null(ligID))) {
    # if column provided, rename column
    if (ligID %in% names(mainData)) {
        # rename the column to Sequence
        names(mainData)[names(mainData) == ligID] <- 'Ligand_ID'
        # print("already present")
    } else {
        stop("Ligand_ID column absent!\n")
    }
    } 

    if (!(is.null(recID))) {
    # if column provided, rename column
    if (recID %in% names(mainData)) {
        # rename the column to Sequence
        names(mainData)[names(mainData) == recID] <- 'Receptor_ID'
        # print("already present")
    } else {
        stop("Receptor_ID column absent!\n")
    }
    } 

    if (!(is.null(lrID))) {
    # if column provided, rename column
    if (lrID %in% names(mainData)) {
        # rename the column to Sequence
        names(mainData)[names(mainData) == lrID] <- 'IDs'
        # print("already present")
    } else {
        stop("Ligand-Receptor Pair ID column absent!\n")
    }
    }


    # Ligand Receptor IDs

    if (is.null(lrID)) { # if no id column provided
    # add a unique identifier to each lr pair
    mainData$IDs <- paste0("LR", 1:nrow(mainData))
    # print("lr ids added")
    } else { # if column provided
    # check if all ids are unique or not
    if (length(unique(mainData$IDs)) != nrow(mainData)) {
        stop("Duplicates present in Ligand-Receptor Pair IDs! Each LR pair should have a unique ID. Stopping execution.")
    }
    }

    # Ligands
    if (is.null(ligID)) { # if no id column provided
    # get the list of unique ligands
    ligsData <- data.frame("SMILES" = unique(mainData$SMILES))
    # add a unique identifier to each SMILES
    ligsData$Ligand_ID <- paste0("L", 1:nrow(ligsData))
    # rearrange columns
    ligsData <- ligsData[,c("Ligand_ID", "SMILES")]
    # print("l ids added")
    } else { # else use the existing ligand id
    ligsData <- data.frame(unique(mainData[,c("Ligand_ID", "SMILES")]))
    # reset the row indexes
    row.names(ligsData) <- NULL
    # print("existing ids used")
    
    # check if provided ids are unique or not
    if (length(unique(ligsData$Ligand_ID)) != length(unique(ligsData$SMILES))) {
        stop("Mismatch! The number of unique Ligand_IDs does not match the number of unique SMILES. Stopping execution.")
    }
    }

    # merge this file with the main Data to add information about the ligand IDs
    if (is.null(ligID)) { # if no id column provided
    mainData <- merge(mainData, ligsData, by = "SMILES", sort = FALSE)
    } else {
    mainData <- merge(mainData, ligsData, by = c("Ligand_ID", "SMILES"), sort = FALSE)
    }



    # Receptors
    if (is.null(recID)) { # if no receptor id column provided
    # get the list of unique receptor
    recsData <- data.frame("Sequence" = unique(mainData$Sequence))
    # add a unique identifier to each Sequence
    recsData$Receptor_ID <- paste0("R", 1:nrow(recsData))
    # rearrange columns
    recsData <- recsData[,c("Receptor_ID", "Sequence")]
    # print("r ids added")
    } else { # else use the existing ligand id
    # check if all ids are unique or not
    recsData <- data.frame(unique(mainData[,c("Receptor_ID", "Sequence")]))
    # reset the row indexes
    row.names(recsData) <- NULL
    # print("existing ids used")
    
    if (length(unique(recsData$Receptor_ID)) != length(unique(recsData$Sequence))) {
        stop("Mismatch! The number of unique Receptor_IDs does not match the number of unique Sequences Stopping execution.")
    }
    }



    # merge this file with the main Data to add information about the Receptor IDs
    if (is.null(recID)) { # if no id column provided
    mainData <- merge(mainData, recsData, by = "Sequence", sort = FALSE)
    } else {
    mainData <- merge(mainData, recsData, by = c("Receptor_ID", "Sequence"), sort = FALSE)
    }

    # sort main data file back to original order
    mainData <- mainData[order(mainData$SrNum),]
    # reset row indexes to prevent future problems
    row.names(mainData) <- NULL

    # rearrange columns
    mainData <- mainData[,c("SrNum", "IDs", "Ligand_ID", "SMILES", "Receptor_ID", "Sequence")]

    write.csv(mainData, file = "mainData_01.csv", row.names = FALSE)
    write.csv(ligsData, file = "ligsData.csv", row.names = FALSE)
    write.csv(recsData, file = "recsData.csv", row.names = FALSE)

    # make fasta file for receptor

    # function to create fasta files:
    dat2fasta <- function(id, sequence, outfile = "recsData.fasta"){
    writeLines(paste(">", as.character(id), "\n", as.character(sequence), sep = ""), outfile)
    }

    dat2fasta(id = recsData$Receptor_ID, sequence = recsData$Sequence)

    print("Code ran successfully")
}


args <- commandArgs(trailingOnly = TRUE)

if (length(args)!=6) {
  stop("Provide all the required arguements as per the functions definition", call.=FALSE)
}

run_input_prep(inputFile = args[1], 
               ligSmilesColumn = args[2], recSeqColumn = args[3],
               ligID = args[4], recID = args[5], lrID = args[6])