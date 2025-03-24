setwd("/storage1/aayushim/EvOlf_3.3/01_Dry_Lab/02_Descriptors/31_ASGPCRs/")

library(protr)

# 01 - load the receptors information:
rec <- read.csv("../../01_Data/16_ArunShukla_GPCRs/ASGPCRs_Final.csv")
rec <- rec[,c("Receptor_ID", "Sequence")] # remove unwanted columns

# 02 - convert the sequences to a list
x <- as.list(rec$Sequence)
names(x) <- rec$Receptor_ID
length(x)

# 03 - calculate protr descriptors ####

### to check if the file has any unrecognized amino acid
x <- x[(sapply(x, protcheck))]
length(x)

# list of all the features that we want to compute
features <- c("extractAAC", "extractDC", "extractTC", 
              "extractMoreauBroto", "extractMoran", "extractGeary",
              "extractCTDC", "extractCTDT", "extractCTDD", 
              "extractCTriad", "extractSOCN", "extractQSO", 
              "extractPAAC", "extractAPAAC")

# Create an empty dataframe
finalFeautures <- data.frame(matrix(ncol = 0, nrow = length(x)))

for (i in features) {
  # Compute the descriptor using protr functions 
  a <- data.frame(t(sapply(x, i)))
  
  # combine the results into a single file
  finalFeautures <- cbind(finalFeautures, a)
  
  print(paste0(i, " Done"))
}

# add Receptor_ID information
finalFeautures <- data.frame(Receptor_ID = row.names(finalFeautures), finalFeautures)

# saveRDS(finalFeautures, file = "01_EvOlf_Protr_Descriptors.rds")
write.csv(finalFeautures, file = "Raw_ProtR.csv", row.names = FALSE)

print("Code ran successfully")