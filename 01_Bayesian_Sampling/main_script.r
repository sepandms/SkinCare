# Pseudo-Random number seed
set.seed(3)

# Library Used
# install.packages("MASS")

# Load functions from scripts
source("age_sampler.r")

# Load data
file <- "no_duplicates"
path <- paste("Dataset/", file, ".csv", sep = "")
data <- read.csv(path)

# Separate data in 
data.full <- data[!is.na(data$age), ] # Complete data
data.miss <- data[is.na(data$age),  ] # Data with NAs

# Simulate age for each data.miss entry
simulated_ages <- apply(data.miss, 1, function(i) {
  age_sampler(data.full, data.frame(dx = i["dx"], 
                                    dx_type = i["dx_type"], 
                                    sex = i["sex"], 
                                    localization = i["localization"]))})

# Add simulated age to data.miss data frame
data.miss$age <- simulated_ages

# Join data
data.processed <- rbind(data.full, data.miss)

# Export csv
write.csv(data.processed, file=paste("Dataset/", file, "_no_NAs.csv", sep = ""))
