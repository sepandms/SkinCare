filter <- function(dataset, conditions){
  # conditions should be a dataset
  # conditions <- data.frame(cond1 = setcond1, cond2 = setcond2, ...)
  
  vars <- colnames(conditions)
  data <- dataset
  
  for (i in 1:length(vars)){
    data <- data[ data[,vars[i]] == conditions[1, i] , ]
  }
  
  data
}