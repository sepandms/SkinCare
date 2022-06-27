library(MASS)

get_prior_pars <- function(dataset){
  # Fit Poisson in available age data
  fit.pois <- MASS::fitdistr(dataset$age, densfun="poisson")
  
  # Obtain estimates for lambda
  lambda.fit <- fit.pois$estimate
  names(lambda.fit) <- NULL
  
  lambda.fit.sd <- fit.pois$sd
  names(lambda.fit.sd) <- NULL
  lambda.fit.var <- lambda.fit.sd^2
  
  # Get Prior parameters for lambda ~ Gamma(alpha, beta) 
  # such that
  # E[lambda] <-> lambda.fit
  # Var(lambda) <-> lambda.fit.var
  alpha <- lambda.fit^2 / lambda.fit.var
  beta  <- lambda.fit / lambda.fit.var
  
  list(alpha = alpha, beta = beta)
}