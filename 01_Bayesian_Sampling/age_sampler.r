age_sampler <- function(all.available.datapoints, conditioned.on){
  # Load functions from scripts
  source("get_prior_pars.r")
  source("filter.r")
  
  # Compute prior parameters using all data available
  prior <- get_prior_pars(all.available.datapoints)
  
  # Filter data according to the observed variables
  filtered <- filter(all.available.datapoints, conditioned.on)
  
  # Compute sum.age and n.observations
  sum.age <- sum(filtered$age)
  n.observations <- length(filtered$age)
  
  # Sample lambda_k from conjugate posterior
  lambda_k <- rgamma(1, shape = prior$alpha + sum.age, rate = prior$beta + n.observations)
  
  # Sample age from data model
  rpois(1, lambda_k)
}