# Sample code for distributed ComBat

import neuroCombat as nc

# You will need the following variables:
#  - dat: features x subject data matrix for this site
#  - bat: batch identifiers, needs to have same factor levels across sites
#  - mod: covariates to protect in the data, usually output of stats:model.matrix

# first, get summary statistics needed for LS estimation
distributedCombat_site(dat, bat, mod, file="site1_step1.Rdata")

# after step 1 at central site, get summary statistics for sigma estimation
distributedCombat_site(
    dat, bat, mod, file="site1_step2.Rdata", central_out="central_step1.Rdata"
)

# after step 2 at central site, get harmonized data
distributedCombat_site(
    dat, bat, mod, file="site1_harmonized_data.Rdata", central_out="central_step2.Rdata"
)
