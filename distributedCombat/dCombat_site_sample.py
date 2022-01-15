# Sample code for distributed ComBat

import distributedCombat as dc

# You will need the following variables:
#  - dat: features x subject data matrix for this site
#  - bat: batch identifiers, needs to have same factor levels across sites
#  - mod: covariates to protect in the data, usually output of stats:model.matrix

# first, get summary statistics needed for LS estimation
dc.distributedCombat_site(dat, bat, mod, file="site1_step1.pickle")

# after step 1 at central site, get summary statistics for sigma estimation
dc.distributedCombat_site(
    dat, bat, mod, file="site1_step2.pickle", central_out="central_step1.pickle"
)

# after step 2 at central site, get harmonized data
dc.distributedCombat_site(
    dat, bat, mod, file="site1_harmonized_data.pickle", central_out="central_step2.pickle"
)
