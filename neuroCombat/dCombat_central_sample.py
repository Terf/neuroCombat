# Sample code for distributed ComBat, central location

import neuroCombat as nc

# Include outputs from individual sites, can include any number of sites
# Make sure to include site outputs that are on the same step

distributedCombat_central(
    ["site1_step1.Rdata", "site2_step1.Rdata"], file="central_step1.Rdata"
)

distributedCombat_central(
    ["site1_step2.Rdata", "site2_step2.Rdata"], file="central_step2.Rdata"
)
