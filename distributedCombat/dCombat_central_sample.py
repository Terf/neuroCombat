# Sample code for distributed ComBat, central location

import distributedCombat as dc

# Include outputs from individual sites, can include any number of sites
# Make sure to include site outputs that are on the same step

dc.distributedCombat_central(
    ["site1_step1.pickle", "site2_step1.pickle"], file="central_step1.pickle"
)

dc.distributedCombat_central(
    ["site1_step2.pickle", "site2_step2.pickle"], file="central_step2.pickle"
)
