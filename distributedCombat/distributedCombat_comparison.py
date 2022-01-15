import pandas as pd
import numpy as np
from rpy2.robjects import r, numpy2ri, pandas2ri
import neuroCombat as nc
import distributedCombat as dc


numpy2ri.activate()
pandas2ri.activate()
r(
    """
library(matrixStats)
# Simulate data
set.seed(8888)

p=10000 # Number of features
q=3 # Number of covariates
n=100
batch = rep(1:4, n/4) #Batch variable for the scanner id
batch <- as.factor(batch)

mod = matrix(runif(q*n), n, q) #Random design matrix
dat = matrix(runif(p*n), p, n) #Random data matrix
"""
)

dat = pd.DataFrame(r["dat"])
mod = pd.DataFrame(r["mod"])
batch_col = "batch"
covars = pd.DataFrame({batch_col: r["batch"]})

#### Ground truth ComBat outputs ####
com_out = nc.neuroCombat(dat, covars, batch_col)
com_out_ref = nc.neuroCombat(dat, covars, batch_col, ref_batch="1")

#### Distributed ComBat: No reference batch ####
### Step 1
site_outs = []
for b in covars[batch_col].unique():
    s = list(map(lambda x: x == b, covars[batch_col]))
    df = dat.loc[:, s]
    bat = covars[batch_col][s]
    x = mod.loc[s, :]
    f = "site_out_" + str(b) + ".pickle"
    out = dc.distributedCombat_site(df, bat, x, verbose=True, file=f)
    site_outs.append(f)

central = dc.distributedCombat_central(site_outs)


### Step 2
site_outs = []
for b in covars[batch_col].unique():
    s = list(map(lambda x: x == b, covars[batch_col]))
    df = dat.loc[:, s]
    bat = covars[batch_col][s]
    x = mod.loc[s, :]
    f = "site_out_" + str(b) + ".pickle"
    out = dc.distributedCombat_site(
        df, bat, x, verbose=True, central_out=central, file=f
    )
    site_outs.append(f)

central = dc.distributedCombat_central(site_outs)

### Compare distributed vs original
site_outs = []
error = []
perror = []  # percent difference
for b in covars[batch_col].unique():
    s = list(map(lambda x: x == b, covars[batch_col]))
    df = dat.loc[:, s]
    bat = covars[batch_col][s]
    x = mod.loc[s, :]
    f = "site_out_" + str(b) + ".pickle"
    out = dc.distributedCombat_site(df, bat, x, central_out=central, file=f)
    site_outs.append(f)
    print(
        "ground truth", np.allclose(out["dat_combat"], com_out["data"][:, s], atol=0.1)
    )
    error.append(com_out["data"][:, s] - out["dat_combat"])
    perror.append(
        abs(com_out["data"][:, s] - out["dat_combat"]) / com_out["data"][:, s]
    )


print("error", np.array(error))
print("perror", np.array(perror))


print("===============================================================")


#### Distributed ComBat: With reference batch ####
### Step 1
site_outs = []
for b in covars[batch_col].unique():
    s = list(map(lambda x: x == b, covars[batch_col]))
    df = dat.loc[:, s]
    bat = covars[batch_col][s]
    x = mod.loc[s, :]
    f = "site_out_" + str(b) + ".pickle"
    out = dc.distributedCombat_site(df, bat, x, verbose=True, file=f, ref_batch="1")
    site_outs.append(f)

central = dc.distributedCombat_central(site_outs, ref_batch="1")

### Step 2
site_outs = []
for b in covars[batch_col].unique():
    s = list(map(lambda x: x == b, covars[batch_col]))
    df = dat.loc[:, s]
    bat = covars[batch_col][s]
    x = mod.loc[s, :]
    f = "site_out_" + str(b) + ".pickle"
    out = dc.distributedCombat_site(
        df, bat, x, verbose=True, central_out=central, file=f, ref_batch="1"
    )
    site_outs.append(f)

central = dc.distributedCombat_central(site_outs, ref_batch="1")

### Compare distributed vs original
site_outs = []
error = []
perror = []  # percent difference
for b in covars[batch_col].unique():
    s = list(map(lambda x: x == b, covars[batch_col]))
    df = dat.loc[:, s]
    bat = covars[batch_col][s]
    x = mod.loc[s, :]
    f = "site_out_" + str(b) + ".pickle"
    out = dc.distributedCombat_site(
        df, bat, x, central_out=central, file=f, ref_batch="1"
    )
    site_outs.append(f)
    print(
        "ground truth", np.allclose(out["dat_combat"], com_out["data"][:, s], atol=0.5)
    )
    error.append(com_out["data"][:, s] - out["dat_combat"])
    perror.append(
        abs(com_out["data"][:, s] - out["dat_combat"]) / com_out["data"][:, s]
    )

print("error", np.array(error))
print("perror", np.array(perror))
