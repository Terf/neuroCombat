import pandas as pd
import numpy as np
from rpy2.robjects import r, numpy2ri, pandas2ri
from rpy2.rinterface_lib import callbacks

# from rpy2.robjects.packages import importr
import neuroCombat as nc
import distributedCombat as dc

# import neuroCombat.distributedCombat as dc
import sys


def null(x):
    pass


numpy2ri.activate()
pandas2ri.activate()
callbacks.consolewrite_print = null
r(
    """
library(neuroCombat)
library(matrixStats)
source("/opt/dCombatR/distributedCombat.R")
source("/opt/dCombatR/neuroComBat_helpers.R")
source("/opt/dCombatR/neuroComBat.R")

# Simulate data
set.seed(8888)

p=10000 # Number of features
q=3 # Number of covariates
n=100
batch = rep(1:4, n/4) #Batch variable for the scanner id
batch <- as.factor(batch)

mod = matrix(runif(q*n), n, q) #Random design matrix
dat = matrix(runif(p*n), p, n) #Random data matrix

#### Ground truth ComBat outputs ####
com_out <- neuroCombat(dat, batch, mod)
com_out_ref <- neuroCombat(dat, batch, mod, ref.batch = "1")
"""
)

dat = pd.DataFrame(r["dat"])
mod = pd.DataFrame(r["mod"])
batch_col = "batch"
covars = pd.DataFrame({batch_col: r["batch"]})

#### Ground truth ComBat outputs ####
com_out = nc.neuroCombat(dat, covars, batch_col)
com_out_ref = nc.neuroCombat(dat, covars, batch_col, ref_batch="1")

r(
    """
#### Distributed ComBat: No reference batch ####
### Step 1
site.outs <- NULL
for (b in unique(batch)) {
  s <- batch == b
  df <- dat[,s]
  bat <- batch[s]
  x <- mod[s,]
  
  site.outs <- c(site.outs, list(distributedCombat_site(df, bat, x)))
}
central <- distributedCombat_central(site.outs)
"""
)
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
print("Step 1 comparisons:")
print(np.allclose(r["central"].rx("stand.mean"), central["stand_mean"]))
print(np.allclose(r["central"].rx("B.hat"), central["B_hat"]))
# print(r["central"].rx("var.pooled"), central["var_pooled"])


r(
    """
### Step 2
site.outs <- NULL
for (b in unique(batch)) {
  s <- batch == b
  df <- dat[,s]
  bat <- batch[s]
  x <- mod[s,]
  
  site.outs <- c(site.outs, list(distributedCombat_site(df, bat, x,
                                                        central.out = central)))
}
central <- distributedCombat_central(site.outs)
"""
)
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
print("Step 2 comparisons:")
print(np.allclose(r["central"].rx("stand.mean"), central["stand_mean"]))
print(np.allclose(r["central"].rx("B.hat"), central["B_hat"]))

### Compare distributed vs original
site_outs = []
error = []
perror = []  # percent difference
r(
    """
### Compare distributed vs original
site.outs <- NULL
error <- NULL
perror <- NULL # percent difference
"""
)
for b in covars[batch_col].unique():
    r(
        """
    b <- {0}
    s <- batch == b
    df <- dat[,s]
    bat <- batch[s]
    x <- mod[s,]

    #site.out <- distributedCombat_site(df, bat, x, central.out = central)
    old.dat <- dat
    dat <- df
    old.batch <- batch
    batch <- bat
    old.mod <- mod
    mod <- x
    central.out <- central
    ref.batch=NULL
    eb=TRUE
    parametric=TRUE
    mean.only=FALSE
    verbose=TRUE
    file=NULL
    """.format(
            b
        )
    )
    s = list(map(lambda x: x == b, covars[batch_col]))
    df = dat.loc[:, s]
    bat = covars[batch_col][s]
    x = mod.loc[s, :]
    f = "site_out_" + str(b) + ".pickle"
    out = dc.distributedCombat_site(df, bat, x, central_out=central, file=f, debug=r)
    r(
        """
    dat <- old.dat
    batch <- old.batch
    mod <- old.mod


    site.outs <- c(site.outs, site.out)
    estimates <- site.out$estimates
    
    error <- c(error, max(c(com_out$dat.combat[,s] - site.out$dat.combat)))
    perror <- c(perror,
                max(c(abs(com_out$dat.combat[,s] - site.out$dat.combat)/
                        site.out$dat.combat)))
    """
    )
    print("Site comparisons:")
    print(
        "dat_combat",
        np.allclose(
            np.array(r["site.out"].rx("dat.combat")), np.array(out["dat_combat"])
        ),
    )
    print(
        "gamma_hat",
        np.array(r["estimates"].rx("gamma.hat")).shape,
        out["estimates"]["gamma_hat"].shape,
    )
    print(
        "gamma_hat",
        np.allclose(r["estimates"].rx("gamma.hat"), out["estimates"]["gamma_hat"]),
    )
    print(
        "delta_hat",
        np.allclose(r["estimates"].rx("delta.hat"), out["estimates"]["delta_hat"]),
    )
    print(
        "gamma_star",
        np.allclose(r["estimates"].rx("gamma.star"), out["estimates"]["gamma_star"]),
    )
    print(
        "delta_star",
        np.allclose(r["estimates"].rx("delta.star"), out["estimates"]["delta_star"]),
    )
    print(
        "gamma_bar",
        np.allclose(r["estimates"].rx("gamma.bar"), out["estimates"]["gamma_bar"]),
    )
    print("t2", np.allclose(r["estimates"].rx("t2"), out["estimates"]["t2"]))
    print(
        "a_prior",
        np.allclose(r["estimates"].rx("a.prior"), out["estimates"]["a_prior"]),
    )
    print(
        "b_prior",
        np.allclose(r["estimates"].rx("b.prior"), out["estimates"]["b_prior"]),
    )
    print(
        "stand_mean",
        np.allclose(r["estimates"].rx("stand.mean"), out["estimates"]["stand_mean"]),
    )
    print(
        "mod_mean",
        np.allclose(r["estimates"].rx("mod.mean"), out["estimates"]["mod_mean"]),
    )
    print(
        "var_pooled",
        np.allclose(r["estimates"].rx("var.pooled"), out["estimates"]["var_pooled"]),
    )
    print(
        "beta_hat",
        np.allclose(r["estimates"].rx("beta.hat"), out["estimates"]["beta_hat"]),
    )
    print("mod", np.allclose(r["estimates"].rx("mod"), out["estimates"]["mod"]))
    # print('batch', np.allclose(r["estimates"].rx("batch"), out["estimates"]["batch"]))
    # print('ref_batch', np.allclose(r["estimates"].rx("ref.batch"), out["estimates"]["ref_batch"]))
    print("eb", np.allclose(r["estimates"].rx("eb"), out["estimates"]["eb"]))
    print(
        "parametric",
        np.allclose(r["estimates"].rx("parametric"), out["estimates"]["parametric"]),
    )
    print(
        "mean_only",
        np.allclose(r["estimates"].rx("mean.only"), out["estimates"]["mean_only"]),
    )
    print("site", b)
    site_outs.append(f)
    # error.append(com_out['data'][:,s] - out["dat_combat"])
    # perror.append(abs(com_out['data'][:,s] - out["dat_combat"]) / com_out['data'])

# print("ERROR", len(error), error[0])
# print(error, perror)

# sys.exit(0)
print("with ref batch")

r(
    """
#### Distributed ComBat: With reference batch ####
### Step 1
site.outs <- NULL
for (b in unique(batch)) {
  s <- batch == b
  df <- dat[,s]
  bat <- batch[s]
  x <- mod[s,]
  
  site.outs <- c(site.outs, list(distributedCombat_site(df, bat, x, ref.batch = "1")))
}
central <- distributedCombat_central(site.outs, ref.batch = "1")

"""
)

# sys.exit(1)
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
print("Step 1 (ref batch) comparisons:")
print(np.allclose(r["central"].rx("stand.mean"), central["stand_mean"]))
print(np.allclose(r["central"].rx("B.hat"), central["B_hat"]))
# print(np.array(r["central"].rx("B.hat")[0]).sum().sum(), central["B_hat"].sum().sum())
# print(np.array(r["central"].rx("B.hat")[0]).shape, central["B_hat"].shape)


r(
    """
### Step 2
site.outs <- NULL
for (b in unique(batch)) {
  s <- batch == b
  df <- dat[,s]
  bat <- batch[s]
  x <- mod[s,]
  
  site.outs <- c(site.outs, list(distributedCombat_site(df, bat, x, ref.batch = "1",
                                                        central.out = central)))
}
central <- distributedCombat_central(site.outs, ref.batch = "1")
"""
)
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
print("Step 2 (ref batch) comparisons:")
print(np.allclose(r["central"].rx("stand.mean"), central["stand_mean"]))
print(np.allclose(r["central"].rx("B.hat"), central["B_hat"]))

r(
    """
### Compare distributed vs original
site.outs <- NULL
error_ref <- NULL
perror_ref <- NULL # percent difference
"""
)
### Compare distributed vs original
site_outs = []
error = []
perror = []  # percent difference
for b in covars[batch_col].unique():
    r(
        """
    b <- {0}
    s <- batch == b
    df <- dat[,s]
    bat <- batch[s]
    x <- mod[s,]

    old.dat <- dat
    dat <- df
    old.batch <- batch
    batch <- bat
    old.mod <- mod
    mod <- x
    central.out <- central
    ref.batch=NULL
    eb=TRUE
    parametric=TRUE
    mean.only=FALSE
    verbose=TRUE
    file=NULL
    
    #site.out <- distributedCombat_site(df, bat, x, ref.batch = "1", 
    #                                    central.out = central)
    """.format(
            b
        )
    )
    s = list(map(lambda x: x == b, covars[batch_col]))
    df = dat.loc[:, s]
    bat = covars[batch_col][s]
    x = mod.loc[s, :]
    f = "site_out_" + str(b) + ".pickle"
    out = dc.distributedCombat_site(
        df, bat, x, central_out=central, file=f, ref_batch="1", debug=r
    )
    r(
        """
    dat <- old.dat
    batch <- old.batch
    mod <- old.mod


    site.outs <- c(site.outs, site.out)

    error_ref <- c(error_ref, max(c(com_out_ref$dat.combat[,s] - site.out$dat.combat)))
    perror_ref <- c(perror_ref,
                max(c(abs(com_out_ref$dat.combat[,s] - site.out$dat.combat)/
                        site.out$dat.combat)))
    """
    )
    site_outs.append(f)
    print("Site comparisons (ref batch):")
    print(
        "dat_combat",
        np.allclose(
            np.array(r["site.out"].rx("dat.combat")), np.array(out["dat_combat"])
        ),
    )
    print(
        "gamma_hat",
        np.array(r["estimates"].rx("gamma.hat")).shape,
        out["estimates"]["gamma_hat"].shape,
    )
    print(
        "gamma_hat",
        np.allclose(r["estimates"].rx("gamma.hat"), out["estimates"]["gamma_hat"]),
    )
    print(
        "delta_hat",
        np.allclose(r["estimates"].rx("delta.hat"), out["estimates"]["delta_hat"]),
    )
    print(
        "gamma_star",
        np.allclose(r["estimates"].rx("gamma.star"), out["estimates"]["gamma_star"]),
    )
    print(
        "delta_star",
        np.allclose(r["estimates"].rx("delta.star"), out["estimates"]["delta_star"]),
    )
    print(
        "gamma_bar",
        np.allclose(r["estimates"].rx("gamma.bar"), out["estimates"]["gamma_bar"]),
    )
    print("t2", np.allclose(r["estimates"].rx("t2"), out["estimates"]["t2"]))
    print(
        "a_prior",
        np.allclose(r["estimates"].rx("a.prior"), out["estimates"]["a_prior"]),
    )
    print(
        "b_prior",
        np.allclose(r["estimates"].rx("b.prior"), out["estimates"]["b_prior"]),
    )
    print(
        "stand_mean",
        np.allclose(r["estimates"].rx("stand.mean"), out["estimates"]["stand_mean"]),
    )
    print(
        "mod_mean",
        np.allclose(r["estimates"].rx("mod.mean"), out["estimates"]["mod_mean"]),
    )
    print(
        "var_pooled",
        np.allclose(r["estimates"].rx("var.pooled"), out["estimates"]["var_pooled"]),
    )
    print(
        "beta_hat",
        np.allclose(r["estimates"].rx("beta.hat"), out["estimates"]["beta_hat"]),
    )
    print("mod", np.allclose(r["estimates"].rx("mod"), out["estimates"]["mod"]))
    # print('batch', np.allclose(r["estimates"].rx("batch"), out["estimates"]["batch"]))
    # print('ref_batch', np.allclose(r["estimates"].rx("ref.batch"), out["estimates"]["ref_batch"]))
    print("eb", np.allclose(r["estimates"].rx("eb"), out["estimates"]["eb"]))
    print(
        "parametric",
        np.allclose(r["estimates"].rx("parametric"), out["estimates"]["parametric"]),
    )
    print(
        "mean_only",
        np.allclose(r["estimates"].rx("mean.only"), out["estimates"]["mean_only"]),
    )
    # error.append(com_out['data'][:,s] - out["dat_combat"])

# print("ERROR", len(error), error[0])
