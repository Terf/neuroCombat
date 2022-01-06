import pandas as pd
import numpy as np
import patsy
import pickle
import math
import sys


def betaNA(yy, designn):
    # designn <- designn[!is.na(yy),]
    # yy <- yy[!is.na(yy)]
    # B <- solve(crossprod(designn), crossprod(designn, yy))
    designn = designn.dropna()
    yy = yy[~yy.isna()]
    B = np.linalg.lstsq(designn, yy, rcond=None)[0]
    return B

def aprior(delta_hat):
    # todo check this
    m = delta_hat.mean().mean()
    s2 = np.cov(delta_hat)
    return (2 * s2 + m ** 2) / s2


def bprior(delta_hat):
    m = delta_hat.mean().mean()
    s2 = np.cov(delta_hat)
    return (m * s2 + m ** 3) / s2


def postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


def it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001):
    n = (1 - np.isnan(sdat)).sum(axis=1)
    g_old = g_hat.copy()
    d_old = d_hat.copy()
    ones = np.ones((1, sdat.shape[1]))

    change = 1
    count = 0
    while change > conv:
        g_new = np.array(postmean(g_hat, g_bar, n, d_old, t2))
        sum2 = ((sdat - np.dot(g_new.reshape((g_new.shape[0], 1)), ones)) ** 2).sum(
            axis=1
        )
        d_new = postvar(sum2, n, a, b)

        # change = max(
        #     (abs(g_new - g_old.item()) / g_old.item()).max(),
        #     (abs(d_new - d_old) / d_old).max(),
        # )
        change = max(max(abs(g_new - g_old) / g_old), max(abs(d_new - d_old) / d_old))
        g_old = g_new  # .copy()
        d_old = d_new  # .copy()
        count = count + 1
    adjust = (g_new, d_new)
    return adjust


def int_eprior(sdat, g_hat, d_hat):
    r = sdat.shape[0]
    gamma_star, delta_star = [], []
    for i in range(0, r, 1):
        g = np.delete(g_hat, i)
        d = np.delete(d_hat, i)
        x = sdat[i, :]
        n = x.shape[0]
        j = np.repeat(1, n)
        A = np.repeat(x, g.shape[0])
        A = A.reshape(n, g.shape[0])
        A = np.transpose(A)
        B = np.repeat(g, n)
        B = B.reshape(g.shape[0], n)
        resid2 = np.square(A - B)
        sum2 = resid2.dot(j)
        LH = 1 / (2 * math.pi * d) ** (n / 2) * np.exp(-sum2 / (2 * d))
        LH = np.nan_to_num(LH)
        gamma_star.append(sum(g * LH) / sum(LH))
        delta_star.append(sum(d * LH) / sum(LH))
    adjust = (gamma_star, delta_star)
    return adjust


def getdata_dictDC(batch, mod, verbose, mean_only, ref_batch=None):
    nbatch = len(batch.cat.categories)
    batches = []
    n_batches = []
    for x in batch.cat.categories:
        indices = np.where(batch == x)
        batches.append(indices)
        n_batches.append(len(indices[0]))
    n_array = np.array(n_batches).sum()
    batchmod = patsy.dmatrix("~-1+batch", batch, return_type="dataframe")
    if verbose:
        print("[combat] Found", nbatch, "batches")
    if not mean_only and np.all(n_batches == 1):
        raise ValueError("Found site with only one sample; consider using mean_only=True")
    ref = None
    if ref_batch is not None:
        if ref_batch not in batch.cat.categories:
            raise ValueError("Reference batch not in batch list")
        if verbose:
            print("[combat] Using batch=%s as a reference batch" % ref_batch)
        ref = np.where(np.any(batch.cat.categories == ref_batch))[0][0] # find the reference
        batchmod.iloc[:, ref] = 1
    # combine batch variable and covariates
    design = pd.concat([batchmod, mod], axis=1)
    # design = pd.concat([batchmod.reset_index(drop=True), mod.reset_index(drop=True)], axis=1)
    n_covariates = design.shape[1] - batchmod.shape[1]
    if verbose:
        print(
            "[combat] Adjusting for ",
            n_covariates,
            " covariate(s) or covariate level(s)",
        )
    out = {}
    out["batch"] = batch
    out["batches"] = batches
    out["n_batch"] = nbatch
    out["n_batches"] = n_batches
    out["n_array"] = n_array
    out["n_covariates"] = n_covariates
    out["design"] = design
    out["batch_design"] = design.iloc[:, :nbatch]
    out["ref"] = ref
    out["ref_batch"] = ref_batch
    return out


def getSigmaSummary(dat, data_dict, design, hasNAs, central_out):
    batches = data_dict["batches"]
    nbatches = data_dict["n_batches"]
    narray = data_dict["n_array"]
    nbatch = data_dict["n_batch"]
    ref_batch = data_dict["ref_batch"]
    ref = data_dict["ref"]
    Bhat = central_out["B_hat"]
    stand_mean = central_out["stand_mean"][:, 0:narray]

    if not hasNAs:
        if ref_batch is not None:
            ref_dat = dat.iloc[:, batches[ref][0]]
            factors = nbatches[ref] / (nbatches[ref] - 1)
            var_pooled = (
                np.cov(dat - np.matmul(design.iloc[batches[ref][0]], Bhat).transpose())
                / factors
            )
        else:
            factors = narray / (narray - 1)
            var_pooled = np.cov(dat - np.matmul(design, Bhat).transpose()) / factors
    else:
        if ref_batch is not None:
            ref_dat = dat.iloc[:, batches[ref][0]]
            ns = ref_dat.isna().sum()
            factors = nbatches[ref] / (nbatches[ref] - 1)
            var_pooled = (
                np.cov(dat - np.matmul(design.iloc[batches[ref][0]], Bhat).transpose())
                / factors
            )
        else:
            ns = dat.isna().sum()
            factors = ns / (ns - 1)
            var_pooled = np.cov(dat - np.matmul(design, Bhat).transpose()) / factors
    # todo: note sure why I had to do this
    var_pooled = np.diagonal(var_pooled)
    return var_pooled


def getStandardizedDataDC(dat, data_dict, design, hasNAs, central_out):
    batches = data_dict["batches"]
    nbatches = data_dict["n_batches"]
    narray = data_dict["n_array"]
    nbatch = data_dict["n_batch"]
    ref_batch = data_dict["ref_batch"]
    ref = data_dict["ref"]
    Bhat = central_out["B_hat"]
    stand_mean = central_out["stand_mean"]
    var_pooled = central_out["var_pooled"]

    if design is not None:
        tmp = design
        tmp.iloc[:, :nbatch] = 0
        mod_mean = np.matmul(tmp, Bhat).transpose()
    else:
        mod_mean = np.zeros(narray)
    # todo check stand_mean
    stand_mean = stand_mean[:, 0:narray]
    # s_data = (dat - stand_mean - mod_mean) / np.matmul(np.sqrt(var_pooled), np.ones(narray))
    s_data = (dat - stand_mean - mod_mean) / np.tile(
        np.sqrt(var_pooled), (narray, 1)
    ).transpose()
    return {
        "s_data": s_data,
        "stand_mean": stand_mean,
        "mod_mean": mod_mean,
        "var_pooled": var_pooled,
        "beta_hat": Bhat,
    }


def getNaiveEstimators(s_data, data_dict, hasNAs, mean_only):
    # todo double check this
    batch_design = data_dict["batch_design"]
    batches = data_dict["batches"]
    if not hasNAs:
        gamma_hat = np.matmul(
            np.linalg.inv(np.matmul(batch_design.transpose(), batch_design)),
            batch_design.transpose(),
        )
        gamma_hat = np.matmul(gamma_hat, s_data.transpose())
    else:
        # todo check
        gamma_hat = s_data.apply(betaNA, axis=0, args=(batch_design,))
    delta_hat = None
    for i in batches:
        if mean_only:
            delta_hat = pd.concat([delta_hat, np.ones(s_data.shape[1])])
        else:
            delta_hat = pd.concat(
                [delta_hat, pd.DataFrame(np.cov(s_data.iloc[:, i[0]], rowvar=True))]
            )
    # todo not sure why I had to take diagonal
    return {"gamma_hat": gamma_hat, "delta_hat": np.diagonal(delta_hat)}


def getEbEstimators(
    naiveEstimators, s_data, data_dict, parametric=True, mean_only=False
):
    gamma_hat = (
        naiveEstimators["gamma_hat"]
        .to_numpy()
        .reshape(naiveEstimators["gamma_hat"].shape[1])
    )
    delta_hat = naiveEstimators["delta_hat"]
    # pd.DataFrame(naiveEstimators["delta_hat"].reshape(1, len(naiveEstimators["delta_hat"])))
    batches = data_dict["batches"]
    nbatch = data_dict["n_batch"]
    ref_batch = data_dict["ref_batch"]
    ref = data_dict["ref"]

    def getParametricEstimators():
        gamma_star = delta_star = []
        for i in range(nbatch):
            if mean_only:
                gamma_star.append(postmean(gamma_hat[i], gamma_bar[i], 1, 1, t2))
                delta_star.append(np.ones(len(s_data)))
            else:
                temp = it_sol(
                    s_data.iloc[:, batches[i][0]],
                    gamma_hat,
                    delta_hat,
                    gamma_bar,
                    t2,
                    a_prior,
                    b_prior,
                )
                gamma_star.append(temp[0])
                delta_star.append(temp[1])
        return gamma_star[0], delta_star[1]

    def getNonParametricEstimators():
        gamma_star = delta_star = []
        for i in range(nbatch):
            if mean_only:
                delta_hat[i] = 1
            else:
                temp = int_eprior(s_data[batches[i]], gamma_hat[i], delta_hat[i])
                gamma_star.append(temp[0])
                delta_star.append(temp[1])
        return gamma_star, delta_star

    gamma_bar = gamma_hat.mean().mean()
    # t2 = gamma_hat.var(axis=0)
    t2 = np.cov(gamma_hat)
    a_prior = aprior(delta_hat)
    b_prior = bprior(delta_hat)
    tmp = getParametricEstimators() if parametric else getNonParametricEstimators()
    if ref_batch is not None:
        # set reference batch mean equal to 0
        tmp[0][ref] = 0
        # set reference batch variance equal to 1
        tmp[1][ref] = 1
    out = {}
    out["gamma_star"] = tmp[0]
    out["delta_star"] = tmp[1]
    out["gamma_bar"] = gamma_bar
    out["t2"] = t2
    out["a_prior"] = a_prior
    out["b_prior"] = b_prior
    return out


def getNonEbEstimators(naiveEstimators, data_dict):
    out = {}
    out["gamma_star"] = naiveEstimators["gamma_hat"]
    out["delta_star"] = naiveEstimators["delta_hat"]
    out["gamma_bar"] = None
    out["t2"] = None
    out["a_prior"] = None
    out["b_prior"] = None
    ref_batch = data_dict["ref_batch"]
    ref = data_dict["ref"]
    if ref_batch is not None:
        # set reference batch mean equal to 0
        out["gamma_star"][batches[ref]] = 0
        # set reference batch variance equal to 1
        out["delta_star"][batches[ref]] = 1
    return out


def getCorrectedData(
    dat, s_data, data_dict, estimators, naive_estimators, std_objects, eb=True
):
    var_pooled = std_objects["var_pooled"]
    stand_mean = std_objects["stand_mean"]
    mod_mean = std_objects["mod_mean"]
    batches = data_dict["batches"]
    batch_design = data_dict["batch_design"]
    n_batches = data_dict["n_batches"]
    n_array = data_dict["n_array"]
    ref_batch = data_dict["ref_batch"]
    ref = data_dict["ref"]

    if eb:
        gamma_star = estimators["gamma_star"]
        delta_star = estimators["delta_star"]
    else:
        gamma_star = naive_estimators["gamma_hat"]
        delta_star = naive_estimators["delta_hat"]
    gamma_star = gamma_star.reshape(1, len(gamma_star))

    bayesdata = s_data.copy()
    j = 0
    for i in batches:
        # einsum: https://stackoverflow.com/a/33641428/2624391
        top = (
            bayesdata.iloc[:, i[0]]
            - np.einsum(
                "ij,i->ij", gamma_star, batch_design.iloc[i[0], :].to_numpy().flatten()
            ).transpose()
        )
        bottom = np.sqrt(delta_star[j]) * np.ones(n_batches[j])
        bayesdata.iloc[:, i[0]] = top / bottom
        j += 1
    bayesdata = (
        (
            bayesdata
            * np.einsum("i,j->ij", np.sqrt(var_pooled), np.ones(n_array).transpose())
        )
        + stand_mean
        + mod_mean
    )
    if ref_batch is not None:
        bayesdata.iloc[:, batches[ref][0]] = dat.iloc[:, batches[ref][0]]
    return bayesdata


#' Distributed ComBat step at each site
#'
#' @param dat A \emph{p x n} matrix (or object coercible by
#'   \link[base]{as.matrix} to a numeric matrix) of observations where \emph{p}
#'   is the number of features and \emph{n} is the number of subjects.
#' @param batch Factor indicating batch. Needs to have the same levels across
#'   all individual sites, but can have multiple batches per site (i.e.
#'   multiple levels in each site)
#' @param mod Optional design matrix of covariates to preserve, usually from
#'    \link[stats]{model.matrix}. This matrix needs to have the same columns
#'    across sites. The rows must be in the same order as the data columns.
#' @param central.out Output list from \code{distributedCombat_central}. Output
#'   of \code{distributedCombat_site} will depend on the values of
#'   \code{central.out}. If \code{NULL}, then the output will be sufficient for
#'   estimation of \code{B_hat}. If \code{B_hat} is provided, then the output
#'   will be sufficient for estimation of \code{sigma} or for harmonization if
#'   \code{mean_only} is \code{TRUE}. If \code{sigma} is provided, then
#'   harmonization will be performed.
#' @param eb If \code{TRUE}, the empirical Bayes step is used to pool
#'   information across features, as per the original ComBat methodology. If
#'   \code{FALSE}, adjustments are made for each feature individually.
#'   Recommended left as \code{TRUE}.
#' @param parametric If \code{TRUE}, parametric priors are used for the
#'   empirical Bayes step, otherwise non-parametric priors are used. See
#'   neuroComBat package for more details.
#' @param mean_only If \code{TRUE}, distributed ComBat does not harmonize the
#'   variance of features.
#' @param verbose If \code{TRUE}, print progress updates to the console.
#' @param file File name of .pickle file to export
#'
def distributedCombat_site(
    dat,
    batch,
    mod=None,
    ref_batch=None,
    central_out=None,
    eb=True,
    parametric=True,
    mean_only=False,
    verbose=False,
    file=None,
):
    if file is None:
        file = "distributedCombat_site.pickle"
        print(
            "Must specify filename to output results as a file. Currently saving output to current workspace only."
        )
    if isinstance(central_out, str):
        central_out = pd.read_pickle(central_out)
    hasNAs = np.isnan(dat).any(axis=None)
    if verbose and hasNAs:
        print("[neuroCombat] WARNING: NaNs detected in data")
    if mean_only:
        print("[neuroCombat] Performing ComBat with mean only")

    ##################### Getting design ############################
    data_dict = getdata_dictDC(
        batch, mod, verbose=verbose, mean_only=mean_only, ref_batch=ref_batch
    )

    design = data_dict["design"]
    #################################################################

    ############### Site matrices for standardization ###############
    # W^T W used in LS estimation
    ls_site = []
    ls_site.append(np.dot(design.transpose(), design))
    ls_site.append(np.dot(design.transpose(), dat.transpose()))
    # print("confirming ls_site")
    # print(design.shape)
    # print(ls_site[0].shape)

    data_dict_out = data_dict.copy()
    data_dict_out["design"] = None

    # new data_dict with batches within current site
    incl_bat = [x > 0 for x in data_dict["n_batches"]]
    data_dict_site = data_dict.copy()
    data_dict_site["batches"] = [
        data_dict["batches"][i] for i in range(len(data_dict["batches"])) if incl_bat[i]
    ]
    data_dict_site["n_batch"] = incl_bat.count(True)
    data_dict_site["n_batches"] = [
        data_dict["n_batches"][i]
        for i in range(len(data_dict["n_batches"]))
        if incl_bat[i]
    ]
    data_dict_site["batch_design"] = data_dict["batch_design"].loc[:, incl_bat]

    # remove reference batch information if reference batch is not in site
    if ref_batch is not None:
        if data_dict_site["ref"] in data_dict_site["batch"]:
            data_dict_site["ref"] = np.where(np.any(data_dict_site["batch"] == ref_batch))[0][0]
        else:
            data_dict_site["ref"] = None
            data_dict_site["ref_batch"] = None

    if central_out is None:
        site_out = {
            "ls_site": ls_site,
            "data_dict": data_dict,
            "sigma_site": None,
        }
        with open(file, "wb") as handle:
            pickle.dump(site_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return site_out

    # If beta.estimates given, get summary statistics for sigma estimation

    if "var_pooled" not in central_out or central_out["var_pooled"] is None:
        sigma_site = getSigmaSummary(dat, data_dict, design, hasNAs, central_out)
        site_out = {
            "ls_site": ls_site,
            "data_dict": data_dict,
            "sigma_site": sigma_site,
        }
        with open(file, "wb") as handle:
            pickle.dump(site_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return site_out

    stdObjects = getStandardizedDataDC(
        dat=dat,
        data_dict=data_dict,
        design=design,
        hasNAs=hasNAs,
        central_out=central_out,
    )
    s_data = stdObjects["s_data"]

    ##################### Getting L/S estimates #######################
    if verbose:
        print("[distributedCombat] Fitting L/S model and finding priors")
    naiveEstimators = getNaiveEstimators(
        s_data=s_data, data_dict=data_dict_site, hasNAs=hasNAs, mean_only=mean_only
    )
    ####################################################################
    ########################### Getting final estimators ###############
    if eb:
        if verbose:
            print(
                "[distributedCombat] Finding ",
                ("" if parametric else "non-"),
                "parametric adjustments",
                sep="",
            )
        estimators = getEbEstimators(
            naiveEstimators=naiveEstimators,
            s_data=s_data,
            data_dict=data_dict_site,
            parametric=parametric,
            mean_only=mean_only,
        )
    else:
        estimators = getNonEbEstimators(
            naiveEstimators=naiveEstimators, data_dict=data_dict
        )

    ######################### Correct data #############################
    if verbose:
        print("[distributedCombat] Adjusting the Data")
    bayesdata = getCorrectedData(
        dat=dat,
        s_data=s_data,
        data_dict=data_dict_site,
        estimators=estimators,
        naive_estimators=naiveEstimators,
        std_objects=stdObjects,
        eb=eb,
    )

    # List of estimates:
    estimates = {
        "gamma_hat": naiveEstimators["gamma_hat"],
        "delta_hat": naiveEstimators["delta_hat"],
        "gamma_star": estimators["gamma_star"],
        "delta_star": estimators["delta_star"],
        "gamma_bar": estimators["gamma_bar"],
        "t2": estimators["t2"],
        "a_prior": estimators["a_prior"],
        "b_prior": estimators["b_prior"],
        "stand_mean": stdObjects["stand_mean"],
        "mod_mean": stdObjects["mod_mean"],
        "var_pooled": stdObjects["var_pooled"],
        "beta_hat": stdObjects["beta_hat"],
        "mod": mod,
        "batch": batch,
        "ref_batch": ref_batch,
        "eb": eb,
        "parametric": parametric,
        "mean_only": mean_only,
    }
    site_out = {"dat_combat": bayesdata, "estimates": estimates}
    with open(file, "wb") as handle:
        pickle.dump(site_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return site_out


#' Distributed ComBat step at analysis core
#'
#' @param site.outs List of filenames containing site outputs.
#' @param file File name of .pickle file to export
def distributedCombat_central(site_outs, ref_batch=None, verbose=False, file=None):
    if file is None:
        print(
            "Must specify filename to output results as a file. Currently saving output to current workspace only."
        )
        file = "combat_central.pickle"
    site_outs = [pickle.load(open(site_out, "rb")) for site_out in site_outs]
    m = len(site_outs)  # number of sites
    # get n.batches and n.array from sites
    batch_levels = site_outs[0]["data_dict"]["batch"].cat.categories
    n_batches = np.cumsum(
        [site_out["data_dict"]["n_batches"] for site_out in site_outs], axis=0
    )[-1]
    n_batch = len(n_batches)
    n_array = np.array(n_batches).sum()
    n_arrays = [site_out["data_dict"]["n_array"] for site_out in site_outs]

    # get reference batch if specified
    ref = None
    if ref_batch is not None:
        # todo is this the right batch?
        batch = site_outs[0]["data_dict"]["batch"]
        if ref_batch not in batch.cat.categories:
            raise ValueError("ref_batch not in batch.cat.categories")
        if verbose:
            print("[combat] Using batch=%s as a reference batch" % ref_batch)
        ref = np.where(batch_levels == ref_batch)

    # check if beta estimates have been given to sites
    # todo check sigma site
    step1s = np.array([site_out["sigma_site"] is None for site_out in site_outs])
    if len(np.unique(step1s)) > 1:
        raise ValueError(
            "Not all sites are at the same step, please confirm with each site."
        )
    step1 = np.all(step1s)  # todo check

    #### Step 1: Get LS estimate across sites ####
    ls1 = np.array([x["ls_site"][0] for x in site_outs])
    ls2 = np.array([x["ls_site"][1] for x in site_outs])
    ls1 = np.cumsum(ls1, axis=0)[-1]
    ls2 = np.cumsum(ls2, axis=0)[-1]
    B_hat = np.matmul(np.transpose(np.linalg.inv(ls1)), ls2)

    if ref_batch is not None:
        grand_mean = B_hat[ref].transpose()
    else:
        grand_mean = np.matmul(
            np.transpose(n_batches / n_array), B_hat[range(n_batch), :]
        )
    grand_mean = np.reshape(grand_mean, (1, len(grand_mean)))
    stand_mean = np.matmul(
        np.transpose(grand_mean), np.transpose(np.ones(n_array)).reshape(1, n_array)
    )

    if step1:
        central_out = {"B_hat": B_hat, "stand_mean": stand_mean, "var_pooled": None}
        with open(file, "wb") as handle:
            pickle.dump(central_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return central_out

    # #### Step 2: Get standardization parameters ####
    vars = list(map(lambda x: x["sigma_site"], site_outs))

    # if ref_batch specified, use estimated variance from reference site
    if ref_batch is not None:
        var_pooled = vars[ref[0][0]]
    else:
        var_pooled = np.zeros(len(vars[0]))
        for i in range(m):
            var_pooled += n_arrays[i] * np.array(vars[i])
        var_pooled = var_pooled / n_array

    central_out = {"B_hat": B_hat, "stand_mean": stand_mean, "var_pooled": var_pooled}
    with open(file, "wb") as handle:
        pickle.dump(central_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return central_out
