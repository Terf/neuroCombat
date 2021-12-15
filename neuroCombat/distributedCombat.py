import pandas as pd
import numpy as np
import patsy
import pickle


def aprior(delta_hat):
    m = np.mean(delta_hat)
    s2 = np.var(delta_hat, ddof=1)
    return (2 * s2 + m ** 2) / float(s2)


def bprior(delta_hat):
    m = delta_hat.mean()
    s2 = np.var(delta_hat, ddof=1)
    return (m * s2 + m ** 3) / s2


def postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


def it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001):
    n = (1 - np.isnan(sdat)).sum(axis=1)
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    change = 1
    count = 0
    while change > conv:
        g_new = postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = (
            (
                sdat
                - np.dot(
                    g_new.reshape((g_new.shape[0], 1)), np.ones((1, sdat.shape[1]))
                )
            )
            ** 2
        ).sum(axis=1)
        d_new = postvar(sum2, n, a, b)

        change = max(
            (abs(g_new - g_old) / g_old).max(), (abs(d_new - d_old) / d_old).max()
        )
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
    ref = None
    if ref_batch is not None:
        if ref_batch not in batch:
            raise ValueError("Reference batch not in batch list")
        if verbose:
            print("[combat] Using batch=%s as a reference batch" % ref_batch)
        # ref <- which(levels(as.factor(batch))==ref_batch) # find the reference
        # batchmod[,ref] <- 1
    # combine batch variable and covariates
    design = pd.concat([batchmod, mod], axis=1)
    n_covariates = len(design) - len(batchmod)
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
    Bhat = None if "B.hat" not in central_out else central_out["B.hat"]

    if not hasNAs:
        if ref_batch is not None:
            ref_dat = dat.iloc[:, batches[ref]]
            factors = nbatches[ref] / nbatches[ref] - 1
            var_pooled = (
                np.cov(dat - np.matmul(design[batches[ref]], Bhat).transpose())
                / factors
            )
        else:
            factors = narray / (narray - 1)
            var_pooled = np.cov(dat - np.matmul(design, Bhat).transpose()) / factors
    else:
        if ref_batch is not None:
            ref_dat = dat.iloc[:, batches[ref]]
            ns = ref_dat.isna().sum()
            factors = nbatches[ref] / nbatches[ref] - 1
            var_pooled = (
                np.cov(dat - np.matmul(design[batches[ref]], Bhat).transpose())
                / factors
            )
        else:
            ns = dat.isna().sum()
            factors = ns / (ns - 1)
            var_pooled = np.cov(dat - np.matmul(design, Bhat).transpose()) / factors
    return var_pooled


def getStandardizedDataDC(dat, data_dict, design, hasNAs, central_out):
    batches = data_dict["batches"]
    nbatches = data_dict["n_batches"]
    narray = data_dict["n_array"]
    nbatch = data_dict["n_batch"]
    ref_batch = data_dict["ref_batch"]
    ref = data_dict["ref"]
    Bhat = central_out["B.hat"]
    stand_mean = central_out["stand_mean"]
    var_pooled = central_out["var_pooled"]

    if design is not None:
        tmp = design
        tmp.iloc[:, :nbatch] = 0
        mod_mean = np.matmul(tmp, Bhat).transpose()
    else:
        mod_mean = np.zeros(narray)
    s_data = (dat - stand_mean - mod_mean) / np.sqrt(var_pooled)
    return {
        "s_data": s_data,
        "stand_mean": stand_mean,
        "mod_mean": mod_mean,
        "var_pooled": var_pooled,
        "beta_hat": Bhat,
    }


def getNaiveEstimators(s_data, data_dict, hasNAs, mean_only):
    batch_design = data_dict["batch_design"]
    batches = data_dict["batches"]
    if not hasNAs:
        gamma_hat = np.matmul(
            np.linalg.solve(np.matmul(batch_design.transpose(), batch_design)),
            batch_design.transpose(),
        )
        gamma_hat = np.matmul(gamma_hat, s_data.transpose())
    else:
        gamma_hat = None
    delta_hat = None
    return gamma_hat, delta_hat


def getEbEstimators(
    naiveEstimators, s_data, data_dict, parametric=True, mean_only=False
):
    gamma_hat = naiveEstimators["gamma_hat"]
    delta_hat = naiveEstimators["delta_hat"]
    batches = data_dict["batches"]
    nbatch = data_dict["n_batch"]
    ref_batch = data_dict["ref_batch"]
    ref = data_dict["ref"]

    def getParametricEstimators():
        gamma_star = delta_star = []
        for i in range(nbatch):
            if mean_only:
                gamma_star.append(postmean(gamma_hat[i], gamma_bar[i], 1, 1, t2[i]))
                delta_star.append(range(len(s_data)))
            else:
                temp = it_sol(
                    s_data[batches[i]],
                    gamma_hat[i],
                    delta_hat[i],
                    gamma_bar[i],
                    t2[i],
                    a_prior[i],
                    b_prior[i],
                )
                gamma_star.append(temp[0])
                delta_star.append(temp[1])
        return gamma_star, delta_star

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

    gamma_bar = gamma_hat.mean(axis=0)
    t2 = gamma_hat.var(axis=0)
    a_prior = aprior(delta_hat)
    b_prior = bprior(delta_hat)
    tmp = getParametricEstimators() if parametric else getNonParametricEstimators()
    if ref_batch is not None:
        # set reference batch mean equal to 0
        tmp["gamma_star"][ref] = 0
        # set reference batch variance equal to 1
        tmp["delta_star"][ref] = 1
    out = {}
    out["gamma_star"] = tmp["gamma_star"]
    out["delta_star"] = tmp["delta_star"]
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
#'   estimation of \code{B.hat}. If \code{B.hat} is provided, then the output
#'   will be sufficient for estimation of \code{sigma} or for harmonization if
#'   \code{mean.only} is \code{TRUE}. If \code{sigma} is provided, then
#'   harmonization will be performed.
#' @param eb If \code{TRUE}, the empirical Bayes step is used to pool
#'   information across features, as per the original ComBat methodology. If
#'   \code{FALSE}, adjustments are made for each feature individually.
#'   Recommended left as \code{TRUE}.
#' @param parametric If \code{TRUE}, parametric priors are used for the
#'   empirical Bayes step, otherwise non-parametric priors are used. See
#'   neuroComBat package for more details.
#' @param mean.only If \code{TRUE}, distributed ComBat does not harmonize the
#'   variance of features.
#' @param verbose If \code{TRUE}, print progress updates to the console.
#' @param file File name of .Rdata file to export
#'
def distributedCombat_site(
    dat,
    batch,
    mod=None,
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
    if central_out is not None:
        central_out = pd.read_pickle(central_out)
    hasNAs = np.isnan(dat).any()
    if verbose and hasNAs:
        print("[neuroCombat] WARNING: NaNs detected in data")
    if mean_only:
        print("[neuroCombat] Performing ComBat with mean only")

    ##################### Getting design ############################
    print("[neuroCombat] Getting design matrix")
    data_dict = getdata_dictDC(
        batch, mod, verbose=verbose, mean_only=mean_only, ref_batch=None
    )
    # print('data_dict:', data_dict)

    design = data_dict["design"]
    #################################################################

    ############### Site matrices for standardization ###############
    # W^T W used in LS estimation
    ls_site = []
    ls_site.append(np.dot(design.transpose(), design))
    ls_site.append(np.dot(design.transpose(), dat.transpose()))

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
        naiveEstimators=naiveEstimators,
        stdObjects=stdObjects,
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
        "mean.only": mean_only,
    }
    site_out = {"dat.combat": bayesdata, "estimates": estimates}
    with open(file, "wb") as handle:
        pickle.dump(site_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return site_out


#' Distributed ComBat step at analysis core
#'
#' @param site.outs List or vector of filenames containing site outputs.
#' @param file File name of .Rdata file to export
def distributedCombat_central(site_outs, file=None):
    site_outs = [pickle.load(open(site_out, "rb")) for site_out in site_outs]
    m = len(site_outs)  # number of sites
    # get n.batches and n.array from sites
    batch_levels = len(site_outs[0]["data_dict"]["batch"].cat.categories)
    n_batches = [site_out["data_dict"]["n_batch"] for site_out in site_outs]
    n_batch = len(n_batches)
    n_array = np.array(n_batches).sum()
    n_arrays = [site_out["data_dict"]["n_array"] for site_out in site_outs]

    # check if beta estimates have been given to sites
    step1s = [site_out["sigma_site"] is None for site_out in site_outs]
    if not np.all(step1s):
        raise ValueError(
            "Not all sites are at the same step, please confirm with each site."
        )
    ls1 = np.array([x["ls_site"][0] for x in site_outs])[0]
    ls2 = np.array([x["ls_site"][1] for x in site_outs])[0]
    print("test", ls1.shape, ls2.shape)
    id_mat = ls1.shape[0]
    # ls1 = np.cumsum(ls1)
    # ls2 = np.cumsum(ls2)
    # print('test2', ls1.shape, ls2.shape)
    # print('test3', np.linalg.solve(ls1, np.identity(id_mat)), ls2)
    # print('test3', np.linalg.pinv(ls1, np.identity(id_mat)), ls2)
    B_hat = np.cross(np.linalg.solve(ls1, np.identity(id_mat)), ls2)

    grand_mean = B_hat[ref].transpose()
    stand_mean = np.cross(grand_mean, range(1, n_array).transpose())

    if step1:
        central_out = {"B_hat": B_hat, "stand_mean": stand_mean, "var_pooked": None}
        with open(file, "wb") as handle:
            pickle.dump(central_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
