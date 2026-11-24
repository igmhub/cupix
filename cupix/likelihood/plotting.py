"""Plotting utilities for Px data and window-convolved theory across redshift bins."""
from warnings import warn

import numpy as np
import matplotlib.pyplot as plt

from cupix.likelihood.likelihood import Likelihood


def get_continuous_windowed_px(like, params={}, n_sub=7):
    """Window-convolved theory sampled finely in k, for all theta_A bins of like.data.

    The data only provide window matrices for the coarse k_M bins. To draw a
    smooth curve we assume the window kernel of each k_M bin is translation
    invariant, so the windowed model at k_M + dk is obtained by convolving the
    theory evaluated at |k_m + dk| with the window of bin M. Each sub-bin
    offset costs one likelihood evaluation (all k_M bins at once).

    Args:
        like: Likelihood object (its data defines the k_M bins and windows).
        params: theory parameters.
        n_sub: number of k samples per k_M bin (forced odd, so the bin centre,
            i.e. the exact binned prediction, is always included).

    Returns:
        k_fine: (N_M * n_sub,) wavenumbers [1/AA]
        px_fine: (N_A, N_M * n_sub) windowed theory
        px_binned: (N_A, N_M) windowed theory at the k_M bin centres
            (identical to like.get_convolved_px(params))
    """
    if n_sub % 2 == 0:
        n_sub += 1
    edges = np.asarray(like.data.k_M_edges[like.iz])
    k_M = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    dk = np.median(widths)
    if not np.allclose(widths, dk, rtol=1e-3):
        warn("k_M bins are not uniform; using the median width to sub-sample the model.")

    fracs = (np.arange(n_sub) - (n_sub - 1) / 2) / n_sub
    px = np.stack([like.get_convolved_px(params=params, k_shift_AA=f * dk) for f in fracs],
                  axis=-1)  # (N_A, N_M, n_sub)
    k_fine = (k_M[:, None] + fracs[None, :] * dk).reshape(-1)
    px_fine = px.reshape(px.shape[0], -1)
    px_binned = px[:, :, (n_sub - 1) // 2]
    return k_fine, px_fine, px_binned


def _select(obj, iz, i):
    """Pick the entry for redshift index iz (position i) from a dict / list / single object."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        if iz in obj:
            return obj[iz]
        if str(iz) in obj:
            return obj[str(iz)]
        raise KeyError(f"No entry for iz={iz} in {list(obj.keys())}")
    if isinstance(obj, (list, tuple)):
        return obj[i]
    return obj


def _per_z_params(params, izs):
    """params can be a single dict (same for all z) or a dict keyed by iz."""
    if params is None:
        return {iz: {} for iz in izs}
    keys = set(params.keys())
    if keys and keys <= ({iz for iz in izs} | {str(iz) for iz in izs}):
        return {iz: _select(params, iz, i) for i, iz in enumerate(izs)}
    return {iz: params for iz in izs}


def plot_px_multiz(data, params=None, likelihoods=None, theories=None, izs=None,
                   like_config={}, n_free_p=0, n_sub=7,
                   multiply_by_k=True, multiply_by_k2=False,
                   theta_inds=None, every_other_theta=False,
                   residuals=True, residual_to_theory=False, connect_residuals=False,
                   shade_unused=True, xshade=None, unused_alpha=0.35,
                   include_chi2=False, include_probability=False,
                   xlim=None, ylim=None, ylim2=None,
                   theorylabel='Theory prediction, windowed', datalabel='Data',
                   title=None, ncols=2, ncols_leg=4, figsize=None, fontsize=16,
                   cmap='tab20', plot_fname=None, show=True):
    """Plot Px data and continuous window-convolved theory, one panel per redshift bin.

    Args:
        data: data object with all the measurements to plot (e.g. DESI_DR2 without cuts).
        params: theory parameters; one dict for all z, or a dict keyed by iz
            (int or str), e.g. best-fit values from separate per-z chains.
        likelihoods: Likelihood per z (dict keyed by iz or str(iz), list aligned with
            izs, or a single object). Its theory gives the model, and its data
            (possibly with theta / k cuts) defines which points were used in the fit.
        theories: Theory per z (same formats). Used only when likelihoods is None;
            then all plotted data are considered used.
        izs: redshift indices to plot; defaults to the keys of likelihoods/theories,
            or all redshifts in data.
        like_config: likelihood config (e.g. N_theta_average) used when building
            likelihoods from theories.
        n_free_p: number of free parameters (int, or dict keyed by iz) for chi2 / probability.
        n_sub: number of model samples per data k bin (cost: n_sub likelihood calls per z).
        theta_inds: indices of data theta_A bins to plot (default all).
        every_other_theta: plot every other theta bin (ignored if theta_inds given).
        residuals: add a (data - theory)/sigma panel below each redshift.
        shade_unused: shade k ranges outside the fit range and draw unused points
            (k or theta cuts) with open markers / dashed model.
        xshade: extra k range(s) to shade, [kmin, kmax] or list of them.
    Returns:
        fig, axes dict with keys 'main' and 'resid' (lists aligned with izs).
    """
    # ---- figure out redshifts, likelihoods for the fit and for plotting -------
    if izs is None:
        src = likelihoods if likelihoods is not None else theories
        if isinstance(src, dict):
            izs = sorted(int(k) for k in src.keys())
        else:
            izs = list(range(len(data.z)))
    izs = list(np.atleast_1d(izs))
    if likelihoods is None and theories is None:
        raise ValueError("Provide likelihoods and/or theories.")
    params_z = _per_z_params(params, izs)

    fit_likes, plot_likes = {}, {}
    for i, iz in enumerate(izs):
        like = _select(likelihoods, iz, i)
        theory = like.theory if like is not None else _select(theories, iz, i)
        config = dict(like_config)
        if like is not None:
            config.setdefault('N_theta_average', like.N_theta_average)
        # likelihood on the full data, used only to draw the model everywhere
        plot_likes[iz] = Likelihood(data=data, theory=theory, iz=iz, config=config)
        fit_likes[iz] = like if like is not None else plot_likes[iz]

    # ---- theta bins to plot ----------------------------------------------------
    Nt_A = len(data.theta_centers_arcmin)
    if theta_inds is None:
        theta_inds = range(0, Nt_A, 2 if every_other_theta else 1)
    theta_inds = list(theta_inds)
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, Nt_A))

    if xshade is not None and np.ndim(xshade) == 1:
        xshade = [xshade]

    # ---- layout: one cell per redshift, each optionally main + residual -------
    nz = len(izs)
    ncols = min(ncols, nz)
    nrows = int(np.ceil(nz / ncols))
    if figsize is None:
        figsize = (6.5 * ncols, (5.5 if residuals else 4.2) * nrows + 1.2)

    with plt.rc_context({'font.size': fontsize}):
        fig = plt.figure(figsize=figsize, layout='constrained')
        outer = fig.add_gridspec(nrows, ncols)
        ax_main, ax_res = [], []
        for i in range(nz):
            r, c = divmod(i, ncols)
            if residuals:
                inner = outer[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0)
                a0 = fig.add_subplot(inner[0], sharex=ax_main[0] if ax_main else None,
                                     sharey=ax_main[0] if ax_main else None)
                a1 = fig.add_subplot(inner[1], sharex=a0, sharey=ax_res[0] if ax_res else None)
                ax_res.append(a1)
            else:
                a0 = fig.add_subplot(outer[r, c], sharex=ax_main[0] if ax_main else None,
                                     sharey=ax_main[0] if ax_main else None)
            ax_main.append(a0)

        if multiply_by_k:
            ylabel = r'$k_\parallel P_\times$'
        elif multiply_by_k2:
            ylabel = r'$k_\parallel^2 P_\times$ [$\AA^{-1}$]'
        else:
            ylabel = r'$P_\times$ [$\AA$]'
        divname = 'Theory' if residual_to_theory else r'$\sigma$'

        any_unused = False
        y_used = []  # (lo, hi) of used data in plotted k range, for default ylim
        for i, iz in enumerate(izs):
            ax = ax_main[i]
            like_fit, like_plot = fit_likes[iz], plot_likes[iz]
            pars = params_z[iz]

            edges = np.asarray(data.k_M_edges[iz])
            k_M = 0.5 * (edges[:-1] + edges[1:])
            k_fine, px_fine, px_binned = get_continuous_windowed_px(like_plot, pars, n_sub=n_sub)

            def factor(k):
                if multiply_by_k:
                    return k
                if multiply_by_k2:
                    return k ** 2
                return 1.0

            # which data entered the fit
            fit_edges = np.asarray(like_fit.data.k_M_edges[iz])
            k_used = (k_M >= fit_edges[0] - 1e-8) & (k_M <= fit_edges[-1] + 1e-8)
            fit_thetas = np.asarray(like_fit.data.theta_centers_arcmin)

            for it_A in theta_inds:
                col = colors[it_A]
                theta = data.theta_centers_arcmin[it_A]
                theta_used = np.any(np.isclose(fit_thetas, theta))
                used = k_used & theta_used
                if shade_unused and not np.all(used):
                    any_unused = True

                px = data.Px_ZAM[iz, it_A, :]
                err = np.sqrt(np.diag(np.squeeze(data.cov_ZAM[iz, it_A])))
                label = r'$\theta_A={:.2f}^\prime$'.format(theta) if i == 0 else None

                ax.errorbar(k_M[used], (px * factor(k_M))[used], (err * factor(k_M))[used],
                            color=col, linestyle='none', marker='^', markersize=5, label=label)
                if np.any(~used):
                    ax.errorbar(k_M[~used], (px * factor(k_M))[~used], (err * factor(k_M))[~used],
                                color=col, linestyle='none', marker='^', markersize=5,
                                markerfacecolor='none', alpha=unused_alpha if shade_unused else 1.0)

                in_x = used if xlim is None else used & (k_M >= xlim[0]) & (k_M <= xlim[1])
                if np.any(in_x):
                    y_used.append([np.min(((px - err) * factor(k_M))[in_x]),
                                   np.max(((px + err) * factor(k_M))[in_x])])

                ls = '-' if (theta_used or not shade_unused) else '--'
                model = px_fine[it_A] * factor(k_fine)
                ax.plot(k_fine, model, color=col, lw=2, ls=ls)
                in_xf = slice(None) if xlim is None else (k_fine >= xlim[0]) & (k_fine <= xlim[1])
                y_used.append([np.min(model[in_xf]), np.max(model[in_xf])])

                if residuals:
                    div = px_binned[it_A] if residual_to_theory else err
                    res = (px - px_binned[it_A]) / div
                    rls = '-' if connect_residuals else 'none'
                    ax_res[i].plot(k_M[used], res[used], color=col, marker='o', ls=rls, ms=4)
                    if np.any(~used):
                        ax_res[i].plot(k_M[~used], res[~used], color=col, marker='o', ls='none',
                                       ms=4, mfc='none', alpha=unused_alpha if shade_unused else 1.0)

            # shading of k ranges not used in the fit (plus any user-defined)
            spans = []
            if shade_unused:
                if fit_edges[0] > edges[0] + 1e-8:
                    spans.append((edges[0] if xlim is None else min(xlim[0], edges[0]), fit_edges[0]))
                if fit_edges[-1] < edges[-1] - 1e-8:
                    spans.append((fit_edges[-1], edges[-1] if xlim is None else max(xlim[1], edges[-1])))
            if xshade is not None:
                spans += [tuple(s) for s in xshade]
            for axx in [ax] + ([ax_res[i]] if residuals else []):
                for s in spans:
                    axx.axvspan(s[0], s[1], color='k', alpha=0.1, lw=0)

            # annotations
            ax.set_title(r'$z={:.2f}$'.format(data.z[iz]), fontsize=fontsize)
            nfp = _select(n_free_p, iz, i) if isinstance(n_free_p, (dict, list, tuple)) else n_free_p
            text = []
            if include_chi2:
                chi2 = like_fit.get_chi2(params=pars)
                ndof = like_fit.get_ndata() - nfp
                text.append(rf'$\chi^2/n_\mathrm{{dof}}$ = {chi2:.1f}/{ndof}')
            if include_probability:
                prob = like_fit.get_probability(params=pars, n_free_p=nfp)
                text.append(f'Fit prob = {prob:.2f}')
            if text:
                ax.text(0.97, 0.95, '\n'.join(text), transform=ax.transAxes, ha='right', va='top',
                        fontsize='small')

            if residuals:
                ax_res[i].axhline(0, color='black', ls='dashed', lw=1)

        # ---- shared axes cosmetics ---------------------------------------------
        for i, ax in enumerate(ax_main):
            r, c = divmod(i, ncols)
            bottom_of_column = (i + ncols >= nz)
            if residuals:
                ax.tick_params(labelbottom=False)
                ax_res[i].tick_params(labelbottom=bottom_of_column)
            else:
                ax.tick_params(labelbottom=bottom_of_column)
            if c == 0:
                ax.set_ylabel(ylabel)
                if residuals:
                    ax_res[i].set_ylabel(f'(D-T)/{divname}', fontsize='small')
            else:
                ax.tick_params(labelleft=False)
                if residuals:
                    ax_res[i].tick_params(labelleft=False)
            if bottom_of_column:
                (ax_res[i] if residuals else ax).set_xlabel(r'$k_\parallel$ [$\AA^{-1}$]')

        if xlim is not None:
            ax_main[0].set_xlim(xlim)
        if ylim is None and y_used:
            lo, hi = np.min(y_used), np.max(y_used)
            pad = 0.05 * (hi - lo)
            ylim = [lo - pad, hi + pad]
        if ylim is not None:
            ax_main[0].set_ylim(ylim)
        if residuals:
            if ylim2 is None:
                ylim2 = [-0.5, 0.5] if residual_to_theory else [-3, 3]
            ax_res[0].set_ylim(ylim2)
        if title is not None:
            fig.suptitle(title)

        # ---- shared legend --------------------------------------------------
        handles, _ = ax_main[0].get_legend_handles_labels()
        handles.append(plt.Line2D([], [], color='black', marker='^', ls='none', label=datalabel))
        handles.append(plt.Line2D([], [], color='black', ls='solid', lw=2, label=theorylabel))
        if any_unused:
            handles.append(plt.Line2D([], [], color='black', marker='^', ls='none', mfc='none',
                                      alpha=unused_alpha, label='Not used in fit'))
        fig.legend(handles=handles, loc='outside lower center', ncol=ncols_leg, fontsize='small')

        if plot_fname is not None:
            fig.savefig(plot_fname + ".pdf", bbox_inches='tight')
            fig.savefig(plot_fname + ".png", bbox_inches='tight')
        if show:
            plt.show()

    return fig, {'main': ax_main, 'resid': ax_res}
