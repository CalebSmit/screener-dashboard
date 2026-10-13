// Factor Lab - the arithmetic the page runs when a reader changes a setting.
//
// The same calculations as factor_lab.py, which is the reference: tests/test_factor_lab.py
// runs this file under node on the published data and requires the two to agree, at the
// paper's settings and away from them. Change a formula in one and the suite fails until
// the other matches. Plain functions, no dependencies, so the page can also print this
// source beside the chart it drew.
(function (root, factory) {
    if (typeof module === 'object' && module.exports) module.exports = factory();
    else root.LabEngine = factory();
})(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    // The paper's Table 3 columns for monthly returns in percent: best and worst month,
    // annualised mean (12 x) and standard deviation (12^0.5 x, n-1), excess kurtosis and
    // skewness (plain moments), and the Sharpe ratio (mean / standard deviation; these are
    // zero-investment portfolios, so there is no risk-free rate to subtract).
    function stats(x) {
        const n = x.length;
        if (n < 3) return null;
        let sum = 0, max = -Infinity, min = Infinity;
        for (const v of x) { sum += v; if (v > max) max = v; if (v < min) min = v; }
        const mean = sum / n;
        let ss = 0;
        for (const v of x) ss += (v - mean) * (v - mean);
        const sd = Math.sqrt(ss / (n - 1)), pop = Math.sqrt(ss / n);
        if (pop === 0) return null;
        let s3 = 0, s4 = 0;
        for (const v of x) { const z = (v - mean) / pop; s3 += z * z * z; s4 += z * z * z * z; }
        const amean = 12 * mean, asd = Math.sqrt(12) * sd;
        return { n: n, max: max, min: min, mean: amean, std: asd,
                 kurt: s4 / n - 3, skew: s3 / n, sharpe: amean / asd };
    }

    // Plain and risk-managed momentum between two months (YYYYMM, inclusive).
    // Each month's exposure is target / forecast: the target annualised volatility over the
    // volatility forecast made from earlier months' daily returns, optionally capped (an
    // investor who cannot borrow has a cap of 1). A month with no forecast yet is left out
    // of both series, so the two are always compared over the same months.
    function run(data, o) {
        const vol = data.vol[String(o.window)];
        const months = [], plain = [], managed = [], weights = [];
        for (let i = 0; i < data.months.length; i++) {
            const m = data.months[i], v = vol[i];
            if (m < o.start || m > o.end || v === null || !(v > 0)) continue;
            let w = o.target / v;
            if (o.cap !== null && o.cap !== undefined) w = Math.min(w, o.cap);
            months.push(m); plain.push(data.wml[i]); managed.push(w * data.wml[i]); weights.push(w);
        }
        let wmin = Infinity, wmax = -Infinity, wsum = 0;
        for (const w of weights) { if (w < wmin) wmin = w; if (w > wmax) wmax = w; wsum += w; }
        return { months: months, plain: plain, managed: managed, weights: weights,
                 plain_stats: stats(plain), managed_stats: stats(managed),
                 weight_stats: weights.length ? { min: wmin, max: wmax, mean: wsum / weights.length } : null };
    }

    // One row per figure the paper reports: its value, ours, the difference, and whether
    // the difference is inside the tolerance written down for that figure.
    function compare(reference, measured, tolerances, labels) {
        const rows = [];
        for (const leg of ['plain', 'managed']) {
            for (const key of ['max', 'min', 'mean', 'std', 'kurt', 'skew', 'sharpe']) {
                const ref = reference[leg][key], got = measured[leg + '_stats'][key], diff = got - ref;
                rows.push({ leg: leg, stat: key, label: labels[key], paper: ref, here: got, diff: diff,
                            tolerance: tolerances[key], within: Math.abs(diff) <= tolerances[key] });
            }
        }
        return rows;
    }

    // What one dollar becomes, month by month, compounding the returns (percent).
    function growth(returnsPct) {
        const out = []; let v = 1;
        for (const r of returnsPct) { v *= 1 + r / 100; out.push(v); }
        return out;
    }

    // The deepest fall of that compounded value below its own earlier high, as a fraction.
    function deepestFall(returnsPct) {
        let v = 1, peak = 1, worst = 0;
        for (const r of returnsPct) {
            v *= 1 + r / 100;
            if (v > peak) peak = v;
            if (v / peak - 1 < worst) worst = v / peak - 1;
        }
        return worst;
    }

    return { stats: stats, run: run, compare: compare, growth: growth, deepestFall: deepestFall };
});
