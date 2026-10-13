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

    // Momentum's months sorted into four market states, each known before the month begins.
    // "Down": the market's compounded return over the previous `lookback` months is negative.
    // "Volatile": its volatility over the previous `window` trading days is above `threshold`
    // a year. Down and volatile together is the panic state. A month with no reading for
    // either is left out.
    function crashRun(mom, market, o) {
        const trail = market.trail[String(o.lookback)], vol = market.vol[String(o.window)];
        const months = [], wml = [], state = [];
        for (let i = 0; i < mom.months.length; i++) {
            const m = mom.months[i];
            if (m < o.start || m > o.end || trail[i] === null || vol[i] === null) continue;
            months.push(m); wml.push(mom.wml[i]);
            state.push((trail[i] < 0 ? 2 : 0) + (vol[i] > o.threshold ? 1 : 0));
        }
        const n = months.length, cells = [];
        let all = 0, rest = 0, restN = 0;
        for (let s = 0; s < 4; s++) {
            let k = 0, sum = 0, worst = Infinity, neg = 0;
            for (let j = 0; j < n; j++) {
                if (state[j] !== s) continue;
                k++; sum += wml[j];
                if (wml[j] < worst) worst = wml[j];
                if (wml[j] < 0) neg++;
            }
            all += sum;
            if (s !== 3) { rest += sum; restN += k; }
            cells.push({ state: s, n: k, share: n ? k / n : 0, mean: k ? sum / k : null,
                         worst: k ? worst : null, losing: k ? neg / k : null });
        }
        return { months: months, wml: wml, state: state, cells: cells,
                 all_mean: n ? all / n : null, rest_mean: restN ? rest / restN : null };
    }

    // Least-squares coefficients of y on the given columns (include a column of ones for an
    // intercept), by the normal equations.
    function ols(y, columns) {
        const k = columns.length, a = [];
        for (let i = 0; i < k; i++) {
            const row = [];
            for (let j = 0; j < k; j++) { let s = 0; for (let n = 0; n < y.length; n++) s += columns[i][n] * columns[j][n]; row.push(s); }
            let sy = 0; for (let n = 0; n < y.length; n++) sy += columns[i][n] * y[n];
            row.push(sy); a.push(row);
        }
        for (let i = 0; i < k; i++) {
            let p = i;
            for (let r = i + 1; r < k; r++) if (Math.abs(a[r][i]) > Math.abs(a[p][i])) p = r;
            const tmp = a[i]; a[i] = a[p]; a[p] = tmp;
            for (let r = 0; r < k; r++) {
                if (r === i) continue;
                const f = a[r][i] / a[i][i];
                for (let c = 0; c <= k; c++) a[r][c] -= f * a[i][c];
            }
        }
        return a.map((row, i) => row[k] / row[i]);
    }

    function correlation(x, y) {
        const n = x.length;
        let mx = 0, my = 0;
        for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
        mx /= n; my /= n;
        let sxy = 0, sxx = 0, syy = 0;
        for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); syy += (y[i] - my) * (y[i] - my); }
        return sxy / Math.sqrt(sxx * syy);
    }

    // Each factor's average month, volatility, t-statistic (the average over its standard
    // error) and compounded growth over a period; their correlations; and each factor
    // regressed on the other four - the intercept is what the others leave unexplained.
    function factorRun(factors, keys, o) {
        const idx = [];
        for (let i = 0; i < factors.months.length; i++) if (factors.months[i] >= o.start && factors.months[i] <= o.end) idx.push(i);
        const months = idx.map(i => factors.months[i]), n = months.length, series = {}, st = {}, spanned = {}, corr = {};
        keys.forEach(k => { series[k] = idx.map(i => factors[k][i]); });
        keys.forEach(k => {
            const x = series[k];
            if (n < 6) { st[k] = null; return; }
            let mean = 0; for (const v of x) mean += v; mean /= n;
            let ss = 0; for (const v of x) ss += (v - mean) * (v - mean);
            const sd = Math.sqrt(ss / (n - 1)), g = growth(x);
            st[k] = { mean: mean, std: sd, t: sd ? mean / (sd / Math.sqrt(n)) : null, growth: g[g.length - 1] };
        });
        if (n >= 12) keys.forEach(k => {
            const others = keys.filter(q => q !== k), ones = new Array(n).fill(1);
            const coef = ols(series[k], [ones].concat(others.map(q => series[q])));
            let my = 0; for (const v of series[k]) my += v; my /= n;
            let sst = 0, sse = 0;
            for (let j = 0; j < n; j++) {
                let fit = coef[0];
                others.forEach((q, c) => { fit += coef[c + 1] * series[q][j]; });
                sst += (series[k][j] - my) * (series[k][j] - my); sse += (series[k][j] - fit) * (series[k][j] - fit);
            }
            const slopes = {}; others.forEach((q, c) => { slopes[q] = coef[c + 1]; });
            spanned[k] = { intercept: coef[0], slopes: slopes, r2: sst ? 1 - sse / sst : null };
        });
        keys.forEach(a => { corr[a] = {}; keys.forEach(b => { corr[a][b] = n >= 6 ? correlation(series[a], series[b]) : null; }); });
        return { months: months, series: series, stats: st, corr: corr, spanned: spanned };
    }

    return { stats: stats, run: run, compare: compare, growth: growth, deepestFall: deepestFall,
             crashRun: crashRun, ols: ols, factorRun: factorRun };
});
