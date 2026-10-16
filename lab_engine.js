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

    // Average, standard deviation (n-1) and t-statistic of a monthly series.
    function meanT(x) {
        const n = x.length;
        if (n < 6) return null;
        let mean = 0; for (const v of x) mean += v; mean /= n;
        let ss = 0; for (const v of x) ss += (v - mean) * (v - mean);
        const sd = Math.sqrt(ss / (n - 1));
        return { mean: mean, std: sd, t: sd ? mean / (sd / Math.sqrt(n)) : null };
    }

    // Each size group's average month over a period, and three differences with their
    // t-statistics: smallest minus largest, and the eighth and ninth groups minus the tenth -
    // the size effect among large companies.
    function sizeRun(size, o) {
        const idx = [];
        for (let i = 0; i < size.months.length; i++) if (size.months[i] >= o.start && size.months[i] <= o.end) idx.push(i);
        const dec = size[o.weighting].map(col => idx.map(i => col[i]));
        const diff = (a, b) => dec[a].map((v, j) => v - dec[b][j]);
        return { months: idx.map(i => size.months[i]), deciles: dec.map(meanT),
                 small_big: meanT(diff(0, 9)), nine_ten: meanT(diff(8, 9)), eight_ten: meanT(diff(7, 9)) };
    }

    // A sorted-portfolio experiment: each group's average month and volatility over a period,
    // and the last group minus the first, month by month, with its t-statistic.
    function sortRun(data, o) {
        const idx = [];
        for (let i = 0; i < data.months.length; i++) if (data.months[i] >= o.start && data.months[i] <= o.end) idx.push(i);
        const groups = data[o.weighting].map(col => idx.map(i => col[i])), last = groups.length - 1;
        return { months: idx.map(i => data.months[i]), groups: groups.map(meanT),
                 spread: meanT(groups[last].map((v, j) => v - groups[0][j])) };
    }

    // Each factor measured separately among small companies and among big ones over a period,
    // the difference between the two halves month by month, and the two sides of the
    // big-company factor: the group it holds minus the market ("long"), and the market minus
    // the group it is short ("short").
    function splitRun(split, o) {
        const idx = [];
        for (let i = 0; i < split.months.length; i++) if (split.months[i] >= o.start && split.months[i] <= o.end) idx.push(i);
        const out = { months: idx.map(i => split.months[i]), factors: {} };
        Object.keys(split).forEach(k => {
            if (k === 'months') return;
            const small = idx.map(i => split[k].small[i]), big = idx.map(i => split[k].big[i]);
            out.factors[k] = { small: meanT(small), big: meanT(big), gap: meanT(small.map((v, j) => v - big[j])),
                               long: meanT(idx.map(i => split[k].long[i])), short: meanT(idx.map(i => split[k].short[i])) };
        });
        return out;
    }

    // Value and momentum together (experiment 13; Asness, Moskowitz & Pedersen 2013): each spread
    // as a year - mean x 12, volatility x sqrt(12), Sharpe ratio - their 50/50 mix, and how the two
    // move together: on the raw spreads, and on what is left after the market's move, which is the
    // correlation the paper reports. o.group is 'big' (the big half) or 'all' (both halves averaged).
    function residuals(y, x) {
        const n = x.length;
        let mx = 0, my = 0;
        for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
        mx /= n; my /= n;
        let sxy = 0, sxx = 0;
        for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); }
        const beta = sxy / sxx;
        return y.map((v, i) => v - beta * x[i]);
    }
    function annualised(x) {
        const s = meanT(x);
        if (!s) return null;
        return { mean: s.mean, std: s.std, t: s.t, mean_a: s.mean * 12, std_a: s.std * Math.sqrt(12),
                 sharpe: s.std ? s.mean / s.std * Math.sqrt(12) : null };
    }
    function comboRun(split, factors, o) {
        const mk = {};
        factors.months.forEach((m, i) => { mk[m] = factors.mktrf[i]; });
        const idx = [];
        for (let i = 0; i < split.months.length; i++) {
            const m = split.months[i];
            if (m >= o.start && m <= o.end && Object.prototype.hasOwnProperty.call(mk, m)) idx.push(i);
        }
        const series = key => idx.map(i => o.group === 'big' ? split[key].big[i] : (split[key].small[i] + split[key].big[i]) / 2);
        const v = series('value'), mo = series('momentum'), mkt = idx.map(i => mk[split.months[i]]);
        const out = { months: idx.map(i => split.months[i]), value: annualised(v), momentum: annualised(mo),
                      combo: annualised(v.map((a, j) => (a + mo[j]) / 2)), corr: null, corr_resid: null,
                      series: { value: v, momentum: mo, combo: v.map((a, j) => (a + mo[j]) / 2) } };
        if (idx.length >= 6) {
            out.corr = correlation(v, mo);
            out.corr_resid = correlation(residuals(v, mkt), residuals(mo, mkt));
        }
        return out;
    }

    // ---------------------------------------------------------------- "Against the paper"
    // Each experiment's comparison with its paper, computed here the way factor_lab.py's
    // reproduce_* functions compute it, so the page fills the "Here" column from its own
    // arithmetic; tests/test_factor_lab.py holds every row to the build's at 1e-9.
    function compareFigures(figures, got) {
        return figures.map(f => {
            const here = got[f.key], diff = here - f.paper;
            return Object.assign({}, f, { here: here, diff: diff, within: Math.abs(diff) <= f.tolerance });
        });
    }

    // Experiment 2 (reproduce_crashes / crash_figures): momentum's and the market's moments, the
    // market model with and without the bear-market indicator, the paper's Table 5 regression, and
    // the worst months against the paper's list.
    function compareCrashes(mom, market, ref) {
        const idx = [];
        for (let i = 0; i < mom.months.length; i++) if (mom.months[i] >= ref.start && mom.months[i] <= ref.end) idx.push(i);
        const months = idx.map(i => mom.months[i]), wml = idx.map(i => mom.wml[i]), mkt = idx.map(i => market.mktrf[i]);
        const bear = idx.map(i => market.trail['24'][i] === null ? null : (market.trail['24'][i] < 0 ? 1 : 0));
        const ws = stats(wml), ms = stats(mkt), ones = n => new Array(n).fill(1);
        const c0 = ols(wml, [ones(wml.length), mkt]);
        const keep = []; bear.forEach((b, k) => { if (b !== null) keep.push(k); });
        const y = keep.map(k => wml[k]), bb = keep.map(k => bear[k]), mm = keep.map(k => mkt[k]);
        const cb = ols(y, [ones(y.length), bb, mm, bb.map((u, j) => u * mm[j])]);
        const g = keep.filter(k => months[k] >= ref.gamma_start);
        const cg = ols(g.map(k => wml[k]), [ones(g.length), g.map(k => bear[k])]);
        const ranked = wml.map((r, k) => [r, months[k]])
            .sort((a, b) => a[0] - b[0] || (a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0)).slice(0, ref.worst.length);
        const paper = new Set(ref.worst.map(w => w[0]));
        return compareFigures(ref.figures, {
            wml_mean: ws.mean, wml_std: ws.std, wml_sharpe: ws.sharpe, mkt_mean: ms.mean, mkt_std: ms.std,
            beta0: c0[1], alpha0: c0[0], beta_bear: cb[3], alpha_bear: cb[1],
            bear_months: bear.filter(b => b === 1).length, gamma0: cg[0], gamma_bear: cg[1],
            worst1: ranked[0][0], worst2: ranked[1][0], worst_overlap: ranked.filter(r => paper.has(r[1])).length });
    }

    // Experiment 3 (reproduce_factors / factor_figures).
    function compareFactors(factors, keys, ref) {
        const r = factorRun(factors, keys, { start: ref.start, end: ref.end }), got = {};
        keys.forEach(k => { got['mean_' + k] = r.stats[k].mean; got['std_' + k] = r.stats[k].std; got['t_' + k] = r.stats[k].t; });
        got.corr_hml_cma = r.corr.hml.cma;
        got.hml_intercept = r.spanned.hml.intercept; got.hml_on_cma = r.spanned.hml.slopes.cma; got.hml_r2 = r.spanned.hml.r2;
        got.rmw_intercept = r.spanned.rmw.intercept; got.cma_intercept = r.spanned.cma.intercept;
        return compareFigures(ref.figures, got);
    }

    // Experiment 4 (reproduce_size).
    function compareSize(size, ref) {
        const r = sizeRun(size, { start: ref.start, end: ref.end, weighting: ref.weighting }), got = { small_big: r.small_big.mean };
        r.deciles.forEach((d, i) => { got['dec' + (i + 1)] = d.mean; });
        return compareFigures(ref.figures, got);
    }

    // A sorted-portfolio experiment (reproduce_sort). `excess`: the paper reports returns over the
    // T-bill, so each group's average loses the T-bill's average over the same months; `relative`:
    // each group against the average of the groups; `scale`: the paper's unit (12 = % a year).
    function compareSort(data, ref) {
        const r = sortRun(data, { start: ref.start, end: ref.end, weighting: ref.weighting }), scale = ref.scale || 1;
        let centre = 0;
        if (ref.relative) { for (const g of r.groups) centre += g.mean; centre /= r.groups.length; }
        else if (ref.excess) {
            let s = 0, k = 0;
            for (let i = 0; i < data.months.length; i++) {
                const m = data.months[i];
                if (m >= ref.start && m <= ref.end && data.rf[i] !== null) { s += data.rf[i]; k++; }
            }
            centre = s / k;
        }
        const got = { spread: r.spread.mean * scale };
        r.groups.forEach((g, i) => { got['mean' + (i + 1)] = (g.mean - centre) * scale; got['std' + (i + 1)] = g.std; });
        return compareFigures(ref.figures, got);
    }

    // A two-group experiment (reproduce_split): each part's average month, in the paper's unit, and
    // its t-statistic.
    function compareSplit(split, ref) {
        const r = splitRun(split, { start: ref.start, end: ref.end }), scale = ref.scale || 1, got = {};
        Object.keys(r.factors).forEach(k => Object.keys(r.factors[k]).forEach(part => {
            const s = r.factors[k][part];
            got['mean_' + k + '_' + part] = s.mean * scale; got['t_' + k + '_' + part] = s.t;
        }));
        return compareFigures(ref.figures, got);
    }

    // Value and momentum together (reproduce_combo).
    function compareCombo(split, factors, ref) {
        const r = comboRun(split, factors, { start: ref.start, end: ref.end, group: ref.group }), got = { corr_resid: r.corr_resid };
        ['value', 'momentum', 'combo'].forEach(k => { got['mean_' + k] = r[k].mean_a; got['std_' + k] = r[k].std_a; got['sharpe_' + k] = r[k].sharpe; });
        return compareFigures(ref.figures, got);
    }

    return { stats: stats, run: run, compare: compare, growth: growth, deepestFall: deepestFall,
             crashRun: crashRun, ols: ols, factorRun: factorRun, sizeRun: sizeRun, sortRun: sortRun,
             splitRun: splitRun, comboRun: comboRun, compareCrashes: compareCrashes,
             compareFactors: compareFactors, compareSize: compareSize, compareSort: compareSort,
             compareSplit: compareSplit, compareCombo: compareCombo };
});
