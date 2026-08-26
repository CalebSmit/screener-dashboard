# Implementation Plan: Dashboard Frontend Security, Performance & Polish

## Task Type
- [x] Frontend (dashboard.html — both generated output and generate_dashboard.py template)
- [x] Backend (generate_dashboard.py — Python data generation)

## Technical Solution

Surgical fixes across 5 categories: XSS prevention, input debouncing, UX polish, Python code quality, and CSS consistency. All changes are in 2 files: `dashboard.html` (generated output) and `generate_dashboard.py` (template source). Both must be changed in lockstep.

---

## Implementation Steps

### Step 1: Add `escapeHtml()` utility function (Security)

**File**: `dashboard.html` (line ~3111, after `fmt()` function)
**File**: `generate_dashboard.py` (in the JS section of `generate_html()`)

Add a sanitization helper:

```javascript
function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
```

### Step 2: Sanitize all innerHTML injections (Security)

Apply `escapeHtml()` to all user-data interpolated into HTML strings:

**Universe table** (~line 3577-3590):
- `${row.Ticker}` in onclick → `${escapeHtml(row.Ticker)}`
- `${row.Company || ''}` → `${escapeHtml(row.Company || '')}`
- `${row.Sector}` → `${escapeHtml(row.Sector)}`

**Top 5 cards** (~line 3191-3206):
- `${h.ticker}` in onclick → `${escapeHtml(h.ticker)}`
- `${h.company}` → `${escapeHtml(h.company)}`
- `${h.sector}` → `${escapeHtml(h.sector)}`

**Portfolio table** (~line 3234-3248):
- `h.ticker` in onclick → `escapeHtml(h.ticker)`
- `h.company` → `escapeHtml(h.company)`
- `h.sector` → `escapeHtml(h.sector)`

**Peer search dropdown** (~line 3483-3488):
- `${r.Ticker}` in onmousedown → `${escapeHtml(r.Ticker)}`
- `${r.Company || ''}` → `${escapeHtml(r.Company || '')}`

**Peer comparison table** (~line 3933-3946):
- `${r.ticker}` → `${escapeHtml(r.ticker)}`

**Stock detail modal** (~line 3621-3623):
- These use `textContent` — already safe. No change needed.

**Chat AI context builders** (~line 4369-4415):
- These build plain text for API calls, not HTML — no change needed.

### Step 3: Sanitize AI chat response HTML (Security)

**File**: `dashboard.html` (~line 4289, `parseChatMd()`)

The current function converts markdown to HTML but doesn't strip dangerous tags. Add a pre-sanitization step before markdown parsing:

```javascript
function parseChatMd(text) {
    // Strip any HTML tags from the AI response first
    var clean = text.replace(/<[^>]*>/g, '');
    return clean
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/^[-*]\s+(.+)$/gm, '<li>$1</li>')
        .replace(/(<li>[\s\S]*?<\/li>)/g, function(m) { return '<ul>' + m + '</ul>'; })
        .replace(/<\/ul>\s*<ul>/g, '')
        .replace(/\n/g, '<br>');
}
```

### Step 4: Add debounce utility and apply to filters (Performance)

**File**: `dashboard.html` (~line 3111, utilities section)

Add debounce helper:

```javascript
function debounce(fn, ms) {
    let timer;
    return function() {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, arguments), ms);
    };
}
```

**File**: `dashboard.html` (~line 3513-3514, `setupFilters()`)

Change:
```javascript
document.getElementById('filter-comp-min').addEventListener('input', applyFilters);
document.getElementById('filter-search').addEventListener('input', applyFilters);
```
To:
```javascript
document.getElementById('filter-comp-min').addEventListener('input', debounce(applyFilters, 200));
document.getElementById('filter-search').addEventListener('input', debounce(applyFilters, 200));
```

Note: Sector and trap flag selects stay immediate (dropdown change is a single action).

### Step 5: Replace `index.html` with lightweight redirect (UX)

**File**: `index.html`

Replace entire 238KB file with:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="0; url=dashboard.html">
    <title>Redirecting...</title>
</head>
<body>
    <p>Redirecting to <a href="dashboard.html">dashboard</a>...</p>
</body>
</html>
```

### Step 6: Fix print styles (UX)

**File**: `dashboard.html` (~line 2061-2070, `@media print` block)

Add rules to hide interactive-only elements:

```css
@media print {
    body { background: #fff; color: #000; }
    :root {
        --bg-primary: #fff; --bg-card: #fff; --bg-elevated: #f5f5f5;
        --text-primary: #000; --text-secondary: #555; --border: #ddd;
    }
    .dashboard-container { max-width: none; }
    .filters-bar { display: none; }
    .chat-fab, .chat-panel { display: none !important; }
    .modal-overlay { display: none !important; }
    .methodology-btn { display: none; }
    .kpi-card, .chart-container, .table-section { box-shadow: none; border: 1px solid #ddd; }
    .collapsible-section.collapsed .section-body { max-height: none; opacity: 1; }
}
```

### Step 7: Fix bare `except Exception` blocks in Python (Code Quality)

**File**: `generate_dashboard.py`

**Lines 63-64 and 100-101** (in `_find_latest_run` and `_find_raw_fetch`):
These are sorting helpers where an unreadable meta.json should sort last. Keep as-is — returning `""` is correct behavior for sorting. But add a comment:

```python
except Exception:  # Unreadable meta.json → sorts last
    return ""
```

**Lines 177-178** (sensitivity parquet):
```python
except (OSError, ValueError) as e:
    sens_df = None  # Graceful fallback — section simply won't render
```

**Lines 184-186** (correlation parquet):
```python
except (OSError, ValueError) as e:
    corr_df = None  # Graceful fallback — section simply won't render
```

**Lines 193-195** (portfolio parquet):
```python
except (OSError, ValueError) as e:
    port_df = None  # Graceful fallback — uses naive top-25 instead
```

### Step 8: Fix `== False` pandas comparison (Code Quality)

**File**: `generate_dashboard.py` (lines 228, 230)

Change:
```python
eligible = df[df["Value_Trap_Flag"] == False].copy()
...
eligible = eligible[eligible["Growth_Trap_Flag"] == False]
```

To:
```python
eligible = df[~df["Value_Trap_Flag"].fillna(False)].copy()
...
eligible = eligible[~eligible["Growth_Trap_Flag"].fillna(False)]
```

### Step 9: Replace hardcoded font-family strings with CSS variables (CSS)

**File**: `generate_dashboard.py` (`_css()` function, lines ~4540-4948)

Replace all instances of:
- `font-family: 'Space Grotesk', sans-serif` → `font-family: var(--font-heading)`
- `font-family: 'DM Sans', sans-serif` → `font-family: var(--font-body)`
- `font-family: 'JetBrains Mono', monospace` → `font-family: var(--font-mono)`

There are ~21 occurrences in the `_css()` section of the template. The CSS variables are already defined in `:root`.

**File**: `dashboard.html` (same locations in the output file)

Same replacements in the already-generated HTML.

### Step 10: Apply all template changes to `generate_dashboard.py`

Every change made to `dashboard.html` (steps 1-6, 9) must also be applied to the corresponding f-string template in `generate_dashboard.py`'s `generate_html()` function and `_css()` function, with proper `{{`/`}}` brace escaping for the Python f-string.

---

## Key Files

| File | Operation | Description |
|------|-----------|-------------|
| `dashboard.html` | Modify | Add escapeHtml(), debounce(), sanitize innerHTML, fix print CSS |
| `generate_dashboard.py` | Modify | Fix except blocks, == False, update template to match dashboard.html |
| `index.html` | Rewrite | Replace 238KB duplicate with 200-byte redirect |

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| `escapeHtml()` breaks onclick handlers with tickers containing special chars | Tickers are uppercase alpha only (S&P 500), but escaping `'` to `&#39;` prevents breakage for any edge case |
| Debounce delay feels sluggish | 200ms is imperceptible for typing; dropdown selects remain instant |
| `parseChatMd` HTML stripping removes legitimate formatting from AI | AI responses use markdown, not HTML. Stripping `<tags>` before markdown→HTML conversion is safe |
| Template drift between dashboard.html and generate_dashboard.py | Step 10 explicitly addresses this; after changes, regenerate dashboard to verify |

## Verification

After all changes:
1. Run `python generate_dashboard.py` to regenerate dashboard.html
2. Open dashboard in browser — verify all sections render
3. Test: click stock detail, sort table, filter by sector, use search
4. Test: open AI chat, send a message
5. Verify index.html redirects properly
6. Print preview — verify chat FAB and modals are hidden

## SESSION_ID
- CODEX_SESSION: N/A (direct planning)
- GEMINI_SESSION: N/A (direct planning)
