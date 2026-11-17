# Site discovery: what an SEO playbook has to say to a screener

Opened 2026-11-17 (owner-run). The owner shared a long, tiered list of tactics for growing a
product's search presence with an agent ("S tier: find keywords that bring customers ... D tier:
add llms.txt") and asked to *"take what you can from this article to make the dashboard and all
tools better."* This file is the triage: what applied, what shipped, what is left, and what does
not fit this project and why. Read it before acting on any search, sharing or discovery idea.

## The test applied

The article is written for a business selling a product. This is a free teaching and research tool
whose credibility is the product (CLAUDE.md). A tactic was taken only if it makes the site **easier
to find, share, read or trust without changing what it says** - and nothing was taken that makes
the site promise more than it shows. Every item below was checked against the live site on
2026-11-17 before deciding.

## Already in place before this session (no work needed)

| Article item | Where it already is |
|---|---|
| Specific meta descriptions; title and heading match | All three pages (`generate_dashboard.py`, `lab.html`, `research_page.py`) |
| Sharing image and title | `og:*` + `twitter:card` on every page; `og-card.png` (1200x630, `scripts/make_og_card.py`), deliberately free of any daily figure |
| Footer with guides and product pages; descriptive link text | Every page's footer/nav links the other two in words ("The Factor Lab: the research behind each category, rerun") |
| Link pages with no incoming links | The three pages link each other; the methodology links the lab per category (`run_screener.py`, "Rerun the research") |
| Save observations and calculations; evidence beside findings | The project's core: `calc_trace`, `metric_lineage`, `claims.py` (CLAUDE.md rows 0.8, 0.8c) |
| Explain the sample and method; findings as complete sentences | `research.html`, the lab's per-experiment tolerances |
| Keep research URLs stable | Lab experiments have permanent anchors (`lab.html#piotroski`); stocks open from `#` links |
| Compress images, remove unnecessary scripts, fix mobile | Measured payload work (`dashboard_detail.js` split, 2026-11-08), phone passes, 404 page |
| Compare products on the same criteria | The compare drawer (side by side) |
| Build a free tool with a permanent page | The Factor Lab |

## Shipped 2026-11-17

1. **The publish is checked on the live site.** "Open each published URL to confirm the intended
   change appeared." Until today the data loop logged "live dashboard refreshed" because
   `git push` exited 0; a Pages deployment that failed afterwards (the owner received GitHub's
   failure emails in early November) left the brief saying the site was current.
   `scripts/check_live_site.py` reads the live payload's first 400 bytes (a range request, past the
   CDN cache) and compares its `run_timestamp` with the one just pushed, for up to 20 minutes.
   A stale site is an `ATTENTION - LIVE SITE NOT UPDATED` line in the brief with where to look;
   it never discards a run. `tests/test_live_site_check.py`.
2. **One address per page.** Pages serves the dashboard at `/`, `/index.html` and `/dashboard.html`.
   Each page now names a canonical URL equal to its `og:url`. `tests/test_site_discovery.py`.
3. **`sitemap.xml`** listing exactly the three canonical pages, no `lastmod` (a date not kept true is
   ignored). It does nothing until submitted - see owner actions below.
4. **`llms.txt`**, an index for AI assistants. The dashboard draws its rankings with JavaScript, so a
   reader that does not run it (most AI fetchers, link previewers) gets the page's frame and no
   stocks - measured: ~8 KB of shell text, no ticker. The file points such readers at the
   methodology (`SCREENER_OVERVIEW.md`, served as text by Pages), the changelog and the payload, and
   says it is not investment advice. Every link must resolve to a tracked file.
5. **Structured data that says only what the page says** - one `WebApplication` block on the
   dashboard whose description *is* the meta description. Deliberately absent: `isAccessibleForFree`
   (the page never says it), ratings, offers, author.
6. **The research page's title now matches its heading** ("Does the ranking predict returns?").

## Left for the owner - only he can do these

- **Google Search Console.** Verifying the site needs his Google account. Either the URL-prefix
  property `https://calebsmit.github.io/screener-dashboard/` with the *HTML tag* method - send the
  tag's `content` value to a session and it goes into `generate_dashboard.py`'s head - or the
  *HTML file* method (a session commits the file). Then submit `sitemap.xml`. That is also where
  "pages losing clicks", "queries near page one" and "inspect URL" from the article become possible;
  there is no search data to act on until it exists.
- **Who is behind it.** "Add a real author byline ... publish a factual About page." For a tool whose
  credibility is the product, a reader currently cannot tell who built it beyond a GitHub handle. A
  name, role and one paragraph is his decision, not a session's - it is his name.
- **Analytics.** "Measure internal-link clicks; track sales by landing page." The site has no
  analytics, which keeps it free of third-party scripts and cookies. Adding any (even a cookieless,
  free one) is a privacy and trust choice for him; until he makes it, no session should add a tracker.

## Worth doing later, in order (each needs its own session and evidence)

1. **Per-stock pages** (the article's "programmatic pages"). Today a stock exists only inside the
   single page, so "AAPL factor score" can never be found by search or previewed when shared. A
   static `stock/AAPL.html` per company - its ranks, category scores and the not-advice line, with
   its own title, description and canonical, linking into the dashboard sheet - would be the largest
   discovery gain available. Costs to measure first: ~500 files rewritten every weekday (repo growth
   was measured at ~7.6 MB/month for the payload; re-measure), a fourth publish path for every gate
   (CLAUDE.md "gates" row: every published file must be parsed and claim-checked on both paths), and
   `BANNED_TERMS` over 500 new pages. Not before universe-expansion phase 1, which reshapes
   `stock_detail`.
2. **Citable findings.** "Write each finding as a complete sentence ... keep research URLs stable."
   `research.html`'s headline findings have no anchors of their own; giving each an `id` and a
   copy-link control lets a club cite one sentence. Small; a Friday harden-and-teach item.
3. **A static, readable methodology page.** `SCREENER_OVERVIEW.md` is served raw (and Jekyll renders
   an unstyled `SCREENER_OVERVIEW.html`). A styled, linked HTML version generated from the same
   function would be the page a search for "how is the screener scored" should land on. Check
   `tests/test_overview_is_generated.py` before touching the route.

## Does not fit, and why

- **Keywords "by buying intent", pricing pages, trials, discounts, demos, migration guides,
  switching reasons, competitor comparisons.** There is nothing to buy, and a page arguing this tool
  over others would be marketing on a site whose standing rule is decision support, not promotion.
- **Outreach: roundups, journalist requests, broken-link pitching, guest videos, link exchanges,
  parasite SEO, Reddit/Quora/LinkedIn/Medium posts, G2/Capterra, directories, local SEO.** Each
  publishes or sends something in the owner's name; a session never does that on its own
  (permission rules), and several (link exchanges, parasite SEO) are practices search engines
  penalise. If the owner wants any of them, he does them, and the research notes are the material.
- **"Rising topics" from Google Trends.** The screener's methodology is research-led by rule
  (CLAUDE.md rule 4); chasing search trends is the opposite.
- **`robots.txt`.** Crawlers read it only at the host root (`calebsmit.github.io/robots.txt`, a
  different repository, which returns 404 - so everything is crawlable). One in this repository
  would be served at `/screener-dashboard/robots.txt`, where nothing looks.
