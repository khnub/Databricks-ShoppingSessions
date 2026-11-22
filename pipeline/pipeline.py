#!/usr/bin/env python3
"""
Shopping Sessions: DuckDB pipeline.

Stage 1: read every input file exactly as rules.md's "CSV dialect" says.
Later stages (links, events, sessions, clicks, outputs) build on these tables.

Reads  $DATA_DIR    (default /data)
Writes $OUTPUT_DIR  (default /output): user_daily/ and campaign_daily/ as Parquet
Temp   $SCRATCH_DIR (default /scratch)
Set DEBUG=1 to print every parsed row (only sensible on the small fixtures).
"""
import datetime
import glob
import json
import os
import sys

import duckdb

DATA = os.environ.get("DATA_DIR", "/data")
OUT = os.environ.get("OUTPUT_DIR", "/output")
SCRATCH = os.environ.get("SCRATCH_DIR", "/scratch")
NCPU = int(os.environ.get("NCPU", os.cpu_count() or 1))
MEM_GB = int(os.environ.get("MEM_GB", "8"))
DEBUG = os.environ.get("DEBUG") == "1"


def log(*args):
    print(*args, file=sys.stderr)  # stdout is capped at 50 MB, so log to stderr


con = duckdb.connect()  # in-memory database
con.execute(f"SET threads = {NCPU}")
con.execute(f"SET memory_limit = '{max(1, int(MEM_GB * 0.8))}GB'")  # headroom for Python
con.execute(f"SET temp_directory = '{os.path.join(SCRATCH, 'duckdb_tmp')}'")
con.execute("SET preserve_insertion_order = false")

with open(os.path.join(DATA, "manifest.json")) as f:
    manifest = json.load(f)
log("window:", manifest["window_start"], "to", manifest["window_end"])


# ---------------------------------------------------------------------------
# Stage 1: the CSV dialect
#
# We do NOT use DuckDB's read_csv: it treats \r\n as a line ending, repairs
# some malformed lines, and reads "" as '' instead of null. Instead we read
# each file as raw text, split on \n only, and check every line ourselves.
#
# A field is either bare (no quote, no comma) or wrapped whole in quotes with
# no quote or comma inside. Any other quote makes the line malformed.
# ---------------------------------------------------------------------------
FIELD = '(?:"[^",]*"|[^",]*)'

# unq(field): strip the wrapping quotes if present; an empty value means null.
# Nothing else is touched: spaces and \r stay part of the value.
con.execute("""
    CREATE MACRO unq(x) AS NULLIF(
        CASE WHEN starts_with(x, '"') THEN substr(x, 2, length(x) - 2) ELSE x END,
        '')
""")


def sql_str(s):
    return "'" + s.replace("'", "''") + "'"


def load(table, files, columns):
    """Create `table` (all VARCHAR columns) from `files`, following the dialect.

    - Only \\n ends a line; the first line of each file is the header.
    - A line must have exactly len(columns) fields, each bare or wholly quoted.
      Anything else (wrong field count, stray quote, empty line) is dropped.
    - Wrapping quotes are removed; empty fields become NULL.
    """
    if not files:
        cols = ", ".join(f"NULL::VARCHAR AS {c}" for c in columns)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT {cols} WHERE false")
        return
    n = len(columns)
    line_pattern = FIELD + f"(?:,{FIELD}){{{n - 1}}}"  # exactly n fields
    file_list = "[" + ", ".join(sql_str(f) for f in files) + "]"
    cols = ", ".join(f"unq(f[{i + 1}]) AS {c}" for i, c in enumerate(columns))
    con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        WITH lines AS (
            -- [2:] skips the header, which is only ever the first line of a file
            SELECT unnest(string_split(content, chr(10))[2:]) AS line
            FROM read_text({file_list})
        )
        SELECT {cols}
        FROM (
            SELECT string_split(line, ',') AS f
            FROM lines
            WHERE regexp_full_match(line, {sql_str(line_pattern)})
        )
    """)


def csvs(sub):
    return sorted(glob.glob(os.path.join(DATA, sub, "*.csv")))


load("raw_events", csvs("events"),
     ["event_id", "user_id", "ts", "ingest_ts", "event_type", "product_id", "price_cents"])
load("raw_clicks", csvs("clicks"), ["click_id", "user_id", "campaign_id", "ts"])
load("raw_links", csvs("links"), ["user_id_a", "user_id_b"])
load("products", [os.path.join(DATA, "products.csv")], ["product_id", "category"])
load("campaigns", [os.path.join(DATA, "campaigns.csv")], ["campaign_id", "channel"])

for t in ["raw_events", "raw_clicks", "raw_links", "products", "campaigns"]:
    count = con.sql(f"SELECT count(*) FROM {t}").fetchone()[0]
    log(f"{t}: {count} rows")
    if DEBUG:
        for row in con.sql(f"SELECT * FROM {t}").fetchall():
            log("   ", row)  # repr shows \r, \t, spaces and None clearly


# ---------------------------------------------------------------------------
# Stage 2: linked IDs -> people
#
# Every link line with both sides non-empty joins its two IDs into one person,
# and joining is transitive (a-b and b-c make a, b, c one person). That is the
# "connected components" problem; union-find solves it in near-linear time,
# however long the chains are. SQL is bad at this, so we do it in Python.
#
# Each person is named by the byte-wise smallest of their IDs. Python compares
# str by Unicode code point, which for UTF-8 is the same order as bytes.
#
# Result: table person_map(user_id, person) for every ID that appears in a
# link. An ID not in person_map is a person on its own, named by itself.
# ---------------------------------------------------------------------------
pairs = con.execute("""
    SELECT DISTINCT user_id_a, user_id_b FROM raw_links
    WHERE user_id_a IS NOT NULL AND user_id_b IS NOT NULL
""").fetchall()

parent = {}
size = {}


def find(x):
    # Walk up to the root, then point everything on the path straight at it
    # (path compression). Iterative, so chains hundreds long are fine.
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


for a, b in pairs:
    for x in (a, b):
        if x not in parent:
            parent[x] = x
            size[x] = 1
    ra, rb = find(a), find(b)
    if ra != rb:
        if size[ra] < size[rb]:  # union by size: attach the smaller tree
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

name = {}  # root -> smallest ID in its group
for x in parent:
    r = find(x)
    if r not in name or x < name[r]:
        name[r] = x

ids = list(parent)
people = [name[find(x)] for x in ids]
con.execute("""
    CREATE OR REPLACE TABLE person_map AS
    SELECT unnest(?::VARCHAR[]) AS user_id, unnest(?::VARCHAR[]) AS person
""", [ids, people])
log(f"links: {len(pairs)} distinct pairs, {len(ids)} linked IDs, {len(name)} linked people")
if DEBUG:
    for row in con.sql("SELECT * FROM person_map ORDER BY person, user_id").fetchall():
        log("   ", row)

# Later stages find a row's person with:
#   LEFT JOIN person_map pm ON pm.user_id = x.user_id
#   ... COALESCE(pm.person, x.user_id) AS person


# ---------------------------------------------------------------------------
# Stage 3a: timestamps and prices
#
# ts_ms(s) turns a timestamp string into integer milliseconds since
# 1970-01-01 UTC, or NULL if it is a bad timestamp. We do the calendar maths
# ourselves instead of using DuckDB's timestamp parser, so that exactly the
# three forms in rules.md are accepted and nothing else:
#   F1  2026-10-03T23:58:10Z
#   F2  2026-10-03T19:58:10-04:00
#   F3  1791071890000            (digits only, epoch ms)
# [0-9] is used instead of \d so only ASCII digits count.
# ---------------------------------------------------------------------------
con.execute("CREATE MACRO is_leap(y) AS (y % 4 = 0 AND y % 100 <> 0) OR y % 400 = 0")
con.execute("""
    CREATE MACRO days_in_month(y, m) AS
        CASE WHEN m = 2 THEN CASE WHEN is_leap(y) THEN 29 ELSE 28 END
             WHEN m IN (4, 6, 9, 11) THEN 30
             ELSE 31 END
""")
# Days from 1970-01-01 to a calendar date (Howard Hinnant's days_from_civil,
# simplified for years >= 0). Treats Jan/Feb as months 13/14 of the previous
# year so the leap day falls at the end. `//` is integer division in DuckDB.
con.execute("""
    CREATE MACRO days_civil(y, m, d) AS
        365 * (y - (m <= 2)::INT) + (y - (m <= 2)::INT) // 4
        - (y - (m <= 2)::INT) // 100 + (y - (m <= 2)::INT) // 400
        + (153 * (m + CASE WHEN m > 2 THEN -3 ELSE 9 END) + 2) // 5
        + d - 1 - 719468
""")
# Pull a fixed-position number out of an F1/F2 string.
con.execute("CREATE MACRO num(s, i, n) AS TRY_CAST(substr(s, i, n) AS BIGINT)")
# F1/F2 -> ms, or NULL for an impossible date/time or offset.
#   positions: YYYY-MM-DDTHH:MM:SS  then  Z  or  +HH:MM
#              1    6  9  12 15 18      20     20 21 24
con.execute("""
    CREATE MACRO iso_ms(s) AS
        CASE WHEN num(s, 6, 2) BETWEEN 1 AND 12
              AND num(s, 9, 2) BETWEEN 1 AND days_in_month(num(s, 1, 4), num(s, 6, 2))
              AND num(s, 12, 2) <= 23 AND num(s, 15, 2) <= 59 AND num(s, 18, 2) <= 59
              AND (substr(s, 20, 1) = 'Z' OR (num(s, 21, 2) <= 23 AND num(s, 24, 2) <= 59))
        THEN (days_civil(num(s, 1, 4), num(s, 6, 2), num(s, 9, 2)) * 86400
              + num(s, 12, 2) * 3600 + num(s, 15, 2) * 60 + num(s, 18, 2)
              -- local time = UTC + offset, so subtract the offset to get UTC
              - CASE substr(s, 20, 1)
                    WHEN '+' THEN num(s, 21, 2) * 3600 + num(s, 24, 2) * 60
                    WHEN '-' THEN -(num(s, 21, 2) * 3600 + num(s, 24, 2) * 60)
                    ELSE 0 END
             ) * 1000
        END
""")
con.execute("""
    CREATE MACRO ts_ms(s) AS
        CASE
            -- F3: digits only. Leading zeros are fine. A value too big for a
            -- 64-bit integer becomes NULL, which is fine: it's outside any window.
            WHEN regexp_full_match(s, '[0-9]+') THEN
                TRY_CAST(CASE WHEN ltrim(s, '0') = '' THEN '0' ELSE ltrim(s, '0') END AS BIGINT)
            -- F1 / F2: exact shape first, then calendar checks in iso_ms
            WHEN regexp_full_match(s,
                '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:Z|[+-][0-9]{2}:[0-9]{2})')
            THEN iso_ms(s)
            ELSE NULL  -- empty, fractional seconds, lowercase z/t, space, no zone...
        END
""")
# A purchase price: digits only, value 1 to 10,000,000. Leading zeros allowed.
# After stripping zeros, more than 8 digits is always over 10,000,000.
con.execute("""
    CREATE MACRO price_value(x) AS
        CASE WHEN regexp_full_match(x, '[0-9]+')
              AND length(ltrim(x, '0')) BETWEEN 1 AND 8
              AND TRY_CAST(ltrim(x, '0') AS BIGINT) <= 10000000
        THEN TRY_CAST(ltrim(x, '0') AS BIGINT)
        END
""")


def iso_to_ms(s):
    """manifest.json window bounds -> epoch ms (they are always clean F1)."""
    return int(datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)


WS = iso_to_ms(manifest["window_start"])  # event window: WS <= ts < WE
WE = iso_to_ms(manifest["window_end"])
DAY_MS = 86_400_000


# ---------------------------------------------------------------------------
# Stage 3b: events, in the order rules.md gives
#   1. drop lines with an empty event_id          (malformed ones: stage 1)
#   2. dedup on event_id: keep the smallest ingest_ts, BEFORE validating
#   3. validate the kept copy
#   4. drop late events: ingest_ts - ts over 24 h
# Each surviving event gets its person (stage 2) and its product's category.
# ---------------------------------------------------------------------------
con.execute(f"""
    CREATE OR REPLACE TABLE events AS
    WITH kept AS (
        SELECT * FROM raw_events
        WHERE event_id IS NOT NULL
        -- ingest_ts is always clean F1, and F1 strings sort in time order
        QUALIFY row_number() OVER (PARTITION BY event_id ORDER BY ingest_ts) = 1
    ),
    parsed AS (
        SELECT *, ts_ms(ts) AS t, ts_ms(ingest_ts) AS it,
               price_value(price_cents) AS price
        FROM kept
    )
    SELECT
        COALESCE(pm.person, p.user_id)   AS person,
        p.t                              AS t,
        p.event_type                     AS event_type,
        CASE WHEN p.event_type = 'purchase' THEN p.price END AS price,
        COALESCE(pr.category, 'unknown') AS category
    FROM parsed p
    LEFT JOIN person_map pm ON pm.user_id = p.user_id
    LEFT JOIN products pr   ON pr.product_id = p.product_id
    WHERE p.user_id IS NOT NULL
      AND p.t >= {WS} AND p.t < {WE}                      -- NULL t fails this too
      AND p.event_type IN ('view', 'cart', 'purchase')    -- exact, case-sensitive
      AND p.product_id IS NOT NULL
      AND (p.event_type <> 'purchase' OR p.price IS NOT NULL)
      AND p.it - p.t <= {DAY_MS}                          -- exactly 24 h is kept
""")
log("events kept:", con.sql("SELECT count(*) FROM events").fetchone()[0])
if DEBUG:
    for row in con.sql("""
        SELECT person, make_timestamp(t * 1000) AS ts_utc, event_type, price, category
        FROM events ORDER BY person, t
    """).fetchall():
        log("   ", row)


# ---------------------------------------------------------------------------
# Stage 4a: sessions
#
# Per person, sort events by time. A new session starts when the gap from the
# previous event is over 30 min. Events with the SAME timestamp always share a
# session, so we measure gaps between DISTINCT timestamps:
#   1. times:    the distinct (person, t) pairs
#   2. flagged:  new_session = 1 if first time of the person, or gap > 30 min
#   3. numbered: running sum of new_session = session number within the person
#   4. start of each session = smallest t with that number
# Then every event gets the start of its session by joining on (person, t).
# ---------------------------------------------------------------------------
GAP_MS = 1_800_000
con.execute(f"""
    CREATE OR REPLACE TABLE session_times AS
    WITH times AS (
        SELECT DISTINCT person, t FROM events
    ),
    flagged AS (
        SELECT person, t,
               CASE WHEN lag(t) OVER w IS NULL OR t - lag(t) OVER w > {GAP_MS}
                    THEN 1 ELSE 0 END AS new_session
        FROM times
        WINDOW w AS (PARTITION BY person ORDER BY t)
    ),
    numbered AS (
        SELECT person, t,
               sum(new_session) OVER (PARTITION BY person ORDER BY t
                                      ROWS UNBOUNDED PRECEDING) AS session_no
        FROM flagged
    )
    SELECT person, t, min(t) OVER (PARTITION BY person, session_no) AS start
    FROM numbered
""")

# Every surviving event, tagged with the start of its session.
con.execute("""
    CREATE OR REPLACE TABLE session_events AS
    SELECT e.*, st.start
    FROM events e
    JOIN session_times st ON st.person = e.person AND st.t = e.t
""")

# One row per session. A session is identified by (person, start).
# day = UTC date of the session's start, even if it runs past midnight.
con.execute(f"""
    CREATE OR REPLACE TABLE sessions AS
    SELECT person, start,
           DATE '1970-01-01' + (start // {DAY_MS})::INTEGER AS day,
           count(*)                AS n_events,
           COALESCE(sum(price), 0) AS revenue,      -- price is only set on purchases
           count(price) > 0        AS has_purchase  -- count() skips NULLs
    FROM session_events
    GROUP BY person, start
""")
log("sessions:", con.sql("SELECT count(*) FROM sessions").fetchone()[0])
if DEBUG:
    for row in con.sql("""
        SELECT person, make_timestamp(start * 1000) AS start_utc, day,
               n_events, revenue, has_purchase
        FROM sessions ORDER BY person, start
    """).fetchall():
        log("   ", row)


# ---------------------------------------------------------------------------
# Stage 4b: user_daily, one row per (person, day)
#
# top_category: among the purchases in that person's sessions of that day,
# the category with the highest summed price; ties go to the byte-wise
# smallest name (DuckDB's default string order is byte-wise). NULL with no
# purchases, which the LEFT JOIN gives us for free.
# ---------------------------------------------------------------------------
con.execute("""
    CREATE OR REPLACE TABLE user_daily AS
    WITH cat_totals AS (
        SELECT se.person, s.day, se.category, sum(se.price) AS amount
        FROM session_events se
        JOIN sessions s ON s.person = se.person AND s.start = se.start
        WHERE se.price IS NOT NULL
        GROUP BY se.person, s.day, se.category
    ),
    top_cat AS (
        SELECT person, day, category FROM cat_totals
        QUALIFY row_number() OVER (PARTITION BY person, day
                                   ORDER BY amount DESC, category ASC) = 1
    )
    SELECT s.person              AS user_id,
           s.day                 AS day,
           count(*)::BIGINT      AS sessions,
           sum(s.n_events)::BIGINT AS events,
           sum(s.revenue)::BIGINT  AS revenue_cents,
           tc.category           AS top_category
    FROM sessions s
    LEFT JOIN top_cat tc ON tc.person = s.person AND tc.day = s.day
    GROUP BY s.person, s.day, tc.category
""")


# ---------------------------------------------------------------------------
# Stage 5a: clicks
#   1. drop lines with an empty click_id        (malformed ones: stage 1)
#   2. dedup on click_id: copies are identical, keep any one
#   3. validate: user present, ts valid and in the CLICK window
#      (window_start - 7 days <= ts < window_end), campaign in campaigns.csv
# Each click gets its person, exactly like events.
# ---------------------------------------------------------------------------
WEEK_MS = 7 * DAY_MS
con.execute(f"""
    CREATE OR REPLACE TABLE clicks AS
    WITH kept AS (
        SELECT * FROM raw_clicks
        WHERE click_id IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY click_id) = 1
    ),
    parsed AS (
        SELECT *, ts_ms(ts) AS t FROM kept
    )
    SELECT COALESCE(pm.person, c.user_id) AS person,
           c.click_id, c.campaign_id, c.t
    FROM parsed c
    LEFT JOIN person_map pm ON pm.user_id = c.user_id
    WHERE c.user_id IS NOT NULL
      AND c.t >= {WS - WEEK_MS} AND c.t < {WE}
      AND c.campaign_id IN (SELECT campaign_id FROM campaigns)
""")
log("clicks kept:", con.sql("SELECT count(*) FROM clicks").fetchone()[0])


# ---------------------------------------------------------------------------
# Stage 5b: which clicks qualify for which session
#
# A click qualifies for a session when it is the same person's and
#   session_start - 7 days <= click.t <= session_start
# (a click exactly at the start counts; 1 ms after does not). The same set of
# clicks is used for both credit and assists.
#
# NOTE: this is the simple, obviously-correct version. For people with very
# many sessions and clicks it makes many pairs; we'll speed it up later.
# ---------------------------------------------------------------------------
con.execute(f"""
    CREATE OR REPLACE TABLE session_clicks AS
    SELECT s.person, s.start, c.click_id, c.campaign_id, c.t
    FROM sessions s
    JOIN clicks c
      ON c.person = s.person
     AND c.t <= s.start
     AND c.t >= s.start - {WEEK_MS}
""")

# Credit: the latest qualifying click; on a tie in t, the smallest click_id.
con.execute("""
    CREATE OR REPLACE TABLE credit AS
    SELECT person, start, campaign_id
    FROM session_clicks
    QUALIFY row_number() OVER (PARTITION BY person, start
                               ORDER BY t DESC, click_id ASC) = 1
""")

# Assists: every campaign with at least one qualifying click, once per session.
con.execute("""
    CREATE OR REPLACE TABLE assists AS
    SELECT DISTINCT person, start, campaign_id FROM session_clicks
""")


# ---------------------------------------------------------------------------
# Stage 5c: campaign_daily, one row per (campaign, day)
#
#   credited part: every session, credited to its click's campaign or to
#                  `organic` if no click qualified
#   assisted part: sessions that each campaign assisted (organic never does)
# A FULL OUTER JOIN keeps campaigns that only assisted on a day; their
# credited columns become 0.
# ---------------------------------------------------------------------------
con.execute("""
    CREATE OR REPLACE TABLE campaign_daily AS
    WITH credited AS (
        SELECT COALESCE(cr.campaign_id, 'organic') AS campaign_id,
               s.day,
               count(*)                                  AS sessions,
               count(*) FILTER (WHERE s.has_purchase)    AS purchasing_sessions,
               sum(s.revenue)                            AS revenue_cents
        FROM sessions s
        LEFT JOIN credit cr ON cr.person = s.person AND cr.start = s.start
        GROUP BY 1, 2
    ),
    assisted AS (
        SELECT a.campaign_id, s.day, count(*) AS assisted_sessions
        FROM assists a
        JOIN sessions s ON s.person = a.person AND s.start = a.start
        GROUP BY 1, 2
    )
    SELECT COALESCE(cd.campaign_id, ad.campaign_id)           AS campaign_id,
           CASE WHEN COALESCE(cd.campaign_id, ad.campaign_id) = 'organic'
                THEN 'organic' ELSE cp.channel END            AS channel,
           COALESCE(cd.day, ad.day)                           AS day,
           COALESCE(cd.sessions, 0)::BIGINT                   AS sessions,
           COALESCE(cd.purchasing_sessions, 0)::BIGINT        AS purchasing_sessions,
           COALESCE(cd.revenue_cents, 0)::BIGINT              AS revenue_cents,
           COALESCE(ad.assisted_sessions, 0)::BIGINT          AS assisted_sessions
    FROM credited cd
    FULL OUTER JOIN assisted ad
      ON ad.campaign_id = cd.campaign_id AND ad.day = cd.day
    LEFT JOIN campaigns cp
      ON cp.campaign_id = COALESCE(cd.campaign_id, ad.campaign_id)
""")

# The self-check from rules.md: both tables must agree on sessions and revenue.
u = con.sql("SELECT sum(sessions), sum(revenue_cents) FROM user_daily").fetchone()
c = con.sql("SELECT sum(sessions), sum(revenue_cents) FROM campaign_daily").fetchone()
log(f"self-check: user_daily {u}, campaign_daily {c}", "OK" if u == c else "MISMATCH")


# ---------------------------------------------------------------------------
# Write both tables as Parquet. `day` is a DATE column, as the checker needs.
# ---------------------------------------------------------------------------
for table in ["user_daily", "campaign_daily"]:
    folder = os.path.join(OUT, table)
    os.makedirs(folder, exist_ok=True)
    con.execute(f"COPY {table} TO {sql_str(os.path.join(folder, 'part-0.parquet'))} (FORMAT parquet)")
    if DEBUG:
        log(f"{table}:")
        for row in con.sql(f"SELECT * FROM {table} ORDER BY ALL").fetchall():
            log("   ", row)
log("done")
