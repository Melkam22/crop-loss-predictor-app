"""HarvestGuard Streamlit frontend.

Two views in one app, switched with the ``?page=`` query parameter so the
browser's back button works: a landing page (hero, about cards, how it works)
and the predict page, which calls the FastAPI backend deployed on Render.

Run locally from the repo root (so .streamlit/config.toml is picked up):
    streamlit run frontend/app.py
"""

import base64
from html import escape
from pathlib import Path

import requests
import streamlit as st

DEFAULT_API_URL = "https://harvestguard-api-0zuf.onrender.com"
# Render's free tier sleeps after ~15 min idle; the first request can take
# 30-60s while it wakes up.
TIMEOUT_SECONDS = 90
# Entries in the backend's crop list that are land uses, not crops.
NON_CROPS = {"GRAZING LAND", "OTHER LAND USE", "TEMPORARY GR", "OTHERS"}

st.set_page_config(page_title="HarvestGuard · Crop Loss Risk", page_icon="🌾", layout="wide")
# st.html's sanitizer strips <style> (and a style-only st.html never reaches
# the page), so the stylesheet goes in through st.markdown. Blank lines are
# removed because Markdown would end the HTML block at the first one.
_css = (Path(__file__).parent / "styles.css").read_text()
st.markdown("<style>" + "\n".join(l for l in _css.splitlines() if l.strip()) + "</style>", unsafe_allow_html=True)


def icon(name: str, cls: str = "") -> str:
    return f'<span class="msr {cls}">{name}</span>'


def api_url() -> str:
    try:
        return st.secrets["API_URL"].rstrip("/")
    except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return DEFAULT_API_URL


# ---------------------------------------------------------------- API calls
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_options() -> tuple[list[dict], list[str]]:
    base = api_url()
    regions = requests.get(f"{base}/regions", timeout=TIMEOUT_SECONDS)
    regions.raise_for_status()
    crops = requests.get(f"{base}/crops", timeout=TIMEOUT_SECONDS)
    crops.raise_for_status()
    region_list = sorted(regions.json(), key=lambda r: r["region_name"])
    crop_list = [c for c in crops.json() if c not in NON_CROPS]
    return region_list, crop_list


@st.cache_data(ttl=900, show_spinner=False)
def fetch_prediction(crop_name: str, region_code: int, household_size: int) -> dict:
    resp = requests.post(
        f"{api_url()}/predict",
        json={"crop_name": crop_name, "region_code": region_code, "household_size": household_size},
        timeout=TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()


def crop_label(name: str) -> str:
    return name.title().replace("/", " / ")


# ------------------------------------------------------------- navigation
def go(page: str | None) -> None:
    if page:
        st.query_params["page"] = page
    else:
        st.query_params.clear()
    st.session_state.scroll_top = True


def scroll_to_top() -> None:
    if st.session_state.pop("scroll_top", False):
        st.html(
            # Streamlit scrolls inside its main <section>, not the window. Run a
            # few times because stale elements from the previous page are only
            # removed once the script run finishes.
            "<script>const up=()=>{window.scrollTo(0,0);"
            "document.querySelectorAll('[data-testid=\"stMain\"]').forEach(e=>e.scrollTo(0,0));};"
            "up();[150,400,900].forEach(t=>setTimeout(up,t));</script>",
            unsafe_allow_javascript=True,
        )


BRAND = f'<div class="brand"><div class="logo">{icon("eco", "fill")}</div>HarvestGuard</div>'

TEAM = [
    # (name, role, main icon, badge icon, tone class)
    ("Ashenafi Shiferaw", "Idea & data", "lightbulb", "database", "ic-gold"),
    ("Azmain Morshed", "Tech & engineering", "terminal", "memory", "ic-sky"),
    ("Fatih Vardar", "Business & presentation", "co_present", "trending_up", "ic-leaf"),
]


def footer() -> None:
    members = "".join(
        f'<div class="member"><div class="av {tone}">{icon(main, "fill")}'
        f'<span class="sub">{icon(badge)}</span></div>'
        f'<div><div class="nm">{name}</div><div class="rl">{role}</div></div></div>'
        for name, role, main, badge, tone in TEAM
    )
    st.html(f"""
    <div class="foot" id="team">
      <div class="foot-grid">
        <div class="team">{members}</div>
        <div class="uni">{icon("school", "fill")} Built at Tomorrow University of Applied Sciences</div>
      </div>
      <div class="foot-note">
        <span>{icon("database")} Survey data: World Bank LSMS-ISA, Ethiopia 2011–2021</span>
        <span>{icon("satellite_alt")} Rainfall: CHIRPS via HDX</span>
        <span>{icon("copyright")} 2026 HarvestGuard</span>
      </div>
    </div>
    """)


# ------------------------------------------------------------ hero artwork
def hero_svg() -> str:
    stalks = []
    for i, x in enumerate([300, 330, 362, 392, 425, 455, 482]):
        top = 300 + (i % 3) * 14
        grains = []
        for j in range(5):
            y = top + j * 13
            grains.append(
                f'<ellipse cx="{x - 7}" cy="{y}" rx="7" ry="4" transform="rotate(-35 {x - 7} {y})" fill="url(#wheat)"/>'
                f'<ellipse cx="{x + 7}" cy="{y + 6}" rx="7" ry="4" transform="rotate(35 {x + 7} {y + 6})" fill="url(#wheat)"/>'
            )
        stalks.append(
            f'<g class="stalk" style="animation-delay:-{i * 0.6:.1f}s">'
            f'<line x1="{x}" y1="{top - 8}" x2="{x}" y2="440" stroke="#c99a3a" stroke-width="2.5" stroke-linecap="round"/>'
            f'<ellipse cx="{x}" cy="{top - 8}" rx="4" ry="7" fill="url(#wheat)"/>{"".join(grains)}</g>'
        )
    drops = "".join(
        f'<line class="drop" x1="{x}" y1="148" x2="{x - 3}" y2="160" stroke="#6ec8ff" stroke-width="2.5" '
        f'stroke-linecap="round" style="animation-delay:{d}s"/>'
        for x, d in [(132, 0), (152, 0.5), (172, 1.0), (192, 0.25), (212, 0.75), (162, 1.25), (142, 0.9)]
    )
    rays = "".join(
        f'<line x1="390" y1="{115 - 48}" x2="390" y2="{115 - 60}" stroke="#f2c14e" stroke-width="3" '
        f'stroke-linecap="round" transform="rotate({a} 390 115)"/>'
        for a in range(0, 360, 30)
    )
    svg = f"""
    <svg viewBox="0 0 520 470" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Fields, rain and sun">
      <style>
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        @keyframes sway {{ 0%, 100% {{ transform: rotate(-4deg); }} 50% {{ transform: rotate(4deg); }} }}
        @keyframes drift {{ 0%, 100% {{ transform: translateX(0); }} 50% {{ transform: translateX(18px); }} }}
        @keyframes rain {{ 0% {{ transform: translateY(-6px); opacity: 0; }} 20% {{ opacity: 1; }} 100% {{ transform: translateY(70px); opacity: 0; }} }}
        .sun-rays {{ transform-box: fill-box; transform-origin: center; animation: spin 30s linear infinite; }}
        .cloud {{ animation: drift 7s ease-in-out infinite; }}
        .drop {{ animation: rain 1.6s linear infinite; }}
        .stalk {{ transform-box: fill-box; transform-origin: 50% 100%; animation: sway 4s ease-in-out infinite; }}
        .frame {{ fill: rgba(17, 25, 19, .65); stroke: rgba(236, 242, 236, 0.16); }}
      </style>
      <defs>
        <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#16251b"/><stop offset="1" stop-color="#0e1610"/>
        </linearGradient>
        <linearGradient id="hill1" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#2f7a46"/><stop offset="1" stop-color="#173b23"/>
        </linearGradient>
        <linearGradient id="hill2" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#4fae66"/><stop offset="1" stop-color="#1f4d2c"/>
        </linearGradient>
        <linearGradient id="wheat" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#ffd978"/><stop offset="1" stop-color="#d79b2b"/>
        </linearGradient>
        <radialGradient id="sunglow"><stop offset="0" stop-color="#f2c14e" stop-opacity=".55"/><stop offset="1" stop-color="#f2c14e" stop-opacity="0"/></radialGradient>
        <clipPath id="frameclip"><rect x="20" y="20" width="480" height="430" rx="30"/></clipPath>
      </defs>
      <rect class="frame" x="20" y="20" width="480" height="430" rx="30"/>
      <g clip-path="url(#frameclip)">
        <rect x="20" y="20" width="480" height="430" fill="url(#sky)" opacity=".9"/>
        <circle cx="390" cy="115" r="95" fill="url(#sunglow)"/>
        <g class="sun-rays">{rays}</g>
        <circle cx="390" cy="115" r="34" fill="#f2c14e"/>
        <g class="cloud">
          <ellipse cx="170" cy="120" rx="58" ry="24" fill="#d9e6f2"/>
          <circle cx="148" cy="106" r="24" fill="#e8f0f8"/>
          <circle cx="182" cy="96" r="30" fill="#f2f7fb"/>
          {drops}
        </g>
        <path d="M20 330 Q 150 250 280 315 T 520 290 V 470 H20 Z" fill="url(#hill1)"/>
        <g transform="translate(70 262)">
          <rect x="0" y="40" width="74" height="68" rx="6" fill="#24332a" stroke="#3b5244"/>
          <path d="M-8 44 L37 6 L82 44 Z" fill="#c99a3a"/>
          <rect x="26" y="70" width="22" height="38" rx="3" fill="#0e1610"/>
          <circle cx="37" cy="28" r="6" fill="#0e1610"/>
        </g>
        <path d="M20 400 Q 170 345 320 395 T 520 380 V 470 H20 Z" fill="url(#hill2)"/>
        {"".join(stalks)}
        <path d="M20 440 H500" stroke="#0e1610" stroke-width="20"/>
      </g>
    </svg>
    """
    # st.html's sanitizer strips inline <svg>, so ship it as an image.
    data = base64.b64encode(svg.encode()).decode()
    return f'<img class="hero-svg" src="data:image/svg+xml;base64,{data}" alt="Fields, rain and sun">'


# ------------------------------------------------------------ landing page
def landing() -> None:
    with st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute"):
        st.html(BRAND, width="content")
        with st.container(horizontal=True, vertical_alignment="center", width="content", gap="medium"):
            st.html(
                f'<nav class="nav-links"><a class="hide-sm" href="#about">{icon("info")} About</a>'
                f'<a class="hide-sm" href="#how">{icon("route")} How it works</a>'
                f'<a class="hide-sm" href="#team">{icon("groups")} Team</a></nav>',
                width="content",
            )
            st.button("Predict", key="nav_predict", icon=":material/bolt:", on_click=go, args=("predict",))

    with st.container(key="hero"):
        left, right = st.columns([1.05, 1], gap="large", vertical_alignment="center")
        with left:
            st.html(f"""
            <div class="reveal">
              <span class="eyebrow"><span class="dot"></span>Early-warning ML for Ethiopian smallholders</span>
              <h1 class="hero-title">Know the risk <span class="grad-text">before</span> the harvest is lost.</h1>
              <p class="hero-sub">HarvestGuard tells you whether a crop is at <b>High</b> or <b>Low</b> risk of
              post-harvest loss, using the region, the crop and this season's satellite rainfall.
              Built for NGO officers and ministry staff who need to act early.</p>
            </div>
            """)
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.button(
                    "Predict crop loss risk",
                    key="hero_predict",
                    icon=":material/arrow_forward:",
                    on_click=go,
                    args=("predict",),
                )
                st.html(f'<a class="ghost-btn" href="#about">{icon("play_circle")} How it works</a>', width="content")
            st.html(f"""
            <div class="trust reveal d3">
              <span>{icon("verified", "fill")} 5 national surveys</span>
              <span>{icon("satellite_alt")} Live CHIRPS rainfall</span>
              <span>{icon("lock")} No personal data stored</span>
            </div>
            """)
        with right:
            st.html(f"""
            <div class="hero-art reveal d2">
              <div class="blob b1"></div><div class="blob b2"></div>
              {hero_svg()}
              <div class="chip-float chip-a"><div class="ic ic-sky">{icon("water_drop", "fill")}</div>
                <div>Meher rainfall<b>95% of normal</b></div></div>
              <div class="chip-float chip-b"><div class="ic ic-leaf">{icon("verified_user", "fill")}</div>
                <div>Loss risk<b>Low</b></div></div>
              <div class="chip-float chip-c"><div class="ic ic-gold">{icon("grass", "fill")}</div>
                <div>Crop · Region<b>Teff · Amhara</b></div></div>
            </div>
            """)

    st.html(f"""
    <div class="stats reveal d3">
      <div class="stat"><div class="ic ic-gold">{icon("table_rows", "fill")}</div>
        <div><div class="num"><span class="counter" style="--n:70"></span>k+</div><div class="lbl">household-crop records</div></div></div>
      <div class="stat"><div class="ic ic-leaf">{icon("event_repeat", "fill")}</div>
        <div><div class="num counter" style="--n:5"></div><div class="lbl">survey waves, 2011–2021</div></div></div>
      <div class="stat"><div class="ic ic-sky">{icon("psychiatry", "fill")}</div>
        <div><div class="num counter" style="--n:96"></div><div class="lbl">crops you can check</div></div></div>
      <div class="stat"><div class="ic ic-danger">{icon("map", "fill")}</div>
        <div><div class="num counter" style="--n:10"></div><div class="lbl">regions of Ethiopia</div></div></div>
    </div>
    """)

    cards = [
        ("ic-danger", "crisis_alert", "The problem",
         "Ethiopian smallholders lose 15–32% of their crops after harvest to moisture, insects, poor storage "
         "and slow transport. The warning signs are usually missed until it is too late.",
         "trending_down", "15–32% lost"),
        ("ic-gold", "flag", "Our purpose",
         "Give the people who can intervene, NGO officers and ministry staff, an early signal of where losses "
         "are most likely, so resources reach farmers before the season peaks.",
         "target", "SDG 2 · Zero Hunger"),
        ("ic-leaf", "construction", "What we did",
         "Cleaned and joined five World Bank household surveys, each in a different format, into one dataset. "
         "Fixed corrupted IDs, merged crop spellings and removed every source of data leakage.",
         "build", "70,563 records"),
        ("ic-sky", "satellite_alt", "The data",
         "LSMS-ISA farm surveys tell us where losses happened. CHIRPS satellite rainfall adds the Belg "
         "(Feb–May) and Meher (Jun–Sep) seasons for each region.",
         "database", "LSMS + CHIRPS"),
        ("ic-gold", "model_training", "The model",
         "A tuned XGBoost model, tested head-to-head against a Random Forest with household-grouped "
         "cross-validation. On unseen households it catches about 7 in 10 real losses.",
         "insights", "ROC-AUC 0.79"),
        ("ic-leaf", "verified_user", "Used responsibly",
         "A screening tool, not a forecast of tonnes lost. Results are shown as High or Low, never fake "
         "percentages, and thin region-crop combinations are clearly flagged.",
         "shield", "Transparent"),
    ]
    card_html = "".join(
        f'<article class="card reveal d{i + 1}"><div class="ic {tone}">{icon(ic, "fill")}</div>'
        f"<h3>{title}</h3><p>{body}</p>"
        f'<span class="tag">{icon(tag_ic)} {tag}</span></article>'
        for i, (tone, ic, title, body, tag_ic, tag) in enumerate(cards)
    )
    st.html(f"""
    <section class="section" id="about">
      <span class="kicker">{icon("auto_awesome")} About the project</span>
      <h2>Turning survey data into an early warning</h2>
      <p class="lead">Post-harvest loss is preventable when you see it coming. HarvestGuard learns from
      a decade of Ethiopian farm surveys and rainfall records to point out where the risk is highest.</p>
      <div class="cards">{card_html}</div>
    </section>
    """)

    steps = [
        ("location_on", "Pick region & crop", "Choose from Ethiopia's 10 farming regions and 96 crops, then add household size."),
        ("cloud_sync", "Rainfall is added for you", "The server pulls the latest complete Belg and Meher seasons from CHIRPS satellite data."),
        ("task_alt", "Get a clear answer", "See High or Low risk, the rainfall behind it, and practical steps to protect the harvest."),
    ]
    step_html = "".join(
        f'<div class="step reveal d{i + 1}"><span class="n">STEP 0{i + 1}</span>'
        f'{icon(ic, "ic")}<h4>{title}</h4><p>{body}</p>'
        + (f'<span class="arrow">{icon("arrow_forward")}</span>' if i < 2 else "")
        + "</div>"
        for i, (ic, title, body) in enumerate(steps)
    )
    sdgs = [
        ("#DDA63A", "2", "Zero Hunger"),
        ("#FD6925", "9", "Industry & Innovation"),
        ("#BF8B2E", "12", "Responsible Consumption"),
        ("#3F7E44", "13", "Climate Action"),
    ]
    sdg_html = "".join(
        f'<span class="sdg"><span class="badge" style="background:{c}">{n}</span>{label}</span>' for c, n, label in sdgs
    )
    st.html(f"""
    <section class="section" id="how">
      <span class="kicker">{icon("route")} How it works</span>
      <h2>Three inputs. One clear answer.</h2>
      <p class="lead">Nobody has to know rainfall in millimetres. You choose what you know, and the
      backend fills in the rest before the model runs.</p>
      <div class="steps">{step_html}</div>
      <div class="sdgs">{sdg_html}</div>
    </section>
    """)

    with st.container(key="cta"):
        text, btn = st.columns([2, 1], vertical_alignment="center")
        with text:
            st.html(
                '<p class="cta-title">Ready to check a crop?</p>'
                '<p class="cta-sub">It takes three inputs and a few seconds.</p>'
            )
        with btn:
            with st.container(horizontal=True, horizontal_alignment="right"):
                st.button("Start a prediction", key="cta_predict", icon=":material/arrow_forward:", on_click=go, args=("predict",))

    footer()


# ------------------------------------------------------------ predict page
def rain_card(name: str, months: str, mm: float | None, pct: float | None) -> str:
    if pct is None:
        return (
            f'<div class="rain"><div class="rain-head"><span class="rain-name">{icon("water_drop", "fill")} {name}</span></div>'
            f'<div class="rain-pct">n/a</div><div class="bar-legend"><span>{months}</span></div></div>'
        )
    if pct < 85:
        status, cls, color = "Below normal", "s-low", "var(--gold)"
    elif pct <= 115:
        status, cls, color = "Near normal", "s-ok", "var(--leaf)"
    else:
        status, cls, color = "Above normal", "s-high", "var(--sky)"
    width = max(2, min(pct, 200) / 2)
    return f"""
    <div class="rain">
      <div class="rain-head"><span class="rain-name">{icon("water_drop", "fill")} {name}</span>
        <span class="rain-status {cls}">{status}</span></div>
      <div class="rain-pct">{pct:.0f}% <small>of normal · {mm:.0f} mm</small></div>
      <div class="bar"><div class="fill" style="width:{width:.1f}%;background:{color}"></div><span class="mark"></span></div>
      <div class="bar-legend"><span>{months}</span><span>normal</span><span>2×</span></div>
    </div>"""


TIPS_HIGH = [
    ("sunny", "Dry before storing", "Get grain down to a safe moisture level (about 12–13%) before it goes into store."),
    ("inventory_2", "Use airtight storage", "Hermetic bags or metal silos cut insect and mould damage without chemicals."),
    ("search_insights", "Inspect often", "Check stores every week or two for insects, mould, heat or a bad smell."),
    ("local_shipping", "Move it sooner", "Plan transport and sale early so the harvest doesn't sit in the field."),
]
TIPS_LOW = [
    ("cleaning_services", "Clean the store", "Clean and repair stores, and keep sacks off the floor on pallets."),
    ("sunny", "Still dry it well", "Damp grain is the most common cause of loss, even in a good year."),
    ("search_insights", "Spot-check monthly", "A quick look for insects or mould catches problems while they're small."),
    ("rainy", "Watch the rains", "Late or heavy rain at harvest time can change the picture. Check again next season."),
]


def result_html(res: dict, crop: str, region: str, household_size: int) -> str:
    high = res["risk"].lower() == "high"
    tone = "high" if high else "low"
    badge = "warning" if high else "verified_user"
    if high:
        text = (
            f"<b>{crop}</b> in <b>{region}</b>, with this season's rainfall, looks like the situations where "
            "farmers in past national surveys reported post-harvest losses. Treat this as a prompt to check "
            "storage and plan support early. The alert is tuned to catch most real losses, so some alerts "
            "will turn out to be false alarms."
        )
    else:
        text = (
            f"<b>{crop}</b> in <b>{region}</b>, with this season's rainfall, doesn't resemble the situations "
            "where farmers in past surveys reported losses. Low risk is not no risk: good drying and storage "
            "still matter."
        )
    rain = res.get("rainfall_used", {})
    callouts = ""
    if res.get("limited_data"):
        callouts += (
            f'<div class="callout">{icon("data_alert")}<div><b>Limited data.</b> Fewer than 30 past survey '
            "records exist for this crop in this region, so treat this result with extra caution.</div></div>"
        )
    for w in res.get("warnings", []):
        callouts += f'<div class="callout info">{icon("info")}<div>{escape(w)}</div></div>'
    tips = TIPS_HIGH if high else TIPS_LOW
    tips_html = "".join(
        f'<div class="tip" style="animation-delay:{0.35 + i * 0.08:.2f}s">{icon(ic, "fill")}<div><b>{t}</b>{d}</div></div>'
        for i, (ic, t, d) in enumerate(tips)
    )
    return f"""
    <div class="result {tone}">
      <div class="res-top">
        <div class="res-badge"><span class="ring"></span><span class="ring"></span>
          <div class="core">{icon(badge, "fill")}</div></div>
        <div>
          <div class="res-label">Post-harvest loss risk</div>
          <div class="res-risk">{"High" if high else "Low"} risk</div>
          <div class="res-ctx">
            <span class="pill">{icon("grass")} {crop}</span>
            <span class="pill">{icon("location_on")} {region}</span>
            <span class="pill">{icon("family_restroom")} {household_size} people</span>
          </div>
        </div>
      </div>
      <p class="res-text">{text}</p>
      {callouts}
      <div class="sub-h">{icon("rainy")} Rainfall the model used <small>latest complete seasons · CHIRPS</small></div>
      <div class="rain-grid">
        {rain_card("Belg season", "Feb–May", rain.get("rainfall_belg_mm"), rain.get("rainfall_belg_pct_of_avg"))}
        {rain_card("Meher season", "Jun–Sep", rain.get("rainfall_meher_mm"), rain.get("rainfall_meher_pct_of_avg"))}
      </div>
      <div class="sub-h">{icon("tips_and_updates")} {"What you can do now" if high else "Keep the harvest safe"}
        <small>general good practice</small></div>
      <div class="tips">{tips_html}</div>
      <div class="res-foot">{icon("info")} XGBoost model trained on World Bank LSMS-ISA surveys (2011–2021).
        Its score isn't a probability, so we only show High or Low.</div>
    </div>
    """


def predict_page() -> None:
    scroll_to_top()
    with st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute"):
        st.html(BRAND, width="content")
        st.button("Back to home", key="back", icon=":material/arrow_back:", on_click=go, args=(None,))

    st.html(f"""
    <div class="page-head reveal">
      <span class="kicker">{icon("bolt", "fill")} Risk check</span>
      <h1>How likely is post-harvest loss?</h1>
      <p>Choose a region and crop and add household size. We add the latest seasonal rainfall
      automatically and return a High or Low risk rating.</p>
    </div>
    """)

    try:
        with st.spinner("Connecting to the prediction server. The first visit can take up to a minute while it wakes up…"):
            regions, crops = fetch_options()
    except requests.RequestException:
        st.html(f"""
        <div class="err">{icon("cloud_off")}<h3>Can't reach the prediction server</h3>
        <p>It may still be waking up. Give it a moment and try again.</p></div>
        """)
        if st.button("Try again", icon=":material/refresh:"):
            fetch_options.clear()
            st.rerun()
        footer()
        return

    region_names = {r["region_code"]: r["region_name"] for r in regions}
    left, right = st.columns([5, 7], gap="large")

    with left:
        with st.container(key="form_card"):
            st.html(
                f'<div class="form-title"><div class="ic ic-gold">{icon("tune", "fill")}</div>Your details</div>'
            )
            with st.form("predict_form", border=False):
                region_code = st.selectbox(
                    ":material/location_on: Region",
                    options=list(region_names),
                    format_func=region_names.get,
                    index=list(region_names).index(3) if 3 in region_names else 0,  # Amhara
                )
                crop = st.selectbox(
                    ":material/grass: Crop",
                    options=crops,
                    format_func=crop_label,
                    index=crops.index("MAIZE") if "MAIZE" in crops else 0,
                    help="Type to search the list.",
                )
                household_size = st.number_input(
                    ":material/family_restroom: Household size", min_value=1, max_value=40, value=5, step=1,
                    help="Number of people living in the household.",
                )
                submitted = st.form_submit_button("Assess risk", icon=":material/insights:", width="stretch")
            st.html(
                f'<div class="auto-note">{icon("satellite_alt")}<div><b>Rainfall is automatic.</b> '
                "We use satellite estimates for the most recent complete Belg and Meher seasons in your region.</div></div>"
            )

    if submitted:
        try:
            with st.spinner("Checking rainfall and running the model…"):
                res = fetch_prediction(crop, int(region_code), int(household_size))
            st.session_state.result = (res, crop, int(region_code), int(household_size))
        except requests.RequestException:
            st.session_state.result = "error"

    with right:
        result = st.session_state.get("result")
        if result == "error":
            st.html(f"""
            <div class="err">{icon("cloud_off")}<h3>Something went wrong</h3>
            <p>The prediction server didn't respond in time. Please press Assess risk again.</p></div>
            """)
        elif result:
            res, r_crop, r_region, r_size = result
            st.html(result_html(res, crop_label(r_crop), region_names.get(r_region, str(r_region)), r_size))
        else:
            st.html(f"""
            <div class="empty">
              <div class="radar"><span class="ring"></span><span class="ring"></span><span class="ring"></span>
                <div class="core">{icon("radar")}</div></div>
              <h3>Your result will appear here</h3>
              <p>Fill in the details on the left and press <b>Assess risk</b>.</p>
            </div>
            """)

    footer()


if st.query_params.get("page") == "predict":
    predict_page()
else:
    scroll_to_top()
    landing()
