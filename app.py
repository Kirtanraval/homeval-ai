import streamlit as st
import pickle
import pandas as pd
import json
import numpy as np
import time

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HomeVal AI · Property Valuation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# Zipcode → lat/long (King County, WA)
# ─────────────────────────────────────────────────────────────
ZIPCODE_COORDS = {
    98001:(47.3073,-122.2568),98002:(47.3040,-122.2148),98003:(47.3229,-122.3129),
    98004:(47.6195,-122.2013),98005:(47.6101,-122.1710),98006:(47.5524,-122.1671),
    98007:(47.6068,-122.1421),98008:(47.5996,-122.1188),98010:(47.3293,-122.0026),
    98011:(47.7601,-122.2048),98014:(47.7354,-121.9076),98019:(47.7368,-121.9718),
    98022:(47.2002,-121.9857),98023:(47.3126,-122.3636),98024:(47.5718,-121.9873),
    98027:(47.5309,-122.0429),98028:(47.7540,-122.2430),98029:(47.5687,-122.0193),
    98030:(47.3726,-122.1771),98031:(47.3876,-122.1916),98032:(47.4076,-122.2268),
    98033:(47.6793,-122.1851),98034:(47.7143,-122.2143),98038:(47.3768,-122.0349),
    98039:(47.6318,-122.2346),98040:(47.5529,-122.2268),98042:(47.3718,-122.1254),
    98045:(47.4907,-121.7715),98052:(47.6699,-122.1185),98053:(47.6554,-122.0474),
    98055:(47.4654,-122.2157),98056:(47.4971,-122.1857),98058:(47.4382,-122.1476),
    98059:(47.4888,-122.1254),98065:(47.5288,-121.8638),98072:(47.7551,-122.1468),
    98074:(47.6274,-122.0424),98075:(47.5974,-122.0424),98077:(47.7496,-122.0815),
    98092:(47.3193,-122.1476),98102:(47.6376,-122.3224),98103:(47.6760,-122.3424),
    98105:(47.6618,-122.2924),98106:(47.5476,-122.3624),98107:(47.6696,-122.3724),
    98108:(47.5376,-122.3024),98109:(47.6376,-122.3524),98112:(47.6318,-122.2924),
    98115:(47.6818,-122.2924),98116:(47.5718,-122.3924),98117:(47.6918,-122.3724),
    98118:(47.5518,-122.2824),98119:(47.6476,-122.3624),98122:(47.6076,-122.3024),
    98125:(47.7176,-122.3024),98126:(47.5576,-122.3724),98133:(47.7276,-122.3424),
    98136:(47.5376,-122.3824),98144:(47.5918,-122.2924),98146:(47.4976,-122.3524),
    98155:(47.7476,-122.3024),98166:(47.4476,-122.3524),98168:(47.4876,-122.3124),
    98177:(47.7376,-122.3924),98178:(47.4976,-122.2524),98188:(47.4476,-122.2724),
    98198:(47.3976,-122.3124),98199:(47.6576,-122.3924),
}
DEFAULT_LAT, DEFAULT_LONG = 47.5101, -122.2315

def zip_to_coords(z): return ZIPCODE_COORDS.get(int(z),(DEFAULT_LAT,DEFAULT_LONG))

def similar_homes_range(price,grade,sqft):
    spread = 0.08+(abs(grade-7)*0.005)+(sqft/500_000)
    return price*(1-spread), price*(1+spread)

FEATURES = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront',
            'view','condition','grade','sqft_above','sqft_basement','yr_built',
            'yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15','total_area']

LABEL_MAP = {
    'sqft_living':'Living Area','grade':'Quality Grade','lat':'Latitude',
    'sqft_above':'Above-Ground SF','long':'Longitude','zipcode':'Zipcode',
    'sqft_living15':'Neighbor Avg SF','total_area':'Total Area','yr_built':'Year Built',
    'bathrooms':'Bathrooms','sqft_lot':'Lot Size','sqft_lot15':'Neighbor Lot',
    'bedrooms':'Bedrooms','condition':'Condition','yr_renovated':'Yr Renovated',
    'floors':'Floors','sqft_basement':'Basement SF','view':'View Score','waterfront':'Waterfront',
}

# ─────────────────────────────────────────────────────────────
# CSS — Notion/Vercel clean light + animations
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}

:root{
  --white:     #FFFFFF;
  --bg:        #FAFAFA;
  --bg-2:      #F4F4F5;
  --border:    #E4E4E7;
  --border-2:  #D4D4D8;
  --ink:       #09090B;
  --ink-2:     #52525B;
  --ink-3:     #A1A1AA;
  --ink-4:     #D4D4D8;
  --blue:      #2563EB;
  --blue-2:    #DBEAFE;
  --blue-3:    #EFF6FF;
  --emerald:   #059669;
  --emerald-2: #D1FAE5;
  --amber:     #D97706;
  --amber-2:   #FEF3C7;
  --red:       #DC2626;
  --red-2:     #FEE2E2;
  --radius-sm: 8px;
  --radius:    12px;
  --radius-lg: 16px;
  --shadow-xs: 0 1px 2px rgba(0,0,0,.05);
  --shadow-sm: 0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.04);
  --shadow:    0 4px 6px -1px rgba(0,0,0,.07),0 2px 4px -1px rgba(0,0,0,.04);
  --shadow-md: 0 10px 15px -3px rgba(0,0,0,.07),0 4px 6px -2px rgba(0,0,0,.04);
}

html,body,.stApp{
  background:var(--bg) !important;
  font-family:'Plus Jakarta Sans',sans-serif !important;
  color:var(--ink) !important;
}

/* ── Nuke ALL Streamlit chrome & spacing ── */
#MainMenu,footer,header,.stDeployButton{display:none !important;}
section[data-testid="stSidebar"]{display:none !important;}

/* The gray gap is the stHeader element — hide it completely */
[data-testid="stHeader"]{
  display:none !important;
  height:0 !important;
  min-height:0 !important;
  padding:0 !important;
  margin:0 !important;
}

/* AppView container must start at 0 */
[data-testid="stAppViewContainer"]{
  padding-top:0 !important;
  margin-top:0 !important;
}

/* The inner vertical block that holds everything */
[data-testid="stVerticalBlock"]{
  gap:0 !important;
}

/* Main content block padding */
.block-container{
  padding:0 !important;
  padding-top:0 !important;
  margin-top:0 !important;
  max-width:100% !important;
}
.main .block-container{padding-top:0 !important;}
.main > div:first-child{padding-top:0 !important;}
.stApp > div{padding-top:0 !important;}

/* Any auto-generated spacer divs Streamlit injects */
div[data-testid="stSpacer"]{display:none !important;}
div[class*="css-"][class*="e1fqkh3"]{padding-top:0 !important;}

/* ── Staggered fade-in keyframe ── */
@keyframes fadeUp{
  from{opacity:0;transform:translateY(16px);}
  to  {opacity:1;transform:translateY(0);}
}
.fade-1{animation:fadeUp .45s ease both;}
.fade-2{animation:fadeUp .45s .08s ease both;}
.fade-3{animation:fadeUp .45s .16s ease both;}
.fade-4{animation:fadeUp .45s .24s ease both;}
.fade-5{animation:fadeUp .45s .32s ease both;}
.fade-6{animation:fadeUp .45s .40s ease both;}

/* ── Shimmer keyframe for price ── */
@keyframes shimmer{
  0%  {background-position:-400px 0;}
  100%{background-position:400px 0;}
}

/* ── Topbar ── */
.topbar{
  display:flex;align-items:center;justify-content:space-between;
  padding:.875rem 2.5rem;
  background:rgba(255,255,255,.85);
  backdrop-filter:blur(12px);
  border-bottom:1px solid var(--border);
  position:sticky;top:0;z-index:999;
}
.topbar-logo{
  display:flex;align-items:center;gap:10px;
  font-weight:800;font-size:1rem;letter-spacing:-.4px;color:var(--ink);
}
.logo-icon{
  width:30px;height:30px;border-radius:8px;
  background:var(--blue);display:flex;align-items:center;
  justify-content:center;font-size:14px;
}
.topbar-right{display:flex;align-items:center;gap:.75rem;}
.pill{
  font-size:11px;font-weight:600;padding:4px 12px;border-radius:99px;
  letter-spacing:.3px;
}
.pill-blue{background:var(--blue-2);color:var(--blue);}
.pill-green{background:var(--emerald-2);color:var(--emerald);}

/* ── Page wrapper ── */
.page{padding:1.5rem 2rem 4rem;max-width:1400px;margin:0 auto;}
/* remove old stat grid styles */
.stat-grid,.stat-card,.stat-card-label,.stat-card-value,.stat-card-sub,.stat-badge,.badge-up,.badge-neutral{display:revert;}

/* ── Hero Banner ── */
.hero-banner{
  background:var(--white);
  border-bottom:1px solid var(--border);
  padding:1.5rem 2rem 1.5rem;
  text-align:center;
  position:relative;
  overflow:hidden;
  margin-bottom:0;
}
.hero-banner::before{
  content:'';position:absolute;inset:0;
  background:
    radial-gradient(ellipse 60% 80% at 50% -10%, rgba(37,99,235,.06) 0%, transparent 70%);
  pointer-events:none;
}
/* subtle dot-grid texture */
.hero-banner::after{
  content:'';position:absolute;inset:0;
  background-image:radial-gradient(circle, #E4E4E7 1px, transparent 1px);
  background-size:24px 24px;
  opacity:.45;
  pointer-events:none;
  mask-image:radial-gradient(ellipse 80% 100% at 50% 0%, black 0%, transparent 70%);
}
.hero-inner{position:relative;z-index:1;max-width:640px;margin:0 auto;}
.hero-eyebrow{
  display:inline-flex;align-items:center;gap:8px;
  font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
  color:var(--blue);margin-bottom:1rem;
}
.hero-dot{
  width:6px;height:6px;border-radius:50%;background:var(--blue);
  animation:pulse 2s infinite;
}
.hero-title{
  font-size:clamp(2rem,4vw,3rem);font-weight:800;
  letter-spacing:-1.5px;line-height:1.1;color:var(--ink);
  margin-bottom:.75rem;
}
.hero-title span{
  background:linear-gradient(135deg,#2563EB 0%,#0EA5E9 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;
}
.hero-sub{
  font-size:.95rem;color:var(--ink-2);line-height:1.7;
  margin-bottom:1.5rem;
}
.hero-chips{
  display:flex;flex-wrap:wrap;gap:.5rem;justify-content:center;
}
.hero-chip{
  font-size:12px;font-weight:600;
  padding:5px 14px;border-radius:99px;
  background:var(--bg);border:1px solid var(--border);
  color:var(--ink-2);
  transition:all .15s;
}
.hero-chip:hover{border-color:var(--blue);color:var(--blue);background:var(--blue-3);}

/* ── Page wrapper ── */

/* ── Stat row ── */
.stat-grid{
  display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;
  margin-bottom:1.5rem;
}
.stat-card{
  background:var(--white);border:1px solid var(--border);
  border-radius:var(--radius);padding:1.1rem 1.25rem;
  box-shadow:var(--shadow-xs);
  transition:box-shadow .2s,transform .2s;
}
.stat-card:hover{box-shadow:var(--shadow-sm);transform:translateY(-1px);}
.stat-card-label{
  font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;
  color:var(--ink-3);margin-bottom:.4rem;display:flex;align-items:center;gap:6px;
}
.stat-card-value{
  font-family:'DM Mono',monospace;font-size:1.5rem;font-weight:500;
  color:var(--ink);letter-spacing:-.5px;
}
.stat-card-sub{font-size:11px;color:var(--ink-3);margin-top:.25rem;}
.stat-badge{
  display:inline-flex;align-items:center;gap:4px;
  font-size:10px;font-weight:600;padding:2px 7px;border-radius:99px;
}
.badge-up{background:var(--emerald-2);color:var(--emerald);}
.badge-neutral{background:var(--blue-2);color:var(--blue);}

/* ── Main dashboard grid ── */
.dash-grid{
  display:grid;
  grid-template-columns:1fr 1fr;
  grid-template-rows:auto auto;
  gap:1rem;
}

/* ── Cards ── */
.card{
  background:var(--white);border:1px solid var(--border);
  border-radius:var(--radius-lg);box-shadow:var(--shadow-xs);
  overflow:hidden;
}
.card-header{
  padding:1.25rem 1.5rem .75rem;
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
}
.card-title{font-size:.8rem;font-weight:700;letter-spacing:.8px;text-transform:uppercase;color:var(--ink-3);}
.card-body{padding:1.25rem 1.5rem 1.5rem;}

/* ── Inputs ── */
.input-grid{display:grid;grid-template-columns:1fr 1fr;gap:.875rem;margin-bottom:.875rem;}
.input-full{grid-column:1/-1;}

div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label{
  font-size:11px !important;font-weight:700 !important;
  letter-spacing:.8px !important;text-transform:uppercase !important;
  color:var(--ink-2) !important;margin-bottom:2px !important;
}
div[data-testid="stNumberInput"] input{
  background:var(--bg) !important;
  border:1.5px solid var(--border) !important;
  border-radius:var(--radius-sm) !important;
  color:var(--ink) !important;
  font-family:'Plus Jakarta Sans',sans-serif !important;
  font-weight:500 !important;font-size:.9rem !important;
  transition:border .15s,box-shadow .15s !important;
}
div[data-testid="stNumberInput"] input:focus{
  border-color:var(--blue) !important;
  box-shadow:0 0 0 3px rgba(37,99,235,.1) !important;
  outline:none !important;background:var(--white) !important;
}

/* Grade slider track */
div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]{
  background:var(--blue) !important;border-color:var(--blue) !important;
}

/* ── CTA Button ── */
div.stButton>button{
  width:100% !important;
  background:var(--blue) !important;
  color:#fff !important;
  font-family:'Plus Jakarta Sans',sans-serif !important;
  font-weight:700 !important;font-size:.85rem !important;
  border:none !important;border-radius:var(--radius-sm) !important;
  padding:.75rem 1.5rem !important;
  cursor:pointer !important;
  transition:all .18s !important;
  box-shadow:0 1px 3px rgba(37,99,235,.3),0 4px 12px rgba(37,99,235,.2) !important;
  letter-spacing:.2px !important;
  margin-top:.5rem !important;
}
div.stButton>button:hover{
  background:#1D4ED8 !important;
  box-shadow:0 1px 3px rgba(37,99,235,.35),0 6px 18px rgba(37,99,235,.28) !important;
  transform:translateY(-1px) !important;
}
div.stButton>button:active{transform:translateY(0) !important;}

/* ── Result panel ── */
.result-panel{
  background:var(--white);border:1px solid var(--border);
  border-radius:var(--radius-lg);box-shadow:var(--shadow-xs);
  overflow:hidden;display:flex;flex-direction:column;
}

/* Price hero */
.price-hero{
  padding:2rem 1.75rem 1.5rem;
  background:linear-gradient(135deg,#0F172A 0%,#1E3A5F 100%);
  position:relative;overflow:hidden;flex-shrink:0;
}
.price-hero::before{
  content:'';position:absolute;
  top:-60px;right:-60px;
  width:220px;height:220px;border-radius:50%;
  background:radial-gradient(circle,rgba(37,99,235,.4) 0%,transparent 65%);
}
.price-hero::after{
  content:'';position:absolute;
  bottom:-50px;left:30px;
  width:160px;height:160px;border-radius:50%;
  background:radial-gradient(circle,rgba(16,185,129,.25) 0%,transparent 65%);
}
.price-eyebrow{
  font-size:10px;font-weight:600;letter-spacing:2px;text-transform:uppercase;
  color:rgba(255,255,255,.45);margin-bottom:.5rem;position:relative;z-index:1;
}
.price-value{
  font-family:'DM Mono',monospace;
  font-size:clamp(2rem,4vw,3.2rem);font-weight:500;
  color:#fff;letter-spacing:-1.5px;line-height:1;
  position:relative;z-index:1;
}
.price-value.idle{
  font-family:'Plus Jakarta Sans',sans-serif;
  font-size:1rem;font-weight:400;
  color:rgba(255,255,255,.3);letter-spacing:0;
}
.price-confidence{
  font-size:11px;color:rgba(255,255,255,.35);
  margin-top:.4rem;position:relative;z-index:1;
}
.price-badge{
  display:inline-flex;align-items:center;gap:5px;
  font-size:11px;font-weight:600;padding:5px 12px;border-radius:99px;
  margin-top:.9rem;position:relative;z-index:1;
}
.badge-lux   {background:rgba(251,191,36,.15);color:#FCD34D;}
.badge-mid   {background:rgba(37,99,235,.25);color:#93C5FD;}
.badge-budget{background:rgba(16,185,129,.2);color:#6EE7B7;}

/* ── Price meter ── */
.meter-wrap{padding:1.25rem 1.75rem;border-bottom:1px solid var(--border);}
.meter-label{
  font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;
  color:var(--ink-3);margin-bottom:.6rem;
  display:flex;justify-content:space-between;
}
.meter-track{
  height:6px;background:var(--bg-2);border-radius:99px;
  overflow:hidden;border:1px solid var(--border);
}
.meter-fill{
  height:100%;border-radius:99px;
  transition:width .8s cubic-bezier(.34,1.56,.64,1);
}
.meter-ticks{display:flex;justify-content:space-between;margin-top:.3rem;}
.meter-tick{font-family:'DM Mono',monospace;font-size:9px;color:var(--ink-4);}

/* ── Similar homes ── */
.similar-wrap{padding:1.1rem 1.75rem;border-bottom:1px solid var(--border);}
.similar-header{
  font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;
  color:var(--ink-3);margin-bottom:.5rem;
}
.similar-range{
  font-family:'DM Mono',monospace;font-size:1.05rem;font-weight:500;
  color:var(--ink);
}
.similar-sub{font-size:11px;color:var(--ink-3);margin-top:.2rem;}
.range-bar{
  height:4px;border-radius:99px;background:var(--blue-2);
  margin:.5rem 0;position:relative;
}
.range-dot{
  position:absolute;top:50%;transform:translate(-50%,-50%);
  width:10px;height:10px;border-radius:50%;
  background:var(--blue);border:2px solid white;
  box-shadow:0 1px 4px rgba(37,99,235,.4);
}

/* ── Mini stats in result ── */
.mini-stats{display:grid;grid-template-columns:1fr 1fr;gap:0;flex:1;}
.mini-stat{
  padding:1rem 1.25rem;
  border-right:1px solid var(--border);
  border-bottom:1px solid var(--border);
}
.mini-stat:nth-child(2){border-right:none;}
.mini-stat:nth-child(3){border-bottom:none;}
.mini-stat:nth-child(4){border-right:none;border-bottom:none;}
.mini-stat-label{
  font-size:10px;font-weight:600;letter-spacing:.8px;text-transform:uppercase;
  color:var(--ink-3);margin-bottom:.3rem;
}
.mini-stat-value{
  font-family:'DM Mono',monospace;font-size:1.1rem;font-weight:500;color:var(--ink);
}
.mini-stat-value.blue   {color:var(--blue);}
.mini-stat-value.green  {color:var(--emerald);}
.mini-stat-value.amber  {color:var(--amber);}

/* ── Feature importance card (full width) ── */
.imp-card{
  background:var(--white);border:1px solid var(--border);
  border-radius:var(--radius-lg);box-shadow:var(--shadow-xs);
  margin-top:1rem;overflow:hidden;
}
.imp-header{
  padding:1.1rem 1.5rem;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
}
.imp-body{padding:1.25rem 1.5rem;}

/* Horizontal bars */
.imp-row{
  display:grid;grid-template-columns:140px 1fr 48px;
  align-items:center;gap:.875rem;margin-bottom:.6rem;
}
.imp-name{font-size:12px;font-weight:500;color:var(--ink-2);}
.imp-track{
  height:6px;background:var(--bg-2);border-radius:99px;
  overflow:hidden;border:1px solid var(--border);
}
.imp-fill{
  height:100%;border-radius:99px;
  background:linear-gradient(90deg,var(--blue) 0%,#60A5FA 100%);
  animation:growBar .7s ease both;
}
@keyframes growBar{from{width:0 !important;}to{}}
.imp-pct{
  font-family:'DM Mono',monospace;font-size:11px;
  color:var(--ink-3);text-align:right;
}

/* ── Model tag ── */
.model-tag{
  display:inline-flex;align-items:center;gap:6px;
  font-size:11px;font-weight:600;color:var(--ink-2);
  background:var(--bg);border:1px solid var(--border);
  border-radius:var(--radius-sm);padding:5px 12px;
}
.live-dot{
  width:6px;height:6px;border-radius:50%;background:var(--emerald);
  animation:pulse 2s infinite;
}
@keyframes pulse{
  0%,100%{opacity:1;transform:scale(1);}
  50%{opacity:.45;transform:scale(1.4);}
}

/* ── Spinner ── */
div[data-testid="stSpinner"] p{
  font-family:'Plus Jakarta Sans',sans-serif !important;
  color:var(--ink-2) !important;font-size:.85rem !important;
}

/* ── Footer ── */
.footer{
  margin-top:3rem;padding-top:1.25rem;
  border-top:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  font-size:11px;color:var(--ink-3);
}

/* ── Divider ── */
hr{border-color:var(--border) !important;margin:0 !important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("models/house_price_model.pkl","rb") as f: model=pickle.load(f)
    with open("metrics.json","r") as f: metrics=json.load(f)
    return model, metrics

model, metrics = load_artifacts()

# total_area probe
try:
    model.predict(pd.DataFrame({f:[0] for f in FEATURES}))
    MODEL_HAS_TOTAL_AREA = True
except:
    MODEL_HAS_TOTAL_AREA = False

# ─────────────────────────────────────────────────────────────
# Topbar
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar fade-1">
  <div class="topbar-logo">
    <div class="logo-icon">🏠</div>
    HomeVal <span style="color:#2563EB;margin-left:1px;">AI</span>
  </div>
  <div class="topbar-right">
    <span class="pill pill-green"><span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:#059669;margin-right:4px;vertical-align:middle;"></span>Model Live</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="page">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Hero banner — centered
# ─────────────────────────────────────────────────────────────
r2_val   = round(metrics["R2"], 4)
rmse_val = int(metrics["RMSE"])

st.markdown(f"""
<div class="hero-banner fade-2">
  <div class="hero-inner">
    <div class="hero-eyebrow">
      <span class="hero-dot"></span>
      AI-Powered Real Estate Valuation
    </div>
    <h1 class="hero-title">Property Valuation<br><span>Dashboard</span></h1>
    <p class="hero-sub">
      Machine-learning estimates trained on King County sales data.<br>
      Enter your property details below to generate an instant valuation.
    </p>
    <div class="hero-chips">
      <span class="hero-chip">🎯 {r2_val} R² Accuracy</span>
      <span class="hero-chip">📏 ±${rmse_val:,} Avg Error</span>
      <span class="hero-chip">🌲 Random Forest</span>
      <span class="hero-chip">🏘 60+ ZIP codes</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# stat grid moved into hero chips above

# ─────────────────────────────────────────────────────────────
# Dashboard: 2-col — inputs left, results right
# ─────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="medium")

# ── LEFT: Input card ──────────────────────────────────────────
with left:
    st.markdown("""
    <div class="card fade-4">
      <div class="card-header">
        <span class="card-title">Property Details</span>
      </div>
      <div class="card-body">
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1: bedrooms  = st.number_input("Bedrooms",  0, 10, 3, step=1)
    with c2: bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 2.0, step=0.5)

    sqft_living = st.number_input("Living Area (sq ft)", 300, 13000, 1500, step=50)

    grade = st.slider("Quality Grade", 1, 13, 7,
        help="1 = Cabin · 4 = Low quality · 7 = Average · 10 = High quality · 13 = Luxury mansion")

    # Grade label
    grade_labels = {1:"Cabin",2:"Substandard",3:"Poor",4:"Low",5:"Fair",
                    6:"Low Average",7:"Average",8:"Good",9:"Better",
                    10:"Very Good",11:"Excellent",12:"Luxury",13:"Mansion"}
    grade_colors = {**{i:"#6B7280" for i in range(1,7)},
                    7:"#2563EB",8:"#0891B2",9:"#059669",
                    10:"#D97706",11:"#DC2626",12:"#9333EA",13:"#B45309"}
    st.markdown(f"""
    <div style="text-align:right;margin-top:-8px;margin-bottom:8px;">
      <span style="font-size:11px;font-weight:600;padding:3px 10px;border-radius:99px;
        background:{grade_colors.get(grade,'#2563EB')}18;color:{grade_colors.get(grade,'#2563EB')};">
        Grade {grade} · {grade_labels.get(grade,'')}
      </span>
    </div>
    """, unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3: yr_built = st.number_input("Year Built",  1900, 2025, 2000, step=1)
    with c4: zipcode  = st.number_input("Zipcode",     value=98178, step=1)

    lat, lon = zip_to_coords(int(zipcode))

    predict_btn = st.button("→  Generate Valuation")
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Tips card below inputs
    st.markdown("""
    <div class="card fade-5" style="margin-top:1rem;">
      <div class="card-header"><span class="card-title">How it works</span></div>
      <div class="card-body" style="display:flex;flex-direction:column;gap:.6rem;">
        <div style="display:flex;gap:10px;align-items:flex-start;">
          <span style="font-size:16px;flex-shrink:0;">📐</span>
          <div><div style="font-size:12px;font-weight:600;color:#09090B;">Location-aware</div>
          <div style="font-size:11px;color:#71717A;">Zipcode maps to real lat/long for geo-accurate predictions.</div></div>
        </div>
        <div style="display:flex;gap:10px;align-items:flex-start;">
          <span style="font-size:16px;flex-shrink:0;">🌲</span>
          <div><div style="font-size:12px;font-weight:600;color:#09090B;">Random Forest model</div>
          <div style="font-size:11px;color:#71717A;">Trained on 19 features with hyperparameter tuning.</div></div>
        </div>
        <div style="display:flex;gap:10px;align-items:flex-start;">
          <span style="font-size:16px;flex-shrink:0;">📊</span>
          <div><div style="font-size:12px;font-weight:600;color:#09090B;">Market range</div>
          <div style="font-size:11px;color:#71717A;">Similar homes range based on grade, size, and location.</div></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── RIGHT: Results panel ──────────────────────────────────────
with right:
    if predict_btn:
        with st.spinner("Analyzing property data…"):
            time.sleep(0.9)
            data = {
                'bedrooms':[bedrooms],'bathrooms':[bathrooms],
                'sqft_living':[sqft_living],'sqft_lot':[5000],
                'floors':[1],'waterfront':[0],'view':[0],'condition':[3],
                'grade':[grade],'sqft_above':[sqft_living],'sqft_basement':[0],
                'yr_built':[yr_built],'yr_renovated':[0],'zipcode':[int(zipcode)],
                'lat':[lat],'long':[lon],
                'sqft_living15':[sqft_living],'sqft_lot15':[5000],
                'total_area':[sqft_living],
            }
            df = pd.DataFrame(data)
            if not MODEL_HAS_TOTAL_AREA: df = df.drop(columns=["total_area"])
            price = np.expm1(model.predict(df))[0]

        lo, hi = similar_homes_range(price, grade, sqft_living)
        age = 2025 - yr_built

        # Price meter: 0–2M scale
        meter_pct = min(price / 2_000_000 * 100, 100)
        if price > 800_000:
            badge = '<span class="price-badge badge-lux">★ Luxury</span>'
            meter_color = "linear-gradient(90deg,#F59E0B,#FCD34D)"
        elif price > 400_000:
            badge = '<span class="price-badge badge-mid">⬡ Mid-Range</span>'
            meter_color = "linear-gradient(90deg,#2563EB,#60A5FA)"
        else:
            badge = '<span class="price-badge badge-budget">✓ Budget</span>'
            meter_color = "linear-gradient(90deg,#059669,#34D399)"

        st.markdown(f"""
        <div class="result-panel fade-4">

          <!-- Price hero -->
          <div class="price-hero">
            <div class="price-eyebrow">Estimated Market Value</div>
            <div class="price-value">${price:,.0f}</div>
            <div class="price-confidence">Based on historical King County sales patterns</div>
            {badge}
          </div>

          <!-- Price meter -->
          <div class="meter-wrap">
            <div class="meter-label">
              <span>Market Position</span>
              <span style="font-family:'DM Mono',monospace;">{meter_pct:.0f}% of $2M scale</span>
            </div>
            <div class="meter-track">
              <div class="meter-fill" style="width:{meter_pct:.1f}%;background:{meter_color};"></div>
            </div>
            <div class="meter-ticks">
              <span class="meter-tick">$0</span>
              <span class="meter-tick">$500K</span>
              <span class="meter-tick">$1M</span>
              <span class="meter-tick">$1.5M</span>
              <span class="meter-tick">$2M</span>
            </div>
          </div>

          <!-- Similar homes -->
          <div class="similar-wrap">
            <div class="similar-header">🏘 Similar Homes Nearby</div>
            <div class="similar-range">${lo:,.0f} &nbsp;–&nbsp; ${hi:,.0f}</div>
            <div class="range-bar">
              <div class="range-dot" style="left:{((price-lo)/(hi-lo)*100) if hi>lo else 50:.0f}%;"></div>
            </div>
            <div class="similar-sub">Comparable properties · ZIP {int(zipcode)} · {lat:.3f}°N, {abs(lon):.3f}°W</div>
          </div>

          <!-- Mini stats grid -->
          <div class="mini-stats">
            <div class="mini-stat">
              <div class="mini-stat-label">Price / sq ft</div>
              <div class="mini-stat-value blue">${price/sqft_living:,.0f}</div>
            </div>
            <div class="mini-stat">
              <div class="mini-stat-label">Property Age</div>
              <div class="mini-stat-value">{age} yrs</div>
            </div>
            <div class="mini-stat">
              <div class="mini-stat-label">Bedrooms</div>
              <div class="mini-stat-value green">{bedrooms} bd / {bathrooms:.1f} ba</div>
            </div>
            <div class="mini-stat">
              <div class="mini-stat-label">Living Area</div>
              <div class="mini-stat-value amber">{sqft_living:,} sf</div>
            </div>
          </div>

        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-panel fade-4" style="min-height:420px;">
          <div class="price-hero" style="flex:1;justify-content:center;min-height:200px;">
            <div class="price-eyebrow">Estimated Market Value</div>
            <div class="price-value idle">Fill in details and click<br>Generate Valuation</div>
          </div>
          <div class="mini-stats">
            <div class="mini-stat"><div class="mini-stat-label">Price / sq ft</div><div class="mini-stat-value" style="color:#D4D4D8;">—</div></div>
            <div class="mini-stat"><div class="mini-stat-label">Property Age</div><div class="mini-stat-value" style="color:#D4D4D8;">—</div></div>
            <div class="mini-stat"><div class="mini-stat-label">Bedrooms</div><div class="mini-stat-value" style="color:#D4D4D8;">—</div></div>
            <div class="mini-stat"><div class="mini-stat-label">Living Area</div><div class="mini-stat-value" style="color:#D4D4D8;">—</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Feature Importance — full-width below
# ─────────────────────────────────────────────────────────────
imp_df = (
    pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_})
    .sort_values("Importance", ascending=False).head(10).reset_index(drop=True)
)
imp_df["Label"]  = imp_df["Feature"].map(LABEL_MAP).fillna(imp_df["Feature"])
max_imp = imp_df["Importance"].max()

# Split into two columns of 5
left5  = imp_df.iloc[:5]
right5 = imp_df.iloc[5:]

bars_left = bars_right = ""
for _, row in left5.iterrows():
    pct = row["Importance"]/max_imp*100
    bars_left += f"""
    <div class="imp-row">
      <div class="imp-name">{row['Label']}</div>
      <div class="imp-track"><div class="imp-fill" style="width:{pct:.1f}%"></div></div>
      <div class="imp-pct">{row['Importance']*100:.1f}%</div>
    </div>"""

for _, row in right5.iterrows():
    pct = row["Importance"]/max_imp*100
    bars_right += f"""
    <div class="imp-row">
      <div class="imp-name">{row['Label']}</div>
      <div class="imp-track"><div class="imp-fill" style="width:{pct:.1f}%"></div></div>
      <div class="imp-pct">{row['Importance']*100:.1f}%</div>
    </div>"""

st.markdown(f"""
<div class="imp-card fade-6">
  <div class="imp-header">
    <span class="card-title">What Drives the Price?</span>
    <div class="model-tag"><span class="live-dot"></span>Random Forest · 19 features</div>
  </div>
  <div class="imp-body">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;">
      <div>{bars_left}</div>
      <div>{bars_right}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer fade-6">
  <span>© 2025 HomeVal AI — Estimates are informational only, not appraisals.</span>
  <span>Random Forest · scikit-learn · Streamlit · King County Dataset</span>
</div>
</div>
""", unsafe_allow_html=True)