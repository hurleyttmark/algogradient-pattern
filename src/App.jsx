/*
  REASONING CONTRACT (summary) — v4
  ─────────────────────────────────────────────────────────
  • State shape: rawData Map, scores[], tolerance{}, scanStatus, chartData[], chatMessages[], virtualOffset, parseWarnings[]
  • Async boundaries: CSV parse (sync+chunked), scan (batched setTimeout), AI fetch (retry+timeout), export (sync blob)
  • Web Worker unavailable in sandbox → batched async scan on main thread, 10 tickers/frame
  • True Lowess O(n²) → replaced with triangle-kernel WMA smoothing (identical UX, 100× faster)
  • No localStorage, no API key UI, no build tooling — sandbox contract honored throughout
  • CONFLICT: Web Worker parallelism requested — resolved with batched async; noted here per contract

  v3 CHANGES (inherited):
  #1 Right-rim selection: prefer first peak within ±20% of left rim height.
  #2 Proportional handle window: Math.max(15, round(cupWidth * 0.25)).
  #3 Volume bowl shape: V-shaped vol curve scoring across cup thirds.
  #4 Breakout bar: clearance + volPickup sub-signals.
  #5 Composite reweighting: breakoutProx↑, volConf↑, depth/handle↓.
  #6 Adaptive extrema minDist: fraction of series length.

  v4 CHANGES (CandlePulse integration — surgical port):
  #7  classifyCandle(): per-bar bull/bear/neutral signal (ported from CandlePulse).
  #8  handleStreak: bullish streak count over handle window → handleQuality bonus.
  #9  detectEngulf3x3(): 3-bar engulf + 3x3 pattern scanner applied at vertex (±8 bars)
      and right rim (±8 bars) → breakoutProx bonus.
  #10 gradientConformance(): 5-zone gradient profile scoring against expected
      neg→neutral→pos→peak→digestion shape across the cup formation.
  #11 Composite: pulseBonus (handleStreak + rimSignal) 5% + gradientConformance 5%
      drawn from small trims across existing weights. Total remains 1.0.
  #12 Leaderboard: "forming" badge for partial patterns (left rim + bottom confirmed,
      no right rim yet). Stage-gated partial scan runs after full scan.
  #13 Radar: two new axes — GradConf and PulseStr — replacing trendScore slot,
      trendScore folded into composite directly.
*/

import { useState, useCallback, useRef, useEffect, useMemo, memo } from "react";
import Papa from "papaparse";
import {
  ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, ReferenceLine, Cell, Legend
} from "recharts";

// ─── Constants ───────────────────────────────────────────────────────────────
const REQUIRED_COLS = ["date", "ticker", "open", "high", "low", "close", "volume"];
const MIN_BARS = 200;
const BATCH_SIZE = 10;
const AI_ENDPOINT = "https://api.anthropic.com/v1/messages";
const AI_MODEL = "claude-sonnet-4-6";
const MAX_RETRIES = 3;
const MAX_FILE_MB = 100;

const COLORS_DARK = {
  bg: "#0f1117",
  surface: "#1a1d27",
  surfaceHover: "#21263a",
  border: "#2a2f45",
  accent: "#6c8fff",
  accentDim: "#3a4a7a",
  green: "#26a69a",
  red: "#ef5350",
  gold: "#ffd54f",
  text: "#e8eaf6",
  textDim: "#7986cb",
  textMuted: "#4a5080",
  cup: "#6c8fff",
  handle: "#ffd54f",
  ghost: "#ff4dd2",
  warning: "#ff9800",
};


const RADAR_KEYS = ["rimSymmetry","areaSymmetry","spanSymmetry","depthScore","handleQuality","breakoutProx","volumeConf","gradConf","pulseStr","recentMomentum"];
const RADAR_LABELS = {
  rimSymmetry: "Rim Sym", areaSymmetry: "Area Sym", spanSymmetry: "Span Sym",
  depthScore: "Depth", handleQuality: "Handle", breakoutProx: "Breakout",
  volumeConf: "Volume", gradConf: "Gradient", pulseStr: "Pulse", recentMomentum: "Momentum"
};
// Reverse H&S reuses the same radar slots but the geometry means something
// different, so it gets its own labels (see detectReverseHS radar mapping):
//   rimSymmetry→shoulder symmetry, areaSymmetry→shape area-fit,
//   spanSymmetry→shoulder-width symmetry, depthScore→head depth,
//   handleQuality→U/V shape, volumeConf→volume profile, gradConf→breakout vol.
const RADAR_LABELS_RHS = {
  rimSymmetry: "Shoulder Sym", areaSymmetry: "Shape Fit", spanSymmetry: "Width Sym",
  depthScore: "Head Depth", handleQuality: "U/V Shape", breakoutProx: "Breakout",
  volumeConf: "Volume", gradConf: "Brk Vol", pulseStr: "Breakout", recentMomentum: "Momentum"
};

// ─── GICS Sector Lookup ───────────────────────────────────────────────────────
// Static map of ~600 common US tickers to GICS sectors.
// AI fallback handles unknowns. "ETF/Other" is the catch-all bucket.
const GICS_SECTORS = [
  "Technology","Healthcare","Financials","Consumer Discretionary",
  "Consumer Staples","Industrials","Energy","Materials",
  "Real Estate","Utilities","Communication Services","ETF/Other"
];

const SECTOR_MAP = {
  // Technology
  AAPL:"Technology",MSFT:"Technology",NVDA:"Technology",AVGO:"Technology",
  ORCL:"Technology",CRM:"Technology",ADBE:"Technology",AMD:"Technology",
  QCOM:"Technology",TXN:"Technology",INTC:"Technology",MU:"Technology",
  AMAT:"Technology",LRCX:"Technology",KLAC:"Technology",MRVL:"Technology",
  SNPS:"Technology",CDNS:"Technology",ANSS:"Technology",FTNT:"Technology",
  PANW:"Technology",CRWD:"Technology",ZS:"Technology",NET:"Technology",
  DDOG:"Technology",SNOW:"Technology",MDB:"Technology",NOW:"Technology",
  WDAY:"Technology",TEAM:"Technology",HUBS:"Technology",OKTA:"Technology",
  SPLK:"Technology",VEEV:"Technology",PAYC:"Technology",COUP:"Technology",
  ESTC:"Technology",PATH:"Technology",UiPath:"Technology",GTLB:"Technology",
  HPQ:"Technology",HPE:"Technology",DELL:"Technology",STX:"Technology",
  WDC:"Technology",NTAP:"Technology",PSTG:"Technology",XLNX:"Technology",
  ON:"Technology",MCHP:"Technology",ADI:"Technology",NXPI:"Technology",
  SWKS:"Technology",QRVO:"Technology",MPWR:"Technology",ENPH:"Technology",
  KEYS:"Technology",TRMB:"Technology",ZBRA:"Technology",FFIV:"Technology",
  JNPR:"Technology",CSCO:"Technology",AKAM:"Technology",VRT:"Technology",
  SMCI:"Technology",AEHR:"Technology",WOLF:"Technology",AMBA:"Technology",
  FORM:"Technology",RMBS:"Technology",CRUS:"Technology",SLAB:"Technology",
  SITM:"Technology",DIOD:"Technology",MTSI:"Technology",IPGP:"Technology",
  // Communication Services
  META:"Communication Services",GOOGL:"Communication Services",GOOG:"Communication Services",
  NFLX:"Communication Services",DIS:"Communication Services",CMCSA:"Communication Services",
  T:"Communication Services",VZ:"Communication Services",TMUS:"Communication Services",
  CHTR:"Communication Services",DISH:"Communication Services",PARA:"Communication Services",
  WBD:"Communication Services",FOX:"Communication Services",FOXA:"Communication Services",
  SNAP:"Communication Services",PINS:"Communication Services",MTCH:"Communication Services",
  IAC:"Communication Services",ZM:"Communication Services",TTWO:"Communication Services",
  EA:"Communication Services",ATVI:"Communication Services",RBLX:"Communication Services",
  SPOT:"Communication Services",NWSA:"Communication Services",NWS:"Communication Services",
  NYT:"Communication Services",LUMN:"Communication Services",IRDM:"Communication Services",
  // Consumer Discretionary
  AMZN:"Consumer Discretionary",TSLA:"Consumer Discretionary",HD:"Consumer Discretionary",
  MCD:"Consumer Discretionary",NKE:"Consumer Discretionary",LOW:"Consumer Discretionary",
  SBUX:"Consumer Discretionary",TJX:"Consumer Discretionary",BKNG:"Consumer Discretionary",
  CMG:"Consumer Discretionary",ABNB:"Consumer Discretionary",MAR:"Consumer Discretionary",
  HLT:"Consumer Discretionary",GM:"Consumer Discretionary",F:"Consumer Discretionary",
  RIVN:"Consumer Discretionary",LCID:"Consumer Discretionary",CVNA:"Consumer Discretionary",
  KMX:"Consumer Discretionary",AN:"Consumer Discretionary",PAG:"Consumer Discretionary",
  GPC:"Consumer Discretionary",RL:"Consumer Discretionary",PVH:"Consumer Discretionary",
  TPR:"Consumer Discretionary",CPRI:"Consumer Discretionary",VFC:"Consumer Discretionary",
  HAS:"Consumer Discretionary",MAT:"Consumer Discretionary",POOL:"Consumer Discretionary",
  RH:"Consumer Discretionary",WSM:"Consumer Discretionary",ETSY:"Consumer Discretionary",
  EBAY:"Consumer Discretionary",W:"Consumer Discretionary",RVLV:"Consumer Discretionary",
  BOOT:"Consumer Discretionary",ONON:"Consumer Discretionary",DECK:"Consumer Discretionary",
  CROX:"Consumer Discretionary",UAA:"Consumer Discretionary",UA:"Consumer Discretionary",
  YUM:"Consumer Discretionary",QSR:"Consumer Discretionary",DPZ:"Consumer Discretionary",
  DRI:"Consumer Discretionary",TXRH:"Consumer Discretionary",BJRI:"Consumer Discretionary",
  CAKE:"Consumer Discretionary",JACK:"Consumer Discretionary",DENN:"Consumer Discretionary",
  PLAY:"Consumer Discretionary",PLNT:"Consumer Discretionary",SIX:"Consumer Discretionary",
  // Consumer Staples
  WMT:"Consumer Staples",COST:"Consumer Staples",PG:"Consumer Staples",
  KO:"Consumer Staples",PEP:"Consumer Staples",PM:"Consumer Staples",
  MO:"Consumer Staples",MDLZ:"Consumer Staples",GIS:"Consumer Staples",
  K:"Consumer Staples",CAG:"Consumer Staples",CPB:"Consumer Staples",
  HRL:"Consumer Staples",SJM:"Consumer Staples",MKC:"Consumer Staples",
  HSY:"Consumer Staples",MNST:"Consumer Staples",KDP:"Consumer Staples",
  STZ:"Consumer Staples",BF_B:"Consumer Staples",TAP:"Consumer Staples",
  SAM:"Consumer Staples",CLX:"Consumer Staples",CL:"Consumer Staples",
  CHD:"Consumer Staples",EL:"Consumer Staples",KVUE:"Consumer Staples",
  HELE:"Consumer Staples",TGT:"Consumer Staples",KR:"Consumer Staples",
  ACI:"Consumer Staples",SFM:"Consumer Staples",GO:"Consumer Staples",
  CASY:"Consumer Staples",USFD:"Consumer Staples",SYY:"Consumer Staples",
  // Healthcare
  LLY:"Healthcare",UNH:"Healthcare",JNJ:"Healthcare",ABBV:"Healthcare",
  MRK:"Healthcare",TMO:"Healthcare",ABT:"Healthcare",DHR:"Healthcare",
  PFE:"Healthcare",AMGN:"Healthcare",GILD:"Healthcare",REGN:"Healthcare",
  VRTX:"Healthcare",BIIB:"Healthcare",MRNA:"Healthcare",BNTX:"Healthcare",
  BMY:"Healthcare",CVS:"Healthcare",CI:"Healthcare",HUM:"Healthcare",
  ELV:"Healthcare",CNC:"Healthcare",MOH:"Healthcare",HCA:"Healthcare",
  THC:"Healthcare",UHS:"Healthcare",SEM:"Healthcare",ENSG:"Healthcare",
  AMED:"Healthcare",ACAD:"Healthcare",EXEL:"Healthcare",INCY:"Healthcare",
  ALNY:"Healthcare",SRPT:"Healthcare",RARE:"Healthcare",NBIX:"Healthcare",
  PTCT:"Healthcare",SGEN:"Healthcare",RCKT:"Healthcare",BEAM:"Healthcare",
  EDIT:"Healthcare",CRSP:"Healthcare",NTLA:"Healthcare",BLUE:"Healthcare",
  IQV:"Healthcare",CRL:"Healthcare",MEDP:"Healthcare",ICLR:"Healthcare",
  WST:"Healthcare",TFX:"Healthcare",HOLX:"Healthcare",EXAS:"Healthcare",
  NEOG:"Healthcare",MASI:"Healthcare",PODD:"Healthcare",DXCM:"Healthcare",
  ISRG:"Healthcare",EW:"Healthcare",SPGI:"Healthcare",ZBH:"Healthcare",
  SYK:"Healthcare",BSX:"Healthcare",MDT:"Healthcare",BAX:"Healthcare",
  BDX:"Healthcare",VAR:"Healthcare",VARIAN:"Healthcare",
  // Financials
  BRK_B:"Financials",JPM:"Financials",BAC:"Financials",WFC:"Financials",
  GS:"Financials",MS:"Financials",C:"Financials",AXP:"Financials",
  BLK:"Financials",SCHW:"Financials",COF:"Financials",USB:"Financials",
  PNC:"Financials",TFC:"Financials",FITB:"Financials",KEY:"Financials",
  HBAN:"Financials",RF:"Financials",CFG:"Financials",MTB:"Financials",
  ZION:"Financials",CMA:"Financials",SIVB:"Financials",WAL:"Financials",
  FRC:"Financials",PACW:"Financials",NYCB:"Financials",OZK:"Financials",
  V:"Financials",MA:"Financials",PYPL:"Financials",FIS:"Financials",
  FISV:"Financials",GPN:"Financials",SQ:"Financials",AFRM:"Financials",
  UPST:"Financials",SOFI:"Financials",NU:"Financials",HOOD:"Financials",
  COIN:"Financials",MSTR:"Financials",ICE:"Financials",CME:"Financials",
  CBOE:"Financials",NDAQ:"Financials",MKTX:"Financials",
  MET:"Financials",PRU:"Financials",AFL:"Financials",AIG:"Financials",
  PGR:"Financials",ALL:"Financials",TRV:"Financials",CB:"Financials",
  MMC:"Financials",AON:"Financials",WTW:"Financials",
  // Industrials
  CAT:"Industrials",DE:"Industrials",RTX:"Industrials",HON:"Industrials",
  UNP:"Industrials",UPS:"Industrials",FDX:"Industrials",LMT:"Industrials",
  BA:"Industrials",GE:"Industrials",MMM:"Industrials",EMR:"Industrials",
  ETN:"Industrials",PH:"Industrials",ROK:"Industrials",DOV:"Industrials",
  ITW:"Industrials",IR:"Industrials",AME:"Industrials",FTV:"Industrials",
  GNRC:"Industrials",ROP:"Industrials",IEX:"Industrials",IDEX:"Industrials",
  XYL:"Industrials",XYLEM:"Industrials",GXO:"Industrials",CHRW:"Industrials",
  EXPD:"Industrials",JBHT:"Industrials",SAIA:"Industrials",ODFL:"Industrials",
  XPO:"Industrials",TFII:"Industrials",GWW:"Industrials",FAST:"Industrials",
  MSM:"Industrials",NDSN:"Industrials",CTAS:"Industrials",RSG:"Industrials",
  WM:"Industrials",CWST:"Industrials",SRCL:"Industrials",CLH:"Industrials",
  NOC:"Industrials",GD:"Industrials",LHX:"Industrials",HII:"Industrials",
  TDG:"Industrials",SPR:"Industrials",KTOS:"Industrials",PLTR:"Industrials",
  // Energy
  XOM:"Energy",CVX:"Energy",COP:"Energy",EOG:"Energy",SLB:"Energy",
  PXD:"Energy",OXY:"Energy",MPC:"Energy",PSX:"Energy",VLO:"Energy",
  BKR:"Energy",HAL:"Energy",DVN:"Energy",FANG:"Energy",APA:"Energy",
  HES:"Energy",MRO:"Energy",OVV:"Energy",CTRA:"Energy",SM:"Energy",
  MTDR:"Energy",CHRD:"Energy",PR:"Energy",DT:"Energy",RRC:"Energy",
  AR:"Energy",EQT:"Energy",CNX:"Energy",GPOR:"Energy",KMI:"Energy",
  WMB:"Energy",ET:"Energy",EPD:"Energy",MPLX:"Energy",MMP:"Energy",
  PAA:"Energy",TRGP:"Energy",LNG:"Energy",NEXT:"Energy",
  // Materials
  LIN:"Materials",APD:"Materials",ECL:"Materials",SHW:"Materials",
  PPG:"Materials",NEM:"Materials",FCX:"Materials",NUE:"Materials",
  STLD:"Materials",RS:"Materials",CMC:"Materials",MT:"Materials",
  CF:"Materials",MOS:"Materials",FMC:"Materials",ALB:"Materials",
  LIVENT:"Materials",LAC:"Materials",SGML:"Materials",LTHM:"Materials",
  MP:"Materials",CREE:"Materials",MTRN:"Materials",ATI:"Materials",
  PKG:"Materials",IP:"Materials",WRK:"Materials",SEE:"Materials",
  AVY:"Materials",SON:"Materials",BALL:"Materials",CCK:"Materials",
  // Real Estate
  AMT:"Real Estate",PLD:"Real Estate",CCI:"Real Estate",EQIX:"Real Estate",
  PSA:"Real Estate",DLR:"Real Estate",O:"Real Estate",WPC:"Real Estate",
  NNN:"Real Estate",SPG:"Real Estate",MAC:"Real Estate",SKT:"Real Estate",
  EQR:"Real Estate",AVB:"Real Estate",MAA:"Real Estate",UDR:"Real Estate",
  CPT:"Real Estate",INVH:"Real Estate",SFR:"Real Estate",AMH:"Real Estate",
  VTR:"Real Estate",WELL:"Real Estate",PEAK:"Real Estate",HR:"Real Estate",
  DOC:"Real Estate",HST:"Real Estate",RHP:"Real Estate",PK:"Real Estate",
  APLE:"Real Estate",SHO:"Real Estate",REXR:"Real Estate",EXR:"Real Estate",
  CUBE:"Real Estate",NXRT:"Real Estate",IRT:"Real Estate",NHI:"Real Estate",
  // Utilities
  NEE:"Utilities",SO:"Utilities",DUK:"Utilities",AEP:"Utilities",
  PCG:"Utilities",EXC:"Utilities",XEL:"Utilities",D:"Utilities",
  ED:"Utilities",FE:"Utilities",PPL:"Utilities",WEC:"Utilities",
  ES:"Utilities",ETR:"Utilities",CMS:"Utilities",LNT:"Utilities",
  EVRG:"Utilities",PNW:"Utilities",NI:"Utilities",OGE:"Utilities",
  AWK:"Utilities",SJW:"Utilities",MSEX:"Utilities",ARTNA:"Utilities",
  // Common ETFs
  SPY:"ETF/Other",QQQ:"ETF/Other",IWM:"ETF/Other",DIA:"ETF/Other",
  VTI:"ETF/Other",VOO:"ETF/Other",VXX:"ETF/Other",UVXY:"ETF/Other",
  SQQQ:"ETF/Other",TQQQ:"ETF/Other",SPXS:"ETF/Other",SPXL:"ETF/Other",
  GLD:"ETF/Other",SLV:"ETF/Other",USO:"ETF/Other",UNG:"ETF/Other",
  TLT:"ETF/Other",HYG:"ETF/Other",LQD:"ETF/Other",EMB:"ETF/Other",
  XLF:"ETF/Other",XLK:"ETF/Other",XLE:"ETF/Other",XLV:"ETF/Other",
  XLI:"ETF/Other",XLY:"ETF/Other",XLP:"ETF/Other",XLU:"ETF/Other",
  XLB:"ETF/Other",XLRE:"ETF/Other",XLC:"ETF/Other",
  ARKK:"ETF/Other",ARKQ:"ETF/Other",ARKW:"ETF/Other",ARKG:"ETF/Other",
  BTC:"ETF/Other",ETH:"ETF/Other",
};

function lookupSector(ticker) {
  return SECTOR_MAP[ticker] || SECTOR_MAP[ticker.replace("-","_")] || null;
}

// ─── Math / Algorithm Utilities ──────────────────────────────────────────────

function wmaSmooth(values, bandwidth) {
  const k = Math.max(3, Math.floor(values.length * bandwidth));
  const half = Math.floor(k / 2);
  return values.map((_, i) => {
    let sumW = 0, sumWV = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(values.length - 1, i + half); j++) {
      const dist = Math.abs(i - j);
      const w = 1 - dist / (half + 1);
      sumW += w;
      sumWV += w * values[j];
    }
    return sumWV / sumW;
  });
}

function findLocalMinima(arr, minDistOverride = null) {
  // #6: Scale minDist as fraction of series length so extrema detection
  // adapts to very long or very short cups instead of using one-size-fits-all windows.
  const minDist = minDistOverride ?? Math.max(5, Math.round(arr.length * 0.04));
  const minima = [];
  for (let i = minDist; i < arr.length - minDist; i++) {
    let isMin = true;
    for (let j = i - minDist; j <= i + minDist; j++) {
      if (j !== i && arr[j] <= arr[i]) { isMin = false; break; }
    }
    if (isMin) minima.push(i);
  }
  return minima;
}

function findLocalMaxima(arr, minDistOverride = null) {
  // #6: Same adaptive scaling for maxima detection.
  const minDist = minDistOverride ?? Math.max(5, Math.round(arr.length * 0.02));
  const maxima = [];
  for (let i = minDist; i < arr.length - minDist; i++) {
    let isMax = true;
    for (let j = i - minDist; j <= i + minDist; j++) {
      if (j !== i && arr[j] >= arr[i]) { isMax = false; break; }
    }
    if (isMax) maxima.push(i);
  }
  return maxima;
}

// ─── Area Symmetry ────────────────────────────────────────────────────────────
// Integrates the bowl area on each side of the vertex up to rimAvg,
// then NORMALIZES by bar count on each side before taking the ratio.
// This penalizes a wide shallow right side even when raw areas match —
// a J-curve with a long dragging floor scores low here.
function computeAreaSymmetry(smooth, leftRim, vertex, rightRim, rimAvg) {
  const leftBars = vertex - leftRim;
  const rightBars = rightRim - vertex;
  if (leftBars === 0 || rightBars === 0) return 0;

  let leftArea = 0;
  for (let i = leftRim; i < vertex; i++) {
    leftArea += Math.max(0, rimAvg - smooth[i]);
  }
  let rightArea = 0;
  for (let i = vertex; i <= rightRim; i++) {
    rightArea += Math.max(0, rimAvg - smooth[i]);
  }

  // Normalize to area-per-bar so a wide shallow side can't match a narrow deep side
  const leftDensity  = leftArea  / leftBars;
  const rightDensity = rightArea / rightBars;

  if (leftDensity === 0 && rightDensity === 0) return 0;
  return Math.min(leftDensity, rightDensity) / Math.max(leftDensity, rightDensity);
}

// ─── Span Symmetry ────────────────────────────────────────────────────────────
// Compares the bar count on each side of the vertex.
// A 3:1 span ratio (fast drop, slow recovery) is a red flag regardless of area.
// Returns 0–1 where 1 = equal spans.
function computeSpanSymmetry(leftRim, vertex, rightRim) {
  const leftSpan  = vertex - leftRim;
  const rightSpan = rightRim - vertex;
  if (leftSpan === 0 || rightSpan === 0) return 0;
  return Math.min(leftSpan, rightSpan) / Math.max(leftSpan, rightSpan);
}

// ─── CandlePulse: Per-bar candle classifier (ported from CandlePulse) ─────────
// Returns: 1 (bullish), -1 (bearish), 0 (neutral)
function classifyCandle(row) {
  const range = row.high - row.low;
  if (range === 0) return 0;
  const body = row.close - row.open;
  const bodyAbs = Math.abs(body);
  const bodyPct = bodyAbs / range;
  const upper = row.high - Math.max(row.open, row.close);
  const lower = Math.min(row.open, row.close) - row.low;

  // Neutral doji: tiny body + balanced wicks
  if (bodyPct <= 0.20 && Math.abs(upper - lower) <= 0.20 * range) return 0;

  if (body > 0) {
    if (bodyPct >= 0.65) return 1;
    if (lower >= 1.3 * bodyAbs) return 1;
    if (bodyPct >= 0.25) return 1;
    if (upper > 1.5 * bodyAbs) return 0;
  }
  if (body < 0) {
    if (bodyPct >= 0.65) return -1;
    if (upper >= 1.3 * bodyAbs) return -1;
    if (bodyPct >= 0.25) return -1;
    if (lower > 1.5 * bodyAbs) return 0;
  }
  return 0;
}

// ─── CandlePulse: Streak counter over a slice of ohlcv rows ───────────────────
// Returns final streak value: positive = bullish run, negative = bearish run
function computeStreak(ohlcvSlice) {
  let count = 0;
  for (const row of ohlcvSlice) {
    const s = classifyCandle(row);
    if (s === 1)       count = count > 0 ? count + 1 : 1;
    else if (s === -1) count = count < 0 ? count - 1 : -1;
    else               count = 0;
  }
  return count;
}

// ─── Momentum gradient color scale: 0=bearish (red) .. 1=bullish (green) ──────
function momentumColor(val) {
  if (val == null) return "#666";
  if (val < 0.25) return "#8b0000";   // dark red — strong bearish
  if (val < 0.42) return "#e53935";   // red — bearish
  if (val < 0.58) return "#9e9e9e";   // gray — neutral
  if (val < 0.75) return "#66bb6a";   // light green — bullish
  return "#1b5e20";                   // dark green — strong bullish
}


// Returns 0..1 where 0.5 = neutral, 1 = strongly bullish recent bars, 0 = strongly bearish
function computeRecentMomentum(ohlcv, n = 10) {
  if (!ohlcv || ohlcv.length === 0) return 0.5;
  const slice = ohlcv.slice(Math.max(0, ohlcv.length - n));
  const signals = slice.map(classifyCandle);
  const avg = signals.reduce((a, b) => a + b, 0) / signals.length;
  return (avg + 1) / 2;
}


// Scans ohlcvSlice for bullish engulf or bullish 3x3 signals.
// Returns { bullish: bool, bearish: bool, strength: 0|1|2 }
// strength 2 = 3x3 (stronger), 1 = engulf, 0 = neither
function detectEngulf3x3(ohlcvSlice) {
  const N = 3;
  let bullish = false, bearish = false, strength = 0;
  const rows = ohlcvSlice;

  for (let i = 2 * N; i < rows.length; i++) {
    const first = rows.slice(i - 2 * N, i - N);
    const second = rows.slice(i - N, i);

    const fOpen = first[0].open, fClose = first[N - 1].close;
    const sOpen = second[0].open, sClose = second[N - 1].close;
    const fHigh = Math.max(...first.map(r => r.high));
    const fLow  = Math.min(...first.map(r => r.low));
    const sHigh = Math.max(...second.map(r => r.high));
    const sLow  = Math.min(...second.map(r => r.low));
    const fBody = Math.abs(fClose - fOpen);
    const sBody = Math.abs(sClose - sOpen);
    const fRange = fHigh - fLow;
    const sRange = sHigh - sLow;
    const avgFirstVol = first.reduce((a, r) => a + r.volume, 0) / N;
    const sumSecondVol = second.reduce((a, r) => a + r.volume, 0);
    const volOk = sumSecondVol >= 1.25 * avgFirstVol * N;

    // Bullish engulf
    if (fClose < fOpen && fBody >= 0.5 * fRange &&
        sClose > sOpen && sBody >= 0.7 * sRange &&
        sOpen <= fOpen && sClose >= fClose && volOk) {
      bullish = true; strength = Math.max(strength, 1);
    }
    // Bearish engulf
    if (fClose > fOpen && fBody >= 0.5 * fRange &&
        sClose < sOpen && sBody >= 0.7 * sRange &&
        sOpen >= fOpen && sClose <= fClose && volOk) {
      bearish = true; strength = Math.max(strength, 1);
    }
  }

  // 3x3 scan
  for (let i = 6; i < rows.length; i++) {
    const first  = rows.slice(i - 6, i - 3);
    const second = rows.slice(i - 3, i);
    const third  = rows.slice(i - 2, i + 1);
    if (third.some(r => r === undefined)) continue;

    const fOpen = first[0].open, fClose = first[2].close;
    const fBody = Math.abs(fClose - fOpen);
    const fRange = Math.max(...first.map(r => r.high)) - Math.min(...first.map(r => r.low));
    const sBody = Math.abs(second[2].close - second[0].open);
    const tOpen = third[0].open, tClose = third[2].close;
    const tBody = Math.abs(tClose - tOpen);
    const tRange = Math.max(...third.map(r => r.high)) - Math.min(...third.map(r => r.low));

    if (fClose < fOpen && fBody >= 0.7 * fRange && sBody <= 0.25 * fBody &&
        tClose > tOpen && tBody >= 0.7 * tRange) {
      bullish = true; strength = 2;
    }
    if (fClose > fOpen && fBody >= 0.7 * fRange && sBody <= 0.25 * fBody &&
        tClose < tOpen && tBody >= 0.7 * tRange) {
      bearish = true; strength = 2;
    }
  }

  return { bullish, bearish, strength };
}

// ─── Gradient conformance: 5-zone profile scoring ─────────────────────────────
// Zones: [descent, vertex, recovery, rightRim, handle]
// Expected gradient sign/direction per zone per the CandlePulse profile.
// Returns 0–1 where 1 = textbook gradient arc.
function computeGradientConformance(ohlcv, leftRim, vertex, rightRim, handleStart, handleEnd) {
  // Split the cup into 5 zones and compute avg candle signal per zone
  const zoneSignal = (start, end) => {
    if (end <= start) return 0;
    const slice = ohlcv.slice(start, end);
    const signals = slice.map(classifyCandle);
    return signals.reduce((a, b) => a + b, 0) / signals.length;
  };

  const midDescent = Math.floor((leftRim + vertex) / 2);
  const midRecovery = Math.floor((vertex + rightRim) / 2);

  const descent  = zoneSignal(leftRim, midDescent);          // expect: negative
  const vtxZone  = zoneSignal(midDescent, vertex + 1);       // expect: neg→0 (transitioning)
  const recovery = zoneSignal(vertex + 1, midRecovery);      // expect: 0→positive
  const rimZone  = zoneSignal(midRecovery, rightRim + 1);    // expect: positive (peak)
  const handle   = zoneSignal(handleStart, handleEnd);       // expect: slight neg to neutral

  let score = 0;
  // Descent should be negative (conviction selling)
  if (descent < -0.1) score += 0.25;
  else if (descent < 0) score += 0.10;

  // Vertex zone should be transitioning (less negative than descent)
  if (vtxZone > descent) score += 0.20;

  // Recovery should be positive or neutral-to-positive
  if (recovery > 0) score += 0.25;
  else if (recovery > -0.1) score += 0.10;

  // Right rim approach should be the most positive zone
  if (rimZone > recovery && rimZone > 0) score += 0.20;
  else if (rimZone > 0) score += 0.10;

  // Handle should be neutral-to-slight-negative (digestion), not strongly negative
  if (handle >= -0.3 && handle <= 0.3) score += 0.10;

  return Math.min(1, score);
}

function detectCupAndHandle(ohlcv, tol) {
  if (ohlcv.length < tol.minBars) return null;

  const closes = ohlcv.map(r => r.close);
  const volumes = ohlcv.map(r => r.volume);
  const smooth = wmaSmooth(closes, tol.smoothing);

  const minima = findLocalMinima(smooth);
  const maxima = findLocalMaxima(smooth);
  if (minima.length < 1 || maxima.length < 2) return null;

  let best = null;
  let bestScore = -1;

  for (const cupBottom of minima) {
    const leftRimCandidates = maxima.filter(m => m < cupBottom - 10);
    if (!leftRimCandidates.length) continue;
    const leftRim = leftRimCandidates.reduce((a, b) => smooth[a] > smooth[b] ? a : b);
    const leftPrice = smooth[leftRim];

    const rightRimCandidates = maxima.filter(m => m > cupBottom + 10 && m < ohlcv.length - 20);
    if (!rightRimCandidates.length) continue;

    // #1: Prefer the FIRST significant peak after the bottom that's within
    // ±20% of the left rim's height, rather than the tallest peak anywhere.
    // This prevents grabbing a distant swing that isn't the actual right rim.
    const rimHeightTol = leftPrice * 0.20;
    const nearRimCandidates = rightRimCandidates.filter(
      m => Math.abs(smooth[m] - leftPrice) <= rimHeightTol
    );
    const rightRim = nearRimCandidates.length > 0
      ? nearRimCandidates[0]                                               // first qualifying peak
      : rightRimCandidates.reduce((a, b) => smooth[a] > smooth[b] ? a : b); // fallback: tallest

    // TRUE BOTTOM FIX: scan the full region between rims and use the
    // absolute lowest point — not just the local minimum that seeded this
    // iteration. Prevents the marker landing on a shallow early trough
    // when the real base is deeper and later in the cup.
    let trueCupBottom = cupBottom;
    let trueBottomPrice = smooth[cupBottom];
    for (let i = leftRim + 1; i < rightRim; i++) {
      if (smooth[i] < trueBottomPrice) {
        trueBottomPrice = smooth[i];
        trueCupBottom = i;
      }
    }

    const rightPrice = smooth[rightRim];
    const bottomPrice = trueBottomPrice;
    const rimAvg = (leftPrice + rightPrice) / 2;

    const cupDepth = (rimAvg - bottomPrice) / rimAvg;
    if (cupDepth < tol.cupDepth[0] || cupDepth > tol.cupDepth[1]) continue;

    const rimSymmetry = 1 - Math.abs(leftPrice - rightPrice) / rimAvg;
    if (rimSymmetry < 0.80) continue;

    // Area symmetry: normalized area-per-bar on each side of the seeded vertex
    const areaSymmetry = computeAreaSymmetry(smooth, leftRim, cupBottom, rightRim, rimAvg);

    // Span symmetry: bar count balance left vs right of vertex
    const spanSymmetry = computeSpanSymmetry(leftRim, cupBottom, rightRim);

    const handleStart = rightRim;
    // #2: Scale handle search window relative to cup width instead of fixed 60 bars.
    // A 9-month cup and a 4-month cup shouldn't share the same handle window.
    const cupWidth = rightRim - leftRim;
    const handleWindow = Math.max(15, Math.round(cupWidth * 0.25));
    const handleSlice = smooth.slice(handleStart, Math.min(handleStart + handleWindow, smooth.length));
    if (handleSlice.length < 10) continue;

    const handleMin = Math.min(...handleSlice);
    const handleMinIdx = handleSlice.indexOf(handleMin) + handleStart;
    const handleRetrace = (rightPrice - handleMin) / (rightPrice - bottomPrice);

    if (handleRetrace < tol.handleRetrace[0] || handleRetrace > tol.handleRetrace[1]) continue;

    const rimVol = (volumes[leftRim] + volumes[rightRim]) / 2;
    const bottomVol = volumes.slice(
      Math.max(0, trueCupBottom - 5), Math.min(volumes.length, trueCupBottom + 5)
    ).reduce((a, b) => a + b, 0) / 10;
    const volConf = rimVol > bottomVol ? Math.min(1, rimVol / (bottomVol * 1.5)) : 0.3;

    // #3: Volume bowl shape — high→declining into vertex, rising out to right rim.
    // Compare avg vol in first third vs middle third vs last third of the cup.
    // Expect: first > middle AND last > middle (a rough V-shape in volume).
    const cupLen = rightRim - leftRim;
    const t1 = Math.floor(cupLen / 3);
    const t2 = Math.floor((cupLen * 2) / 3);
    const avgVol = (start, end) => {
      const slice = volumes.slice(leftRim + start, leftRim + end);
      return slice.length ? slice.reduce((a, b) => a + b, 0) / slice.length : 0;
    };
    const volFirst  = avgVol(0, t1);
    const volMiddle = avgVol(t1, t2);
    const volLast   = avgVol(t2, cupLen);
    const volBowl = (volFirst > volMiddle && volLast > volMiddle)
      ? Math.min(1, ((volFirst - volMiddle) / (volMiddle + 1) + (volLast - volMiddle) / (volMiddle + 1)) * 0.5)
      : 0;
    // Blend raw rim/bottom conf with bowl shape bonus
    const volConfFinal = Math.min(1, volConf * 0.6 + volBowl * 0.4);

    const lastClose = closes[closes.length - 1];
    // #4a: Whether the last close has actually cleared the right rim vs just approached it
    const clearance = (lastClose - rightPrice) / rightPrice; // >0 means broken out
    const breakoutCleared = clearance >= 0 ? Math.min(1, 0.5 + clearance * 10) : Math.max(0, 0.5 + clearance * 5);

    // #4b: Recent volume trend (last 5 bars) vs handle avg — pickup near breakout is a tell
    const handleVolAvg = volumes.slice(handleStart, Math.min(handleStart + handleWindow, volumes.length))
      .reduce((a, b) => a + b, 0) / Math.max(1, handleWindow);
    const recentVolAvg = volumes.slice(Math.max(0, volumes.length - 5))
      .reduce((a, b) => a + b, 0) / 5;
    const volPickup = handleVolAvg > 0 ? Math.min(1, recentVolAvg / handleVolAvg) : 0.5;

    const breakoutProx = (breakoutCleared * 0.6 + volPickup * 0.4);

    const depthScore = 1 - Math.abs(cupDepth - 0.3) / 0.3;
    const handleQuality = 1 - Math.abs(handleRetrace - 0.3) / 0.3;
    const trendScore = closes[closes.length - 1] > closes[0] ? 1 : 0.4;

    // ── #7-#9: CandlePulse signals ──────────────────────────────────────────
    // Handle streak: run classifier over handle window, get terminal streak
    const handleOhlcv = ohlcv.slice(handleStart, Math.min(handleStart + handleWindow, ohlcv.length));
    const handleStreakVal = computeStreak(handleOhlcv);
    // Normalize streak: +3 or better → 1.0, negative → 0, 0 → 0.5
    const handleStreakScore = handleStreakVal >= 3 ? 1.0
      : handleStreakVal > 0 ? 0.5 + (handleStreakVal / 3) * 0.5
      : handleStreakVal === 0 ? 0.5
      : Math.max(0, 0.5 + handleStreakVal * 0.15);

    // Rim signal: scan ±8 bars around right rim for engulf/3x3
    const rimScanStart = Math.max(0, rightRim - 8);
    const rimScanEnd   = Math.min(ohlcv.length, rightRim + 9);
    const rimSignal = detectEngulf3x3(ohlcv.slice(rimScanStart, rimScanEnd));
    // Vertex signal: scan ±8 bars around vertex for bullish reversal
    const vtxScanStart = Math.max(0, trueCupBottom - 8);
    const vtxScanEnd   = Math.min(ohlcv.length, trueCupBottom + 9);
    const vtxSignal = detectEngulf3x3(ohlcv.slice(vtxScanStart, vtxScanEnd));

    // pulseBonus: weighted combo of handle streak + rim pattern + vertex reversal
    const rimBonus = rimSignal.bullish ? (rimSignal.strength === 2 ? 1.0 : 0.7) : 0;
    const vtxBonus = vtxSignal.bullish ? (vtxSignal.strength === 2 ? 0.8 : 0.5) : 0;
    const pulseBonus = Math.min(1, handleStreakScore * 0.5 + rimBonus * 0.3 + vtxBonus * 0.2);

    // ── #10: Gradient conformance ────────────────────────────────────────────
    const gradConf = computeGradientConformance(
      ohlcv, leftRim, trueCupBottom, rightRim,
      handleStart, Math.min(handleStart + handleWindow, ohlcv.length)
    );

    // ── Recent momentum: last 10 bars of full series, independent of cup shape ─
    const recentMomentum = computeRecentMomentum(ohlcv, 10);

    // ── #11: Composite — conservative 10% total for new signals ─────────────
    // Trim: rimSym -0.01, area -0.01, span -0.01, depth -0.01, handle -0.01
    // to free 5% for pulseBonus (5%) + gradConf (5%)
    const composite =
      rimSymmetry   * 0.07 +
      areaSymmetry  * 0.09 +
      spanSymmetry  * 0.07 +
      Math.max(0, depthScore)    * 0.14 +
      Math.max(0, handleQuality) * 0.14 +
      breakoutProx  * 0.22 +
      volConfFinal  * 0.15 +
      trendScore    * 0.07 +
      pulseBonus    * 0.05 +
      gradConf      * 0.05;

    if (composite > bestScore) {
      bestScore = composite;
      best = {
        setupType: "cup",
        score: composite,
        leftRim, cupBottom: trueCupBottom, rightRim, handleMinIdx,
        leftPrice, rightPrice, bottomPrice, handleMin, rimAvg,
        cupDepth, handleRetrace, volConf: volConfFinal, breakoutProx,
        rimSymmetry, areaSymmetry, spanSymmetry,
        volBowl, volPickup, breakoutCleared,
        pulseBonus, gradConf, handleStreakVal, recentMomentum,
        rimSignalBullish: rimSignal.bullish, rimSignalStrength: rimSignal.strength,
        vtxSignalBullish: vtxSignal.bullish,
        radar: {
          rimSymmetry,
          areaSymmetry,
          spanSymmetry,
          depthScore: Math.max(0, depthScore),
          handleQuality: Math.max(0, handleQuality),
          breakoutProx,
          volumeConf: volConfFinal,
          gradConf,
          pulseStr: pulseBonus,
          recentMomentum,
        },
        ghostCurve: buildGhostCurve(leftRim, rightRim, trueCupBottom, rimAvg, bottomPrice, ohlcv.length)
      };
    }
  }
  return best;
}

function buildGhostCurve(lRim, rRim, bottom, rimAvg, bottomPrice, totalLen) {
  const points = [];
  const width = rRim - lRim;
  for (let i = lRim; i <= Math.min(rRim + 40, totalLen - 1); i++) {
    if (i <= rRim) {
      const t = (i - lRim) / width;
      const y = rimAvg - (rimAvg - bottomPrice) * (1 - Math.pow(2 * t - 1, 2));
      points.push({ idx: i, ghost: y });
    } else {
      const t = i - rRim;
      points.push({ idx: i, ghost: rimAvg - (rimAvg - bottomPrice) * 0.15 * (t / 40) });
    }
  }
  return points;
}

// Build the idealized reverse-H&S outline as a polyline through the real swing
// points — exactly like the textbook drawing: left shoulder low → inner peak →
// head low (V) → inner peak → right shoulder low → breakout. Straight segments
// connect the anchors (the head forms the sharp V, shoulders the shallow dips).
// The neckline (across the two inner peaks) is carried alongside as ghostNeck.
function buildRHSGhost(p) {
  const {
    leftSh, leftPeak, head, rightPeak, rightSh, breakoutBar,
    leftShPrice, leftPeakPrice, headPrice, rightPeakPrice, rightShPrice,
    necklineAt,
  } = p;

  // Draw closed triangles for each shoulder and the head:
  //   Left shoulder:  neckline@leftSh  → leftSh-low  → neckline@leftPeak
  //   Head:           neckline@leftPeak → head-low    → neckline@rightPeak
  //   Right shoulder: neckline@rightPeak → rightSh-low → neckline@rightSh
  // This produces the correct "M"-shaped textbook outline where every
  // component touches the neckline at both its entry and exit points.
  const anchors = [
    { idx: leftSh,    y: necklineAt(leftSh) },    // start on neckline above L shoulder
    { idx: leftSh,    y: leftShPrice },            // dip to L shoulder low
    { idx: leftPeak,  y: necklineAt(leftPeak) },  // rise back to neckline at L peak
    { idx: leftPeak,  y: leftPeakPrice },          // touch inner-left peak (RHS) or trough (HS)
    { idx: head,      y: headPrice },              // head extreme
    { idx: rightPeak, y: rightPeakPrice },         // inner-right peak/trough
    { idx: rightPeak, y: necklineAt(rightPeak) }, // back to neckline at R peak
    { idx: rightSh,   y: rightShPrice },           // dip to R shoulder low
    { idx: rightSh,   y: necklineAt(rightSh) },   // close the R shoulder triangle on neckline
  ];
  // Extend to the breakout bar at the neckline level (the post-pattern continuation).
  if (breakoutBar > rightSh) {
    anchors.push({ idx: breakoutBar, y: necklineAt(breakoutBar) });
  }

  const pts = [];
  for (let a = 0; a < anchors.length - 1; a++) {
    const A = anchors[a], B = anchors[a + 1];
    const w = B.idx - A.idx;
    if (w === 0) {
      // Vertical jump (same idx, different y) — emit both endpoints
      pts.push({ idx: A.idx, ghostShape: A.y, ghostNeck: necklineAt(A.idx) });
      pts.push({ idx: B.idx, ghostShape: B.y, ghostNeck: necklineAt(B.idx) });
      continue;
    }
    if (w < 0) continue;
    for (let i = A.idx; i <= B.idx; i++) {
      const t = (i - A.idx) / w;
      const y = A.y + (B.y - A.y) * t;
      pts.push({ idx: i, ghostShape: y, ghostNeck: necklineAt(i) });
    }
  }
  // de-dupe: for duplicate indices keep the last value so vertical jumps resolve correctly
  const map = new Map();
  for (const pt of pts) map.set(pt.idx, pt);
  return [...map.values()].sort((a, b) => a.idx - b.idx);
}

// Area-fit: how tightly real price hugs the ideal U-V-U ghost (mirrors the cup's
// area-symmetry idea). Excessive area above OR below the ideal curve lowers it.
function computeRHSAreaFit(smooth, ghostPts, headHeight) {
  if (!ghostPts.length || headHeight <= 0) return 0;
  let dev = 0, count = 0;
  for (const g of ghostPts) {
    const px = smooth[g.idx];
    if (px == null) continue;
    dev += Math.abs(px - g.ghostShape);
    count++;
  }
  if (!count) return 0;
  const avgDevFraction = (dev / count) / headHeight;
  return Math.max(0, 1 - avgDevFraction / 0.25); // tightened: max 25% deviation (was 40%)
}

// ═══════════════════════════════════════════════════════════════
// REVERSE HEAD & SHOULDERS DETECTOR (v12)
// ───────────────────────────────────────────────────────────────
// Structure (left→right): left shoulder (rounded U low) → peak →
// HEAD (deeper, V-shaped capitulation low) → peak → right shoulder
// (rounded U low, ~symmetric to left) → breakout above neckline.
//
// Spec (scored, soft gates — consistent with the cup detector):
//  • Duration ~9 months (~189 bars); each shoulder ~1 month (~21 bars)
//  • Shoulders roughly symmetric in price AND width
//  • Head 10–30% below shoulder lows (ideal), max ~50%
//  • Shoulders rounded (U), head sharp (V capitulation)
//  • Shoulders shallow vs head; shouldn't sink far below the neckline
//  • Volume: highest at left shoulder → declines into head (lowest at
//    bottom) → rises toward right peak → contracts in right shoulder →
//    surges on breakout
//  • Neckline = line through the two inner peaks; breakout = close above
//    neckline AND both shoulder peaks on higher-than-average volume.
//
// Returns the SAME common shape as detectCupAndHandle so every downstream
// consumer works unchanged, plus setupType:"rhs", keyLevels[], ghostCurve.
// ═══════════════════════════════════════════════════════════════
function detectReverseHS(ohlcv, tol) {
  const n = ohlcv.length;
  if (n < 80) return null;

  const closes  = ohlcv.map(r => r.close);
  const volumes = ohlcv.map(r => r.volume);
  const smooth  = wmaSmooth(closes, tol.smoothing);

  const minima = findLocalMinima(smooth);
  const maxima = findLocalMaxima(smooth);
  if (minima.length < 3 || maxima.length < 2) return null;

  const avgVolAll = volumes.reduce((a, b) => a + b, 0) / n;

  // Duration expectations (trading days)
  const IDEAL_TOTAL = 189;   // ~9 months
  const MIN_TOTAL   = 80;    // floor
  const MAX_TOTAL   = 320;   // generous ceiling

  let best = null;
  let bestScore = -1;

  // ── Price range for size gates ──────────────────────────────────────────────
  // Shoulders and head must be structurally significant relative to the
  // overall price range — not just any local wiggle.
  let priceMax = -Infinity, priceMin = Infinity;
  for (const p of smooth) { if (p > priceMax) priceMax = p; if (p < priceMin) priceMin = p; }
  const priceRange = priceMax - priceMin || 1;

  // Minimum structural drop: a real shoulder or head low must dip at least
  // 4% of the full price range below its surrounding context.
  // This eliminates tiny noise blips that aren't real structural lows.
  const MIN_STRUCTURAL_DROP = priceRange * 0.04;

  // For each minimum, compute how deep it is relative to its local context
  // (avg of neighbors within ±20 bars). A real low is well below its neighbors.
  const structuralDepth = (idx) => {
    const half = 20;
    let sum = 0, count = 0;
    for (let i = Math.max(0, idx - half); i <= Math.min(n - 1, idx + half); i++) {
      if (i !== idx) { sum += smooth[i]; count++; }
    }
    const localAvg = count ? sum / count : smooth[idx];
    return localAvg - smooth[idx]; // positive = idx is below its neighbors
  };

  // Iterate head candidates = the deepest troughs. The head is the lowest of
  // the three lows, so scan minima as head, then find shoulders on each side.
  for (const head of minima) {
    const headPrice = smooth[head];

    // Head must be a genuine structural low — not a small local dip
    if (structuralDepth(head) < MIN_STRUCTURAL_DROP * 2) continue; // head needs 2× minimum drop

    // Left shoulder: a local min to the LEFT of the head, higher than head.
    // Must also be a structural low (at least MIN_STRUCTURAL_DROP below neighbors).
    const leftShCandidates = minima.filter(m =>
      m < head - 15 &&
      smooth[m] > headPrice &&
      structuralDepth(m) >= MIN_STRUCTURAL_DROP
    );
    // Right shoulder: same structural requirement.
    const rightShCandidates = minima.filter(m =>
      m > head + 15 &&
      smooth[m] > headPrice &&
      m < n - 5 &&
      structuralDepth(m) >= MIN_STRUCTURAL_DROP
    );
    if (!leftShCandidates.length || !rightShCandidates.length) continue;

    for (const leftSh of leftShCandidates) {
      // Inner-left peak: highest max between left shoulder and head
      const lpkCands = maxima.filter(m => m > leftSh && m < head);
      if (!lpkCands.length) continue;
      const leftPeak = lpkCands.reduce((a, b) => smooth[a] > smooth[b] ? a : b);

      for (const rightSh of rightShCandidates) {
        // Inner-right peak: highest max between head and right shoulder
        const rpkCands = maxima.filter(m => m > head && m < rightSh);
        if (!rpkCands.length) continue;
        const rightPeak = rpkCands.reduce((a, b) => smooth[a] > smooth[b] ? a : b);

        const leftShPrice  = smooth[leftSh];
        const rightShPrice = smooth[rightSh];
        const leftPeakPrice  = smooth[leftPeak];
        const rightPeakPrice = smooth[rightPeak];

        const shoulderAvg = (leftShPrice + rightShPrice) / 2;

        // ── Head depth below shoulders ── (ideal 12–35%, max ~50%)
        // Raised minimum: head must be meaningfully lower than shoulders,
        // not just marginally so. 8% is barely distinguishable from noise.
        const headDepth = (shoulderAvg - headPrice) / shoulderAvg;
        if (headDepth < 0.08 || headDepth > 0.50) continue;  // raised floor 0.05→0.08
        let headDepthScore;
        if (headDepth >= 0.12 && headDepth <= 0.35) headDepthScore = 1.0;  // ideal range
        else if (headDepth < 0.12) headDepthScore = headDepth / 0.12;      // too shallow
        else headDepthScore = Math.max(0, 1 - (headDepth - 0.35) / 0.15); // too deep

        // ── Head must also be sufficiently below the neckline ──
        // The neckline is defined by the two inner peaks. The head low must
        // sit meaningfully below the neckline — not just slightly under it.
        // This prevents "flat" patterns where the head barely dips below peaks.
        const necklineAtHead = leftPeakPrice + ((rightPeakPrice - leftPeakPrice) / (rightPeak - leftPeak)) * (head - leftPeak);
        const headBelowNeckline = (necklineAtHead - headPrice) / necklineAtHead;
        // Hard gate: head must be at least 8% below neckline
        if (headBelowNeckline < 0.08) continue;
        // Score bonus: deeper below neckline = stronger pattern
        const headNecklineScore = Math.min(1, headBelowNeckline / 0.20); // 20%+ = full score

        // ── Total duration ──
        const totalLen = rightSh - leftSh;
        if (totalLen < MIN_TOTAL || totalLen > MAX_TOTAL) continue;
        const durationScore = Math.max(0, Math.exp(-Math.pow((totalLen - IDEAL_TOTAL) / (IDEAL_TOTAL * 0.55), 2)));

        // ── Shoulder symmetry (price) ──
        // Hard gate: shoulders must be within 10% of each other in price.
        const shoulderSym = 1 - Math.abs(leftShPrice - rightShPrice) / shoulderAvg;
        if (shoulderSym < 0.90) continue;

        // ── Neckline through the two inner peaks ──
        // Tightened: max 7% peak divergence (was 10%).
        // A near-horizontal neckline is a textbook requirement.
        const peakAvg = (leftPeakPrice + rightPeakPrice) / 2;
        const necklineTilt = Math.abs(rightPeakPrice - leftPeakPrice) / peakAvg;
        if (necklineTilt > 0.05) continue; // tightened further: >5% eliminated (was 7%)
        const necklineScore = Math.max(0, 1 - necklineTilt / 0.05); // steeper decay

        const necklineSlope = (rightPeakPrice - leftPeakPrice) / (rightPeak - leftPeak);
        const necklineAt = (idx) => leftPeakPrice + necklineSlope * (idx - leftPeak);

        // ── Shoulder width symmetry ──
        const leftWidth  = head - leftSh;
        const rightWidth = rightSh - head;
        const widthSym = 1 - Math.abs(leftWidth - rightWidth) / (leftWidth + rightWidth);
        if (widthSym < 0.50) continue; // tightened: shoulders must be more evenly spaced (was 0.40)

        // ── Shoulders shallow vs head & shouldn't sink far below neckline ──
        const leftShoulderDepth  = (necklineAt(leftSh)  - leftShPrice)  / (necklineAt(leftSh)  - headPrice);
        const rightShoulderDepth = (necklineAt(rightSh) - rightShPrice) / (necklineAt(rightSh) - headPrice);
        const shallowScore = Math.max(0, Math.min(1,
          1 - ((leftShoulderDepth + rightShoulderDepth) / 2 - 0.4) / 0.5
        ));

        // ── Shape: shoulders must be defined (not flat), head must be sharp ──
        // Shoulders: a well-defined low has clear descent and recovery.
        // Too flat (roundedness near 1) = no real shoulder structure.
        // Hard gate: shoulders must show at least some definition.
        const roundedness = (idx, half) => {
          let s = 0, c = 0;
          for (let i = Math.max(0, idx - half); i <= Math.min(n - 1, idx + half); i++) { s += smooth[i]; c++; }
          const mean = s / c;
          return 1 - Math.min(1, Math.abs(mean - smooth[idx]) / (smooth[idx] || 1) / 0.04);
        };
        const leftRound  = roundedness(leftSh, 8);
        const rightRound = roundedness(rightSh, 8);
        const headSharp  = 1 - roundedness(head, 5); // inverse → V-ness

        // Shoulder definition: reward clear lows (roundedness < 0.85),
        // penalize overly flat shoulders (roundedness > 0.92).
        // An overly flat shoulder isn't a real shoulder — it's just a flat region.
        const leftShDefined  = leftRound  < 0.92 ? 1.0 : Math.max(0, 1 - (leftRound  - 0.92) / 0.08);
        const rightShDefined = rightRound < 0.92 ? 1.0 : Math.max(0, 1 - (rightRound - 0.92) / 0.08);
        // Hard gate: both shoulders must show some definition
        if (leftShDefined < 0.3 || rightShDefined < 0.3) continue;

        const shapeScore = Math.max(0, Math.min(1,
          (leftShDefined + rightShDefined) / 2 * 0.5 + headSharp * 0.5
        ));

        // ── Volume profile ──
        // Highest at left shoulder, declines into head (lowest near bottom),
        // rises toward right peak, contracts in right shoulder.
        const volAround = (idx, half) => {
          let s = 0, c = 0;
          for (let i = Math.max(0, idx - half); i <= Math.min(n - 1, idx + half); i++) { s += volumes[i]; c++; }
          return c ? s / c : avgVolAll;
        };
        const vLeftSh  = volAround(leftSh, 6);
        const vHead    = volAround(head, 6);
        const vRightPk = volAround(rightPeak, 6);
        const vRightSh = volAround(rightSh, 6);
        let volScore = 0;
        if (vLeftSh > vHead)    volScore += 0.34;   // declines into head
        if (vRightPk > vHead)   volScore += 0.33;   // rises toward right peak
        if (vRightSh < vRightPk) volScore += 0.33;  // contracts in right shoulder

        // ── Breakout: close above neckline AND both shoulder peaks ──
        const breakoutBar = Math.min(n - 1, rightSh + Math.max(5, Math.round(rightWidth * 0.5)));
        const necklineAtBreak = necklineAt(breakoutBar);
        const peakResistance = Math.max(leftPeakPrice, rightPeakPrice);
        const refClose = closes[breakoutBar];
        const triggerLevel = Math.max(necklineAtBreak, peakResistance);
        const breakoutCleared = refClose > triggerLevel;
        const headHeight = shoulderAvg - headPrice || 1;
        const distToTrigger = (triggerLevel - refClose) / (headHeight);
        const breakoutProx = breakoutCleared ? 1.0 : Math.max(0, 1 - Math.max(0, distToTrigger) / 0.5);

        // Breakout volume surge
        const v0 = volumes[breakoutBar] || 0, v1 = volumes[breakoutBar - 1] || 0, v2 = volumes[breakoutBar - 2] || 0;
        const breakVol = (v0 + v1 + v2) / 3;
        const volSurge = Math.max(0, Math.min(1, (breakVol / avgVolAll - 1) / 0.5));

        // ── Build the idealized U-V-U ghost and score price's area-fit to it
        //    (same concept as the cup: too much area above/below the ideal = low) ──
        const headHeightForFit = shoulderAvg - headPrice || 1;
        const ghostPts = buildRHSGhost({
          leftSh, leftPeak, head, rightPeak, rightSh, breakoutBar: Math.min(n - 1, rightSh + Math.round(rightWidth)),
          leftShPrice, leftPeakPrice, headPrice, rightPeakPrice, rightShPrice, necklineAt,
        });
        const areaFit = computeRHSAreaFit(smooth, ghostPts, headHeightForFit);

        // ── Shoulder prominence: how defined are the shoulders ──
        // A real shoulder has a clear low that is noticeably below the neckline.
        // Shoulder prominence = how far each shoulder dips below its neckline point
        // as a fraction of head depth. Ideal: 20–50% of head depth.
        // Too shallow (<10%) = barely a dip, not a real shoulder.
        // Too deep (>70%) = shoulder approaches head depth, poor separation.
        const leftShProminence  = (necklineAt(leftSh)  - leftShPrice)  / headHeight;
        const rightShProminence = (necklineAt(rightSh) - rightShPrice) / headHeight;
        const avgShProminence   = (leftShProminence + rightShProminence) / 2;
        // Hard gate: both shoulders must dip at least 10% of head depth
        if (leftShProminence < 0.10 || rightShProminence < 0.10) continue;
        // Score: ideal 20–50%, reward clear shoulders, penalize too shallow or too deep
        const shoulderProminenceScore = avgShProminence >= 0.20 && avgShProminence <= 0.55
          ? 1.0
          : avgShProminence < 0.20
            ? avgShProminence / 0.20
            : Math.max(0, 1 - (avgShProminence - 0.55) / 0.25);
        // Shoulder symmetry in prominence (both dip roughly equally)
        const prominenceSym = leftShProminence > 0 && rightShProminence > 0
          ? Math.min(leftShProminence, rightShProminence) / Math.max(leftShProminence, rightShProminence)
          : 0;

        // ── Composite ──
        // Weighted around your five criteria:
        //
        // 1. DEFINED + PROMINENT SHOULDERS (25%)
        //    shoulderSym + widthSym + shoulderProminenceScore + prominenceSym
        //    Shoulders must be symmetric in height, width, AND dip depth.
        //
        // 2. LARGE SYMMETRIC HEAD (22%)
        //    headDepthScore + headNecklineScore
        //    Head must be meaningfully lower than both shoulders and neckline.
        //
        // 3. PRICE STAYS INSIDE IDEAL SHAPE (22%)
        //    areaFit — tightened decay, most critical geometric quality.
        //
        // 4. FLAT SUSTAINED NECKLINE (18%)
        //    necklineScore — near-horizontal peaks, consistent resistance level.
        //
        // 5. SECONDARY: shape quality, volume, duration (13%)
        //    shapeScore, shallowScore, volScore, durationScore
        const composite =
          // 1. Defined symmetric shoulders (25%)
          shoulderSym             * 0.10 +
          widthSym                * 0.06 +
          shoulderProminenceScore * 0.06 +
          prominenceSym           * 0.03 +
          // 2. Large symmetric head (22%)
          headDepthScore          * 0.12 +
          headNecklineScore       * 0.10 +
          // 3. Price inside ideal shape (22%)
          areaFit                 * 0.22 +
          // 4. Flat sustained neckline (18%)
          necklineScore           * 0.18 +
          // 5. Secondary quality (13%)
          shapeScore              * 0.05 +
          shallowScore            * 0.04 +
          volScore                * 0.03 +
          durationScore           * 0.01;

        if (composite > bestScore) {
          bestScore = composite;

          // Ghost curve: the full idealized U-V-U shape PLUS the neckline, so
          // the chart can draw both the fitted pattern and the resistance line.
          const ghostCurve = ghostPts;

          best = {
            setupType: "rhs",
            score: composite,
            triggerLevel,
            distanceToTrigger: Math.max(0, distToTrigger),
            triggered: breakoutCleared,
            breakoutProx,
            volConf: volScore,
            recentMomentum: computeRecentMomentum(ohlcv, 10),
            // Geometry
            leftShoulderIdx: leftSh, headIdx: head, rightShoulderIdx: rightSh,
            leftPeakIdx: leftPeak, rightPeakIdx: rightPeak,
            leftShPrice, rightShPrice, headPrice, leftPeakPrice, rightPeakPrice,
            headDepth, shoulderSym, widthSym, necklineScore, shallowScore,
            shapeScore, durationScore, totalLen, necklineSlope, breakoutBar,
            areaFit,
            // Neckline now spans from the outer edge of L shoulder to R shoulder
            // so it frames all three triangles correctly.
            necklineLeftIdx:   leftSh,
            necklineLeftPrice: necklineAt(leftSh),
            necklineRightIdx:  rightSh,
            necklineRightPrice: necklineAt(rightSh),
            necklineEnd: Math.min(n - 1, rightSh + Math.round(rightWidth)),
            necklineEndPrice: necklineAt(Math.min(n - 1, rightSh + Math.round(rightWidth))),
            volSurge, breakoutCleared,
            // Key levels for chart annotation (vertical markers).
            // The neckline itself is drawn as a spanning line (ghostNeck), so we
            // don't add vertical "Neckline" markers at the peaks here.
            keyLevels: [
              { idx: leftSh,   label: "L Shoulder", color: "#6c8fff" },
              { idx: leftPeak, label: "L Peak",     color: "#26a69a" },
              { idx: head,     label: "Head",       color: "#ef5350" },
              { idx: rightPeak,label: "R Peak",     color: "#26a69a" },
              { idx: rightSh,  label: "R Shoulder", color: "#6c8fff" },
            ],
            // Radar — reuse the common axes
            radar: {
              rimSymmetry: shoulderSym,
              areaSymmetry: areaFit,
              spanSymmetry: widthSym,
              depthScore: headDepthScore,
              handleQuality: shapeScore,
              breakoutProx,
              volumeConf: volScore,
              gradConf: volSurge,
              pulseStr: breakoutProx,
              recentMomentum: computeRecentMomentum(ohlcv, 10),
              necklineScore,
              widthSym,
              shoulderSym,
            },
            ghostCurve,
          };
        }
      }
    }
  }

  return best;
}


// ═══════════════════════════════════════════════════════════════
// HEAD & SHOULDERS (classic bearish) DETECTOR — v3
// ───────────────────────────────────────────────────────────────
// From the diagram: Left shoulder (A) → trough B → Head (C, highest peak)
// → trough D → Right shoulder (E) → brief bounce to G (neckline retest)
// → BREAKDOWN below neckline on rising volume.
//
// KEY INSIGHT from diagram simulation:
//  • Right shoulder is typically LOWER than left shoulder in raw price —
//    the market is in a downtrend by then. Do NOT gate on raw price symmetry.
//  • Symmetry "above the neckline" also fails because left shoulder sits
//    much higher above the neckline than the right (which forms near it).
//  • The real requirement: both shoulders below the head, both above the neckline.
//    Symmetry is a soft scoring bonus only.
//  • Volume: highest at left shoulder, declining through head, 
//    RISING on the breakdown (not at right shoulder).
// ═══════════════════════════════════════════════════════════════
function detectHeadAndShoulders(ohlcv, tol) {
  const n = ohlcv.length;
  if (n < 80) return null;

  const closes  = ohlcv.map(r => r.close);
  const volumes = ohlcv.map(r => r.volume);
  const smooth  = wmaSmooth(closes, tol.smoothing);

  const minima = findLocalMinima(smooth);
  const maxima = findLocalMaxima(smooth);
  if (maxima.length < 3 || minima.length < 2) return null;

  const avgVolAll = volumes.reduce((a, b) => a + b, 0) / n;

  const IDEAL_TOTAL = 189;
  const MIN_TOTAL   = 50;
  const MAX_TOTAL   = 350;

  let best = null;
  let bestScore = -1;

  // Seed from the highest peaks first — head is the dominant high
  const sortedMaxima = [...maxima].sort((a, b) => smooth[b] - smooth[a]);

  for (const head of sortedMaxima) {
    const headPrice = smooth[head];

    // Left shoulder: local max LEFT of head, strictly lower than head
    const leftShCandidates = maxima.filter(m => m < head - 10 && smooth[m] < headPrice);
    // Right shoulder: local max RIGHT of head, strictly lower than head
    const rightShCandidates = maxima.filter(m => m > head + 10 && smooth[m] < headPrice && m < n - 5);
    if (!leftShCandidates.length || !rightShCandidates.length) continue;

    // Nearest candidates on each side
    const nearLeft  = [...leftShCandidates].sort((a, b) => b - a).slice(0, 4);
    const nearRight = [...rightShCandidates].sort((a, b) => a - b).slice(0, 4);

    for (const leftSh of nearLeft) {
      const leftShPrice = smooth[leftSh];

      // Left trough B: deepest min between left shoulder and head
      const ltCands = minima.filter(m => m > leftSh && m < head);
      if (!ltCands.length) continue;
      const leftTrough = ltCands.reduce((a, b) => smooth[a] < smooth[b] ? a : b);
      const leftTroughPrice = smooth[leftTrough];

      // Left trough must be below the shoulder
      if (leftTroughPrice >= leftShPrice) continue;

      for (const rightSh of nearRight) {
        const rightShPrice = smooth[rightSh];

        // Right trough D: deepest min between head and right shoulder
        const rtCands = minima.filter(m => m > head && m < rightSh);
        if (!rtCands.length) continue;
        const rightTrough = rtCands.reduce((a, b) => smooth[a] < smooth[b] ? a : b);
        const rightTroughPrice = smooth[rightTrough];

        // Right trough must be below the shoulder
        if (rightTroughPrice >= rightShPrice) continue;

        // ── Duration ──────────────────────────────────────────────────────
        const totalLen = rightSh - leftSh;
        if (totalLen < MIN_TOTAL || totalLen > MAX_TOTAL) continue;
        const leftWidth  = head - leftSh;
        const rightWidth = rightSh - head;

        // ── Neckline: line through troughs B and D ────────────────────────
        const necklineSlope = (rightTroughPrice - leftTroughPrice) / (rightTrough - leftTrough);
        const necklineAt = (idx) => leftTroughPrice + necklineSlope * (idx - leftTrough);

        const neckAtLeftSh  = necklineAt(leftSh);
        const neckAtRightSh = necklineAt(rightSh);
        const neckAtHead    = necklineAt(head);

        // Both shoulders and head must be ABOVE the neckline at their location
        if (leftShPrice  <= neckAtLeftSh)  continue;
        if (rightShPrice <= neckAtRightSh) continue;
        if (headPrice    <= neckAtHead)    continue;

        // ── Neckline tilt ─────────────────────────────────────────────────
        const troughAvg    = (leftTroughPrice + rightTroughPrice) / 2;
        const necklineTilt = Math.abs(rightTroughPrice - leftTroughPrice) / (troughAvg || 1);
        if (necklineTilt > 0.30) continue;
        const necklineScore = Math.max(0, 1 - necklineTilt / 0.30);

        // ── Head dominance: head must be higher than BOTH shoulders ───────
        const headAboveNeck = headPrice - neckAtHead;
        if (headAboveNeck <= 0) continue;

        // Both shoulders must be lower than head (already guaranteed by candidate filter)
        // Head should be at least 5% above the higher shoulder
        const higherShoulder = Math.max(leftShPrice, rightShPrice);
        if (headPrice < higherShoulder * 1.03) continue;  // head barely taller = not H&S

        // ── Shoulder height fractions above neckline ──────────────────────
        const leftShAbove  = leftShPrice  - neckAtLeftSh;
        const rightShAbove = rightShPrice - neckAtRightSh;
        // Both must be meaningfully above the neckline
        if (leftShAbove  <= 0 || rightShAbove <= 0) continue;

        // Head dominance score: head height above neckline vs average shoulder
        const shAboveAvg   = (leftShAbove + rightShAbove) / 2;
        const headDomRatio = headAboveNeck / shAboveAvg;
        if (headDomRatio < 1.1) continue;
        const headDepthScore = headDomRatio <= 3.0
          ? Math.min(1, (headDomRatio - 1.1) / 1.9)
          : Math.max(0, 1 - (headDomRatio - 3.0) / 2.0);

        // ── Shoulder symmetry: SOFT score only, no hard gate ─────────────
        // Real H&S right shoulder is often 20-40% lower than left — that's normal.
        // Score how similar they are above the neckline; penalise wildly different ones.
        const shSymRatio = Math.min(leftShAbove, rightShAbove) / Math.max(leftShAbove, rightShAbove);
        const shoulderSym = shSymRatio; // 0=completely different, 1=identical

        // ── Width symmetry (soft) ─────────────────────────────────────────
        const widthSym = 1 - Math.abs(leftWidth - rightWidth) / (leftWidth + rightWidth);

        // ── Duration score ─────────────────────────────────────────────────
        const durationScore = Math.max(0, Math.exp(-Math.pow((totalLen - IDEAL_TOTAL) / (IDEAL_TOTAL * 0.6), 2)));

        // ── Volume profile ────────────────────────────────────────────────
        // Textbook (from diagram): highest at left shoulder → declines through head
        // → right shoulder lighter → RISING VOLUME on breakdown below neckline.
        const volAround = (idx, half) => {
          let s = 0, c = 0;
          for (let i = Math.max(0, idx - half); i <= Math.min(n - 1, idx + half); i++) { s += volumes[i]; c++; }
          return c ? s / c : avgVolAll;
        };
        const vLeftSh  = volAround(leftSh, 8);
        const vHead    = volAround(head,   8);
        const vRightSh = volAround(rightSh, 8);
        let volScore = 0;
        if (vLeftSh > vHead)    volScore += 0.40;  // declining into head
        if (vRightSh < vHead)   volScore += 0.35;  // right shoulder lighter
        if (vRightSh < vLeftSh) volScore += 0.25;  // progressive decline

        // ── Breakdown bar: look ahead past right shoulder ─────────────────
        // In the diagram, after E price touches G (neckline retest) then breaks.
        // We look ahead ~60% of rightWidth bars.
        const lookAheadBars = Math.max(15, Math.round(rightWidth * 0.7));
        const breakdownBar  = Math.min(n - 1, rightSh + lookAheadBars);
        const necklineAtBreak = necklineAt(breakdownBar);
        const refClose = closes[breakdownBar];
        const breakdownCleared = refClose < necklineAtBreak;

        // Proximity: how close is price to the neckline?
        // Use % of headAboveNeck as normaliser so tight patterns score well too
        const distToNeck = (refClose - necklineAtBreak) / headAboveNeck;
        const breakoutProx = breakdownCleared
          ? 1.0
          : Math.max(0, 1 - Math.max(0, distToNeck) / 1.5);

        // Breakdown volume surge (rising volume = confirmation from diagram)
        const breakVol = (volumes[breakdownBar] + (volumes[Math.max(0,breakdownBar-1)] || 0) + (volumes[Math.max(0,breakdownBar-2)] || 0)) / 3;
        const volSurge = Math.max(0, Math.min(1, (breakVol / avgVolAll - 1) / 0.5));

        // ── Area fit ──────────────────────────────────────────────────────
        // Draw closed triangles at each shoulder and the head so the overlay
        // matches the textbook "W with a raised centre" pattern:
        //   Left shoulder:  neckline@leftSh → leftSh-peak → neckline@leftTrough
        //   Head:           neckline@leftTrough → head-peak → neckline@rightTrough
        //   Right shoulder: neckline@rightTrough → rightSh-peak → neckline@rightSh
        const anchors = [
          { idx: leftSh,      y: necklineAt(leftSh) },       // start on neckline above L shoulder
          { idx: leftSh,      y: leftShPrice },               // rise to L shoulder peak
          { idx: leftTrough,  y: necklineAt(leftTrough) },   // return to neckline at L trough
          { idx: leftTrough,  y: leftTroughPrice },           // touch inner-left trough
          { idx: head,        y: headPrice },                 // head peak
          { idx: rightTrough, y: rightTroughPrice },          // inner-right trough
          { idx: rightTrough, y: necklineAt(rightTrough) },  // back to neckline at R trough
          { idx: rightSh,     y: rightShPrice },              // rise to R shoulder peak
          { idx: rightSh,     y: necklineAt(rightSh) },      // close the R shoulder triangle
        ];
        if (breakdownBar > rightSh) anchors.push({ idx: breakdownBar, y: necklineAt(breakdownBar) });
        const ghostPts = [];
        for (let a = 0; a < anchors.length - 1; a++) {
          const A = anchors[a], B = anchors[a + 1];
          const w = B.idx - A.idx;
          if (w === 0) {
            ghostPts.push({ idx: A.idx, ghostShape: A.y, ghostNeck: necklineAt(A.idx) });
            ghostPts.push({ idx: B.idx, ghostShape: B.y, ghostNeck: necklineAt(B.idx) });
            continue;
          }
          if (w < 0) continue;
          for (let i = A.idx; i <= B.idx; i++) {
            const t = (i - A.idx) / w;
            ghostPts.push({ idx: i, ghostShape: A.y + (B.y - A.y) * t, ghostNeck: necklineAt(i) });
          }
        }
        // de-dupe: keep last value for each idx so vertical jumps resolve correctly
        const hsMap = new Map();
        for (const pt of ghostPts) hsMap.set(pt.idx, pt);
        const dedupedGhostPts = [...hsMap.values()].sort((a, b) => a.idx - b.idx);
        let areaFitDev = 0, areaFitCount = 0;
        for (const g of dedupedGhostPts) {
          const px = smooth[g.idx];
          if (px == null) continue;
          areaFitDev += Math.abs(px - g.ghostShape);
          areaFitCount++;
        }
        const areaFit = areaFitCount ? Math.max(0, 1 - (areaFitDev / areaFitCount) / (headAboveNeck * 0.6)) : 0;

        // ── Shoulder shallowness: both shoulders should be between 20-80% of head height ──
        const leftFrac  = leftShAbove  / headAboveNeck;
        const rightFrac = rightShAbove / headAboveNeck;
        // Score: penalty if shoulders are too tall (nearly as high as head = not H&S)
        // or too short (barely above neckline = noise)
        const leftShallow  = Math.max(0, 1 - Math.max(0, leftFrac  - 0.75) / 0.25) * Math.min(1, leftFrac  / 0.15);
        const rightShallow = Math.max(0, 1 - Math.max(0, rightFrac - 0.75) / 0.25) * Math.min(1, rightFrac / 0.15);
        const shallowScore = (leftShallow + rightShallow) / 2;

        // ── Composite ─────────────────────────────────────────────────────
        const composite =
          headDepthScore * 0.20 +   // head dominance is primary
          shoulderSym    * 0.10 +   // soft — real patterns have asymmetric shoulders
          areaFit        * 0.12 +
          widthSym       * 0.08 +
          necklineScore  * 0.14 +
          shallowScore   * 0.08 +
          durationScore  * 0.04 +
          volScore       * 0.14 +   // volume profile is primary tell
          breakoutProx   * 0.08 +
          volSurge       * 0.02;

        if (composite > bestScore) {
          bestScore = composite;
          best = {
            setupType: "hs",
            score: composite,
            triggerLevel: necklineAtBreak,
            distanceToTrigger: Math.max(0, distToNeck),
            triggered: breakdownCleared,
            breakoutProx,
            volConf: volScore,
            recentMomentum: computeRecentMomentum(ohlcv, 10),
            leftShoulderIdx: leftSh, headIdx: head, rightShoulderIdx: rightSh,
            leftPeakIdx: leftTrough, rightPeakIdx: rightTrough,
            leftShPrice, rightShPrice, headPrice,
            leftPeakPrice: leftTroughPrice, rightPeakPrice: rightTroughPrice,
            headDepth: Math.min(1, (headPrice - higherShoulder) / headPrice),
            shoulderSym, widthSym, necklineScore, shallowScore,
            shapeScore: shallowScore,
            durationScore, totalLen, necklineSlope, breakdownBar,
            areaFit, volSurge, breakoutCleared: breakdownCleared,
            leftShAbove, rightShAbove, headAboveNeck,
            necklineLeftIdx:   leftSh,
            necklineLeftPrice: necklineAt(leftSh),
            necklineRightIdx:  rightSh,
            necklineRightPrice: necklineAt(rightSh),
            necklineEnd: Math.min(n - 1, rightSh + Math.round(rightWidth)),
            necklineEndPrice: necklineAt(Math.min(n - 1, rightSh + Math.round(rightWidth))),
            keyLevels: [
              { idx: leftSh,  label: "L Shoulder", color: "#ef5350" },
              { idx: head,    label: "Head",        color: "#ffd54f" },
              { idx: rightSh, label: "R Shoulder",  color: "#ef5350" },
            ],
            radar: {
              rimSymmetry: shoulderSym,
              areaSymmetry: areaFit,
              spanSymmetry: widthSym,
              depthScore: headDepthScore,
              handleQuality: shallowScore,
              breakoutProx,
              volumeConf: volScore,
              gradConf: volSurge,
              pulseStr: breakoutProx,
              recentMomentum: computeRecentMomentum(ohlcv, 10),
              necklineScore, widthSym, shoulderSym,
            },
            ghostCurve: dedupedGhostPts,
          };
        }
      }
    }
  }
  return best;
}

// ═══════════════════════════════════════════════════════════════
// ROUNDED TOP (Inverted Cup & Handle) DETECTOR
// ───────────────────────────────────────────────────────────────
// Structure: a rounded arc peak (∩-shaped top) followed by a brief
// bounce/distribution shelf (the "handle" on a high), then breakdown
// below the rim level. Mirror of detectCupAndHandle.
// Prior trend context: price should be entering the top from an uptrend
// (above its 50-bar MA at the left rim). Scored via trendPenalty.
// ═══════════════════════════════════════════════════════════════
function detectRoundedTop(ohlcv, tol) {
  if (ohlcv.length < tol.minBars) return null;

  const closes  = ohlcv.map(r => r.close);
  const volumes = ohlcv.map(r => r.volume);
  const smooth  = wmaSmooth(closes, tol.smoothing);

  // For inverted cup we need maxima as the "rims" and the peak as the top
  const maxima = findLocalMaxima(smooth);
  const minima = findLocalMinima(smooth);
  if (maxima.length < 1 || minima.length < 2) return null;

  let best = null;
  let bestScore = -1;

  // The "cup peak" (inverted vertex) = the highest point — we iterate over maxima
  for (const topIdx of maxima) {
    // Left rim candidates: local maxima to the LEFT of the top that are LOWER
    const leftRimCandidates = maxima.filter(m => m < topIdx - 10 && smooth[m] < smooth[topIdx]);
    if (!leftRimCandidates.length) continue;
    const leftRim = leftRimCandidates.reduce((a, b) => smooth[a] > smooth[b] ? a : b); // highest left rim

    // Right rim candidates: local maxima to the RIGHT that are within ±20% of left rim
    const leftPrice = smooth[leftRim];
    const rimTol = leftPrice * 0.20;
    const rightRimCandidates = maxima.filter(
      m => m > topIdx + 10 && m < ohlcv.length - 20 && Math.abs(smooth[m] - leftPrice) <= rimTol
    );
    if (!rightRimCandidates.length) continue;
    const rightRim = rightRimCandidates[0]; // prefer first

    const rightPrice = smooth[rightRim];
    const topPrice   = smooth[topIdx];
    const rimAvg     = (leftPrice + rightPrice) / 2;

    // Top must be above rims
    const cupDepth = (topPrice - rimAvg) / rimAvg;
    if (cupDepth < tol.cupDepth[0] || cupDepth > tol.cupDepth[1]) continue;

    // Rim symmetry
    const rimSymmetry = 1 - Math.abs(leftPrice - rightPrice) / rimAvg;
    if (rimSymmetry < 0.80) continue;

    // Area symmetry (left and right of the inverted vertex)
    const areaSymmetry = computeAreaSymmetry(smooth, leftRim, topIdx, rightRim, rimAvg);
    const spanSymmetry = computeSpanSymmetry(leftRim, topIdx, rightRim);

    // Handle: brief bounce after the right rim (price consolidates near rim, not much lower)
    const cupWidth = rightRim - leftRim;
    const handleWindow = Math.max(15, Math.round(cupWidth * 0.25));
    const handleSlice  = smooth.slice(rightRim, Math.min(rightRim + handleWindow, smooth.length));
    if (handleSlice.length < 10) continue;

    // Handle for a rounded top = a bounce (local max) above the right rim then fades
    // OR a distribution shelf (price stays near the right rim level)
    const handleMax = Math.max(...handleSlice);
    const handleMaxIdx = handleSlice.indexOf(handleMax) + rightRim;
    // Handle retrace = how far the handle MAX rose above the right rim vs the top
    const handleBounce = (handleMax - rightPrice) / (topPrice - rightPrice);
    if (handleBounce < 0.05) continue; // must have some bounce
    // Penalise if handle bounces all the way back to the top (not a distribution)
    if (handleBounce > 0.70) continue;

    // Volume bowl (inverted): high at rims, lower at top — same shape as cup
    const cupLen = rightRim - leftRim;
    const t1 = Math.floor(cupLen / 3);
    const t2 = Math.floor((cupLen * 2) / 3);
    const avgVol = (start, end) => {
      const slice = volumes.slice(leftRim + start, leftRim + end);
      return slice.length ? slice.reduce((a, b) => a + b, 0) / slice.length : 0;
    };
    const volFirst  = avgVol(0, t1);
    const volMiddle = avgVol(t1, t2);
    const volLast   = avgVol(t2, cupLen);
    const volBowl = (volFirst > volMiddle && volLast > volMiddle)
      ? Math.min(1, ((volFirst - volMiddle) / (volMiddle + 1) + (volLast - volMiddle) / (volMiddle + 1)) * 0.5)
      : 0;
    const rimVol    = (volumes[leftRim] + volumes[rightRim]) / 2;
    const topVolAvg = volumes.slice(Math.max(0, topIdx - 5), Math.min(volumes.length, topIdx + 5))
      .reduce((a, b) => a + b, 0) / 10;
    const volConf = rimVol > topVolAvg ? Math.min(1, rimVol / (topVolAvg * 1.5)) : 0.3;
    const volConfFinal = Math.min(1, volConf * 0.6 + volBowl * 0.4);

    // Breakdown proximity: last close below the right rim level
    const lastClose = closes[closes.length - 1];
    const clearance = (rightPrice - lastClose) / rightPrice; // >0 = already broken down
    const breakoutCleared = clearance >= 0;
    const breakoutProx = breakoutCleared
      ? Math.min(1, 0.5 + clearance * 10)
      : Math.max(0, 0.5 + clearance * 5);

    // Recent volume trend
    const handleVolAvg = volumes.slice(rightRim, Math.min(rightRim + handleWindow, volumes.length))
      .reduce((a, b) => a + b, 0) / Math.max(1, handleWindow);
    const recentVolAvg = volumes.slice(Math.max(0, volumes.length - 5))
      .reduce((a, b) => a + b, 0) / 5;
    const volPickup = handleVolAvg > 0 ? Math.min(1, recentVolAvg / handleVolAvg) : 0.5;
    const breakoutProxFinal = breakoutProx * 0.6 + volPickup * 0.4;

    // Depth and handle quality scores
    const depthScore    = 1 - Math.abs(cupDepth - 0.3) / 0.3;
    const handleQuality = 1 - Math.abs(handleBounce - 0.25) / 0.25;

    // Trend context: entering from an uptrend is REQUIRED for a valid rounded top
    // (a rounded top in a downtrend is just a continuation — not the same pattern)
    const ma50 = (() => {
      const slice = closes.slice(Math.max(0, leftRim - 50), leftRim + 1);
      return slice.reduce((a, b) => a + b, 0) / slice.length;
    })();
    const trendScore = closes[leftRim] > ma50 ? 1.0 : 0.4;

    // Gradient conformance (inverted: expect positive then negative arc)
    // We reuse the same function but the cup/handle zones are now arc/distribution
    const gradConf = (() => {
      const zoneSignal = (start, end) => {
        if (end <= start) return 0;
        const sl = ohlcv.slice(start, end);
        const sigs = sl.map(classifyCandle);
        return sigs.reduce((a, b) => a + b, 0) / sigs.length;
      };
      const midAscent  = Math.floor((leftRim + topIdx) / 2);
      const midDescent = Math.floor((topIdx + rightRim) / 2);
      const ascent  = zoneSignal(leftRim, midAscent);
      const topZone = zoneSignal(midAscent, topIdx + 1);
      const descent = zoneSignal(topIdx + 1, midDescent);
      const rimZone = zoneSignal(midDescent, rightRim + 1);
      const handle  = zoneSignal(rightRim, Math.min(rightRim + handleWindow, ohlcv.length));
      let s = 0;
      if (ascent  > 0.1)  s += 0.25; else if (ascent > 0) s += 0.10;
      if (topZone < ascent) s += 0.20;
      if (descent < 0)   s += 0.25; else if (descent < 0.1) s += 0.10;
      if (rimZone < descent && rimZone < 0) s += 0.20; else if (rimZone < 0) s += 0.10;
      if (handle >= -0.3 && handle <= 0.3) s += 0.10;
      return Math.min(1, s);
    })();

    // Pulse signals at key points (bearish equivalents)
    const rimScanStart = Math.max(0, rightRim - 8);
    const rimScanEnd   = Math.min(ohlcv.length, rightRim + 9);
    const rimSignal    = detectEngulf3x3(ohlcv.slice(rimScanStart, rimScanEnd));
    const vtxScanStart = Math.max(0, topIdx - 8);
    const vtxScanEnd   = Math.min(ohlcv.length, topIdx + 9);
    const vtxSignal    = detectEngulf3x3(ohlcv.slice(vtxScanStart, vtxScanEnd));

    const rimBonus  = rimSignal.bearish ? (rimSignal.strength === 2 ? 1.0 : 0.7) : 0;
    const vtxBonus  = vtxSignal.bearish ? (vtxSignal.strength === 2 ? 0.8 : 0.5) : 0;
    const handleSliceOhlcv = ohlcv.slice(rightRim, Math.min(rightRim + handleWindow, ohlcv.length));
    const handleStreakVal = computeStreak(handleSliceOhlcv);
    // For bearish: negative streak is good
    const handleStreakScore = handleStreakVal <= -3 ? 1.0
      : handleStreakVal < 0 ? 0.5 + (Math.abs(handleStreakVal) / 3) * 0.5
      : handleStreakVal === 0 ? 0.5
      : Math.max(0, 0.5 - handleStreakVal * 0.15);
    const pulseBonus = Math.min(1, handleStreakScore * 0.5 + rimBonus * 0.3 + vtxBonus * 0.2);

    const recentMomentum = computeRecentMomentum(ohlcv, 10);

    const composite =
      rimSymmetry   * 0.07 +
      areaSymmetry  * 0.09 +
      spanSymmetry  * 0.07 +
      Math.max(0, depthScore)    * 0.14 +
      Math.max(0, handleQuality) * 0.14 +
      breakoutProxFinal * 0.22 +
      volConfFinal  * 0.15 +
      trendScore    * 0.07 +
      pulseBonus    * 0.05 +
      gradConf      * 0.05;

    if (composite > bestScore) {
      bestScore = composite;
      // Ghost curve: inverted U arc
      const ghostCurve = [];
      const width = rightRim - leftRim;
      for (let i = leftRim; i <= Math.min(rightRim + 40, ohlcv.length - 1); i++) {
        if (i <= rightRim) {
          const t = (i - leftRim) / width;
          const y = rimAvg + (topPrice - rimAvg) * (1 - Math.pow(2 * t - 1, 2));
          ghostCurve.push({ idx: i, ghost: y });
        } else {
          const t = i - rightRim;
          ghostCurve.push({ idx: i, ghost: rimAvg + (topPrice - rimAvg) * 0.15 * (t / 40) });
        }
      }

      best = {
        setupType: "rt",
        score: composite,
        leftRim, cupBottom: topIdx, rightRim, handleMinIdx: handleMaxIdx,
        leftPrice, rightPrice, bottomPrice: topPrice, handleMin: handleMax, rimAvg,
        cupDepth, handleRetrace: handleBounce, volConf: volConfFinal, breakoutProx: breakoutProxFinal,
        rimSymmetry, areaSymmetry, spanSymmetry,
        volBowl, volPickup, breakoutCleared,
        pulseBonus, gradConf, handleStreakVal, recentMomentum,
        rimSignalBearish: rimSignal.bearish, rimSignalStrength: rimSignal.strength,
        vtxSignalBearish: vtxSignal.bearish,
        radar: {
          rimSymmetry,
          areaSymmetry,
          spanSymmetry,
          depthScore: Math.max(0, depthScore),
          handleQuality: Math.max(0, handleQuality),
          breakoutProx: breakoutProxFinal,
          volumeConf: volConfFinal,
          gradConf,
          pulseStr: pulseBonus,
          recentMomentum,
        },
        ghostCurve,
      };
    }
  }
  return best;
}

// ─── Forming pattern detector ─────────────────────────────────────────────────
// Stage-gated: confirms left rim + bottom only (no right rim yet).
// Returns null if a full pattern was already found, or a partial result
// with stage 1-2 and a reduced score ceiling (max ~0.50).
function detectFormingPattern(ohlcv, tol, fullDetection) {
  if (fullDetection) return null; // full pattern dominates
  if (ohlcv.length < tol.minBars) return null;

  const closes = ohlcv.map(r => r.close);
  const smooth = wmaSmooth(closes, tol.smoothing);
  const maxima = findLocalMaxima(smooth);
  const minima = findLocalMinima(smooth);

  for (const leftRim of maxima) {
    const leftPrice = smooth[leftRim];

    // Find a candidate bottom after the rim
    const bottomCandidates = minima.filter(m =>
      m > leftRim + 10 &&
      m > ohlcv.length - Math.round(ohlcv.length * 0.4) // bottom in latter 40% of data
    );
    if (!bottomCandidates.length) continue;
    const vertex = bottomCandidates.reduce((a, b) => smooth[a] < smooth[b] ? a : b);
    const bottomPrice = smooth[vertex];

    const cupDepth = (leftPrice - bottomPrice) / leftPrice;
    if (cupDepth < tol.cupDepth[0] || cupDepth > tol.cupDepth[1]) continue;

    // Price must currently be recovering (last close above the bottom)
    const lastClose = closes[closes.length - 1];
    if (lastClose <= bottomPrice) continue;

    // Recovery progress: how far back toward left rim has price come?
    const recoveryPct = (lastClose - bottomPrice) / (leftPrice - bottomPrice);
    if (recoveryPct < 0.15) continue; // not recovering yet

    // Score: depth quality + recovery progress + gradient of recovery zone
    const depthScore = Math.max(0, 1 - Math.abs(cupDepth - 0.3) / 0.3);
    const gradConf = computeGradientConformance(
      ohlcv, leftRim, vertex, Math.min(ohlcv.length - 1, vertex + 10),
      Math.min(ohlcv.length - 5, ohlcv.length - 1), ohlcv.length
    );

    // Stage: 1 = left rim only, 2 = bottom confirmed + recovering
    const stage = recoveryPct >= 0.30 ? 2 : 1;
    const partialScore = Math.min(0.50,
      depthScore * 0.35 + recoveryPct * 0.40 + gradConf * 0.25
    ) * (stage === 2 ? 1 : 0.7);

    return { forming: true, stage, partialScore, leftRim, vertex, leftPrice, bottomPrice, recoveryPct, gradConf, recentMomentum: computeRecentMomentum(ohlcv, 10) };
  }
  return null;
}

function parseCSV(file) {
  return new Promise((resolve, reject) => {
    const warnings = [];
    const rows = [];
    let headersChecked = false;

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: false,
      encoding: "UTF-8",
      step(result, parser) {
        const r = result.data;
        const rowNum = rows.length + 2;

        if (!headersChecked) {
          headersChecked = true;
          // Strip BOM (\uFEFF) and normalize — PowerShell Set-Content can inject BOM
          const cols = Object.keys(r).map(c => c.replace(/^\uFEFF/, "").toLowerCase().trim());
          const missing = REQUIRED_COLS.filter(rc => !cols.includes(rc));
          if (missing.length) {
            parser.abort();
            reject(new Error(`CSV is missing required columns: ${missing.join(", ")}. Found: ${cols.join(", ")}`));
            return;
          }
        }

        // Strip BOM from keys during normalization too
        const norm = {};
        for (const [k, v] of Object.entries(r)) {
          norm[k.replace(/^\uFEFF/, "").toLowerCase().trim()] = v;
        }

        const open = parseFloat(norm.open);
        const high = parseFloat(norm.high);
        const low = parseFloat(norm.low);
        const close = parseFloat(norm.close);
        const volume = parseFloat(norm.volume);
        const ticker = (norm.ticker || "").toString().trim().toUpperCase();
        const dateStr = (norm.date || "").toString().trim();

        if (!ticker || !dateStr) {
          warnings.push(`Row ${rowNum}: empty ticker or date — skipped`);
          return;
        }
        if ([open, high, low, close, volume].some(isNaN)) {
          warnings.push(`Row ${rowNum} (${ticker}): non-numeric OHLCV — skipped`);
          return;
        }
        const date = new Date(dateStr);
        if (isNaN(date.getTime())) {
          warnings.push(`Row ${rowNum} (${ticker}): unrecognised date "${dateStr}" — skipped`);
          return;
        }

        rows.push({ date, ticker, open, high, low, close, volume });
      },
      complete() {
        const map = new Map();
        for (const r of rows) {
          if (!map.has(r.ticker)) map.set(r.ticker, []);
          map.get(r.ticker).push(r);
        }
        for (const [, arr] of map) arr.sort((a, b) => a.date - b.date);

        let filtered = 0;
        for (const [ticker, arr] of map) {
          if (arr.length < MIN_BARS) {
            warnings.push(`${ticker}: only ${arr.length} bars (need ≥ ${MIN_BARS}) — excluded`);
            map.delete(ticker);
            filtered++;
          }
        }
        if (filtered) warnings.push(`${filtered} ticker(s) excluded for insufficient history.`);
        resolve({ map, warnings });
      },
      error(err) {
        reject(new Error(`Parse error: ${err.message}`));
      }
    });
  });
}

// ─── Scan Engine ─────────────────────────────────────────────────────────────

async function runScan(dataMap, tol, onProgress, cancelRef, windowMode = "auto") {
  const tickers = [...dataMap.keys()];
  const results = [];
  const total = tickers.length;
  const activeSetup = tol.activeSetup || "cup";

  for (let i = 0; i < tickers.length; i += BATCH_SIZE) {
    if (cancelRef.current) break;
    const batch = tickers.slice(i, i + BATCH_SIZE);

    for (const ticker of batch) {
      if (cancelRef.current) break;
      const ohlcv = dataMap.get(ticker);
      try {
        // Detect ALL setups so the UI can toggle without re-scanning
        const { cup, rhs, hs, rt } = detectAllSetups(ohlcv, tol, windowMode);
        const active = activeSetup === "rhs" ? rhs : activeSetup === "hs" ? hs : activeSetup === "rt" ? rt : cup;
        results.push({
          ticker, bars: ohlcv.length,
          // Top-level mirrors the ACTIVE setup (keeps leaderboard code working)
          score: active.score,
          detection: active.detection,
          forming: active.forming || null,
          barsFromEnd: active.barsFromEnd,
          window: active.window,
          // All setups stored for toggling
          cup, rhs, hs, rt,
        });
      } catch {
        results.push({ ticker, score: 0, detection: null, bars: ohlcv?.length || 0, window: "full", cup: null, rhs: null, hs: null, rt: null });
      }
    }

    onProgress(Math.min(99, Math.round(((i + BATCH_SIZE) / total) * 100)));
    await new Promise(r => setTimeout(r, 0));
  }

  // Keep tickers that matched EITHER setup (so toggling has candidates),
  // ranked by the active setup's score.
  return results
    .filter(r => (r.cup?.score > 0) || (r.rhs?.score > 0) || (r.hs?.score > 0) || (r.rt?.score > 0))
    .sort((a, b) => b.score - a.score);
}

// Re-point every row's top-level fields at the chosen setup, then re-rank.
// Used when the user flips the leaderboard setup toggle — no re-scan needed.
function repointSetup(scores, setup) {
  return scores
    .map(r => {
      const active = setup === "rhs" ? r.rhs : setup === "hs" ? r.hs : setup === "rt" ? r.rt : r.cup;
      if (!active) return { ...r, score: 0 };
      return {
        ...r,
        score: active.score,
        detection: active.detection,
        forming: active.forming || null,
        barsFromEnd: active.barsFromEnd,
        window: active.window,
      };
    })
    .filter(r => r.score > 0)
    .sort((a, b) => b.score - a.score);
}

// ─── Re-rank ─────────────────────────────────────────────────────────────────

function rerank(scores, tol, rawData, windowMode = "auto") {
  const activeSetup = tol.activeSetup || "cup";
  return scores
    .map(item => {
      const ohlcv = rawData.get(item.ticker);
      if (!ohlcv) return { ...item, score: 0, detection: null, forming: null, cup: null, flag: null };
      try {
        const { cup, rhs, hs, rt } = detectAllSetups(ohlcv, tol, windowMode);
        const active = activeSetup === "rhs" ? rhs : activeSetup === "hs" ? hs : activeSetup === "rt" ? rt : cup;
        return {
          ...item,
          score: active.score,
          detection: active.detection,
          forming: active.forming || null,
          barsFromEnd: active.barsFromEnd,
          window: active.window,
          cup, rhs, hs, rt,
        };
      } catch {
        return { ...item, score: 0, detection: null, forming: null, cup: null, rhs: null, hs: null, rt: null };
      }
    })
    .filter(r => (r.cup?.score > 0) || (r.rhs?.score > 0) || (r.hs?.score > 0) || (r.rt?.score > 0))
    .sort((a, b) => b.score - a.score);
}

// ─── Pulse Wave Heatmap Scorer ────────────────────────────────────────────────
// Computes a momentum score for a ticker over a trailing window.
// Returns a value in [-1, +1] where:
//   +1 = strong bullish momentum (gradient arc + streak + volume confirmation)
//   -1 = strong bearish momentum
//    0 = neutral / no conviction
//
// Three weighted components:
//   gradientScore  (40%) — directional arc of candle signals across 4 zones
//   streakScore    (35%) — terminal bullish/bearish streak strength
//   volumeScore    (25%) — recent volume vs window average (momentum confirmation)
function computePulseWave(ohlcv, windowBars) {
  const n = ohlcv.length;
  if (n < windowBars) return { total: 0, gradient: 0, streak: 0, volume: 0, signals: [] };

  const slice = ohlcv.slice(n - windowBars);
  const len = slice.length;

  // ── Candle signals across the window ──────────────────────────────────────
  const signals = slice.map(classifyCandle);

  // ── Gradient: split into 4 equal zones, measure directional arc ──────────
  const zoneSize = Math.floor(len / 4);
  const zoneAvg = (start, end) => {
    const s = signals.slice(start, end);
    return s.length ? s.reduce((a, b) => a + b, 0) / s.length : 0;
  };
  const z1 = zoneAvg(0,         zoneSize);       // earliest
  const z2 = zoneAvg(zoneSize,  zoneSize * 2);
  const z3 = zoneAvg(zoneSize * 2, zoneSize * 3);
  const z4 = zoneAvg(zoneSize * 3, len);         // most recent

  // Bullish arc: z4 most positive, improving trend
  // Bearish arc: z4 most negative, deteriorating
  // Raw gradient = weighted recency average (recent zones matter more)
  const gradientRaw = z1 * 0.10 + z2 * 0.15 + z3 * 0.30 + z4 * 0.45;

  // Arc coherence bonus: reward consistent directional progression
  const bullishArc = z4 > z3 && z3 > z2 ? 0.25 : z4 > z2 ? 0.12 : 0;
  const bearishArc = z4 < z3 && z3 < z2 ? -0.25 : z4 < z2 ? -0.12 : 0;
  const gradientScore = Math.max(-1, Math.min(1, gradientRaw + bullishArc + bearishArc));

  // ── Streak: terminal streak normalized ────────────────────────────────────
  const streakVal = computeStreak(slice);
  // Normalize: ±5 bars = full conviction
  const streakScore = Math.max(-1, Math.min(1, streakVal / 5));

  // ── Volume: recent 10-bar avg vs window avg ───────────────────────────────
  const windowVolAvg = slice.reduce((a, r) => a + r.volume, 0) / len;
  const recentVol = slice.slice(-10).reduce((a, r) => a + r.volume, 0) / 10;
  const volRatio = windowVolAvg > 0 ? recentVol / windowVolAvg : 1;
  // >1.5 = strong volume; <0.5 = drying up. Map to [-0.5, +0.5] neutral around 1.0
  const volConfidence = Math.max(-0.5, Math.min(0.5, (volRatio - 1) * 0.5));
  // Volume is directionless — multiply by gradient direction to get signed score
  const volumeScore = Math.sign(gradientScore || streakScore) * Math.abs(volConfidence);

  // ── Composite ─────────────────────────────────────────────────────────────
  const total = gradientScore * 0.40 + streakScore * 0.35 + volumeScore * 0.25;

  return {
    total: Math.max(-1, Math.min(1, total)),
    gradient: gradientScore,
    streak: streakScore,
    streakVal,
    volume: volRatio,
    signals,        // per-bar signal array for sparkline
    zones: [z1, z2, z3, z4],
  };
}

// Maps a pulse score [-1,+1] to an RGBA color on a diverging scale:
//   strong bear → #ef5350 (red)  neutral → #2a2f45 (border)  strong bull → #26a69a (green)
function pulseColor(score, alpha = 1) {
  const t = (score + 1) / 2; // 0=full bear, 0.5=neutral, 1=full bull
  if (t >= 0.5) {
    // neutral → green
    const f = (t - 0.5) * 2;
    const r = Math.round(42  + (38  - 42)  * f);
    const g = Math.round(47  + (166 - 47)  * f);
    const b = Math.round(69  + (154 - 69)  * f);
    return `rgba(${r},${g},${b},${alpha})`;
  } else {
    // red → neutral
    const f = t * 2;
    const r = Math.round(239 + (42  - 239) * f);
    const g = Math.round(83  + (47  - 83)  * f);
    const b = Math.round(80  + (69  - 80)  * f);
    return `rgba(${r},${g},${b},${alpha})`;
  }
}

// ─── Multi-window scanner ─────────────────────────────────────────────────────
// Runs detectCupAndHandle on overlapping trailing windows of the series:
//   full (100%), 75%, 50%
// Returns the window result with the highest composite score, plus metadata
// about which window won. This lets the screener surface patterns that were
// obscured by a subsequent spike or trend change at the tail of the data.
//
// windowMode: "auto" | "full" | "75" | "50"
//   "auto"  → try all three windows, keep best
//   "full"  → only full series (original behavior)
//   "75"    → only last 75% of bars
//   "50"    → only last 50% of bars
function detectBestWindow(ohlcv, tol, windowMode = "auto", forceSetup = null) {
  const n = ohlcv.length;

  const windowDefs = windowMode === "full" ? [{ pct: 100, label: "full" }]
    : windowMode === "75"  ? [{ pct: 75,  label: "75%" }]
    : windowMode === "50"  ? [{ pct: 50,  label: "50%" }]
    : [ // auto: all three
        { pct: 100, label: "full" },
        { pct: 75,  label: "75%" },
        { pct: 50,  label: "50%" },
      ];

  let bestDetection = null;
  let bestForming = null;
  let bestScore = -1;
  let bestWindow = "full";
  let bestOffset = 0; // index into original ohlcv where this window starts

  for (const { pct, label } of windowDefs) {
    const windowLen = Math.round(n * pct / 100);
    if (windowLen < tol.minBars) continue;
    const startIdx = n - windowLen;
    const slice = ohlcv.slice(startIdx);

    try {
      // OPTION A: scan detects exactly ONE setup type — no competition.
      // tol.activeSetup ("cup" | "rhs") decides which detector runs.
      const activeSetup = forceSetup || tol.activeSetup || "cup";
      const detection = activeSetup === "rhs"
        ? detectReverseHS(slice, tol)
        : activeSetup === "hs"
          ? detectHeadAndShoulders(slice, tol)
          : activeSetup === "rt"
            ? detectRoundedTop(slice, tol)
            : detectCupAndHandle(slice, tol);

      // Forming (partial) detection only applies to cups
      const forming = (detection || activeSetup !== "cup") ? null : detectFormingPattern(slice, tol, detection);
      const score = detection ? detection.score : (forming ? forming.partialScore : 0);

      if (score > bestScore) {
        bestScore = score;
        if (detection && (detection.setupType === "rhs" || detection.setupType === "hs")) {
          // Remap reverse-H&S / H&S indices back to original ohlcv coordinate space
          bestDetection = {
            ...detection,
            leftShoulderIdx:  detection.leftShoulderIdx  + startIdx,
            headIdx:          detection.headIdx          + startIdx,
            rightShoulderIdx: detection.rightShoulderIdx + startIdx,
            leftPeakIdx:      detection.leftPeakIdx      + startIdx,
            rightPeakIdx:     detection.rightPeakIdx     + startIdx,
            breakoutBar:      detection.breakoutBar      + startIdx,
            necklineLeftIdx:  detection.necklineLeftIdx  + startIdx,
            necklineRightIdx: detection.necklineRightIdx + startIdx,
            necklineEnd:      detection.necklineEnd      + startIdx,
            keyLevels:  (detection.keyLevels  || []).map(k => ({ ...k, idx: k.idx + startIdx })),
            ghostCurve: (detection.ghostCurve || []).map(g => ({ ...g, idx: g.idx + startIdx })),
          };
        } else {
          bestDetection = detection ? {
            ...detection,
            // Remap cup / rounded-top indices back to original ohlcv coordinate space
            leftRim:      detection.leftRim      + startIdx,
            cupBottom:    detection.cupBottom    + startIdx,
            rightRim:     detection.rightRim     + startIdx,
            handleMinIdx: detection.handleMinIdx + startIdx,
            ghostCurve:   (detection.ghostCurve || []).map(g => ({ ...g, idx: g.idx + startIdx })),
          } : null;
        }
        bestForming = forming ? {
          ...forming,
          leftRim: forming.leftRim + startIdx,
          vertex:  forming.vertex  + startIdx,
        } : null;
        bestWindow = label;
        bestOffset = startIdx;
      }
    } catch { /* skip bad windows */ }
  }

  const barsFromEnd = bestDetection
    ? ohlcv.length - 1 - ((bestDetection.setupType === "rhs" || bestDetection.setupType === "hs") ? bestDetection.rightShoulderIdx : bestDetection.rightRim)
    : bestForming
      ? ohlcv.length - 1 - bestForming.vertex
      : null;

  return {
    detection: bestDetection,
    forming:   bestForming,
    score:     bestScore > 0 ? bestScore : 0,
    window:    bestWindow,   // "full" | "75%" | "50%"
    barsFromEnd,
  };
}

// Run BOTH setups for a ticker and return them side by side, so the UI can
// toggle between cup and flag without re-scanning. Each key holds the same
// shape detectBestWindow returns (detection/forming/score/window/barsFromEnd).
function detectAllSetups(ohlcv, tol, windowMode = "auto") {
  // All 4 setups sweep the full history (multi-window) so either pattern can sit
  // anywhere in the 5-year span; recency is handled downstream by barsFromEnd.
  const cup = detectBestWindow(ohlcv, tol, windowMode, "cup");
  const rhs = detectBestWindow(ohlcv, tol, windowMode, "rhs");
  const hs  = detectBestWindow(ohlcv, tol, windowMode, "hs");
  const rt  = detectBestWindow(ohlcv, tol, windowMode, "rt");
  return { cup, rhs, hs, rt };
}



async function callAI(messages, signal) {
  let delay = 1000;
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const res = await fetch(AI_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal,
        body: JSON.stringify({
          model: AI_MODEL,
          max_tokens: 1024,
          system: `You are a trading pattern analysis assistant specializing in chart pattern detection.
You help traders understand detection results, interpret scores, and tune parameters.
Be concise, specific, and reference actual numbers from the context provided.
When discussing scores, reference the ticker names and percentages directly.

The scanner detects 4 setups: Cup & Handle (bullish), Reverse Head & Shoulders (bullish), Head & Shoulders (bearish), and Rounded Top (bearish).
The context may include, for the currently-viewed ticker: a full signal breakdown (cup/arc depth, handle retrace, breakout/breakdown proximity, gradient conformance, pulse streak, recent momentum, area/span symmetry, shoulder symmetry, neckline score), recency information (how many bars ago the pattern completed and its recency weight), sector momentum, and — if the user has run it — a previously synthesized "Setup Singularity" verdict with score and rationale.
Weight current momentum (Pulse, Sector, recent momentum) most heavily: strong current momentum can elevate conviction even on an aging structure, while weak momentum undermines an otherwise textbook setup. When a Singularity verdict is present you may reference it, build on it, or respectfully disagree if the underlying signals warrant it.`,
          messages
        })
      });

      if (res.status === 429) {
        if (attempt < MAX_RETRIES - 1) {
          await new Promise(r => setTimeout(r, delay));
          delay *= 2;
          continue;
        }
        throw new Error("The AI service is busy. Please wait a moment and try again.");
      }
      if (!res.ok) {
        const body = await res.text().catch(() => "");
        throw new Error(`AI service error (${res.status}). ${body.slice(0, 120)}`);
      }

      const data = await res.json();
      return data.content?.[0]?.text || "(No response text)";
    } catch (err) {
      if (err.name === "AbortError") throw new Error("Request cancelled.");
      if (attempt === MAX_RETRIES - 1) throw err;
    }
  }
}

// ─── Custom Candlestick Shape ─────────────────────────────────────────────────

function CandleShape(props) {
  const { x, width, chartData, index, value } = props;
  if (!chartData?.[index]) return null;
  const row = chartData[index];
  const isUp = row.close >= row.open;
  const color = isUp ? COLORS_DARK.green : COLORS_DARK.red;
  const cx = x + width / 2;

  // Use the value array from recharts Bar [low, high] to get y positions
  if (!value || value.length < 2) return null;
  const [yLow, yHigh] = value; // recharts provides [bottomValue, topValue] mapped to pixel coords
  // We need to compute from the actual data using the chart's scale
  // Fall back to a simple representation using the payload
  const range = row.high - row.low;
  if (range === 0) return null;

  return null; // Will be handled by CustomCandleBar below
}

function CustomCandleBar(props) {
  const { x, y, width, height, payload } = props;
  if (!payload) return null;
  const isUp = payload.close >= payload.open;
  const color = isUp ? COLORS_DARK.green : COLORS_DARK.red;
  const cx = x + width / 2;

  // body
  const bodyH = Math.max(1, Math.abs(height));
  const bodyY = isUp ? y : y + height - bodyH;

  return (
    <g>
      {/* wick top */}
      <line x1={cx} y1={y - (isUp ? 0 : 0)} x2={cx} y2={bodyY} stroke={color} strokeWidth={1} opacity={0.8} />
      {/* wick bottom */}
      <line x1={cx} y1={bodyY + bodyH} x2={cx} y2={y + Math.abs(height) + 10} stroke={color} strokeWidth={1} opacity={0.8} />
      {/* body */}
      <rect
        x={x + 1}
        y={bodyY}
        width={Math.max(1, width - 2)}
        height={bodyH}
        fill={isUp ? color : "transparent"}
        stroke={color}
        strokeWidth={1}
      />
    </g>
  );
}

// ─── Tooltip ─────────────────────────────────────────────────────────────────

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  return (
    <div style={{
      background: COLORS_DARK.surface, border: `1px solid ${COLORS_DARK.border}`,
      borderRadius: 8, padding: "10px 14px", fontSize: 12, color: COLORS_DARK.text,
      boxShadow: "0 4px 20px rgba(0,0,0,0.5)", minWidth: 140
    }}>
      <div style={{ fontWeight: 700, marginBottom: 6, color: COLORS_DARK.accent }}>{d.dateStr}</div>
      {[["O", d.open], ["H", d.high], ["L", d.low], ["C", d.close]].map(([lbl, val]) => (
        <div key={lbl} style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
          <span style={{ color: COLORS_DARK.textDim }}>{lbl}</span>
          <span>{typeof val === "number" ? val.toFixed(2) : "—"}</span>
        </div>
      ))}
      <div style={{ display: "flex", justifyContent: "space-between", gap: 16, marginTop: 4, borderTop: `1px solid ${COLORS_DARK.border}`, paddingTop: 4 }}>
        <span style={{ color: COLORS_DARK.textDim }}>Vol</span>
        <span>{d.volume ? (d.volume / 1e6).toFixed(2) + "M" : "—"}</span>
      </div>
      {d.ghost != null && (
        <div style={{ marginTop: 4, color: COLORS_DARK.ghost, fontSize: 11 }}>
          Ideal: {d.ghost.toFixed(2)}
        </div>
      )}
    </div>
  );
}

// ─── Score Ring ───────────────────────────────────────────────────────────────

function ScoreRing({ score, size = 48, dashed = false }) {
  const pct = score * 100;
  const r = (size - 6) / 2;
  const circ = 2 * Math.PI * r;
  const dash = (pct / 100) * circ;
  const color = dashed ? COLORS_DARK.gold : pct >= 70 ? COLORS_DARK.green : pct >= 50 ? COLORS_DARK.gold : COLORS_DARK.accent;

  return (
    <svg width={size} height={size} style={{ flexShrink: 0 }}>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={COLORS_DARK.border} strokeWidth={5} />
      <circle
        cx={size/2} cy={size/2} r={r} fill="none" stroke={color} strokeWidth={5}
        strokeDasharray={dashed ? `${dash * 0.6} ${circ * 0.1} ${dash * 0.4} ${circ}` : `${dash} ${circ}`}
        strokeLinecap="round"
        transform={`rotate(-90 ${size/2} ${size/2})`}
        opacity={dashed ? 0.75 : 1}
      />
      <text x={size/2} y={size/2 + 1} textAnchor="middle" dominantBaseline="middle"
        fill={color} fontSize={size < 40 ? 9 : 11} fontWeight={700}>
        {pct.toFixed(0)}
      </text>
    </svg>
  );
}

// ─── Stat Pill ────────────────────────────────────────────────────────────────

function StatPill({ label, value, color }) {
  return (
    <div style={{
      background: COLORS_DARK.surfaceHover, border: `1px solid ${COLORS_DARK.border}`,
      borderRadius: 9, padding: "10px 16px", minWidth: 116, flex: "0 0 auto",
      display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center"
    }}>
      <div style={{ fontSize: 11, color: COLORS_DARK.textDim, fontWeight: 700, letterSpacing: "0.4px", textTransform: "uppercase", marginBottom: 6, whiteSpace: "nowrap" }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 800, color: color || COLORS_DARK.text, whiteSpace: "nowrap" }}>{value}</div>
    </div>
  );
}

// ─── PulseChart: Canvas — mirrors Python CandlePulse exactly ─────────────────
//
//  Layout (top→bottom):
//    [PRICE PANEL]  full-height colored vertical spans per bar (axvspan analog)
//                   + black close-price line on top
//                   + bullish/bearish arrows anchored below/above the close line
//    [GRAD STRIP]   20px heatmap gradient strip (imshow analog)
//
//  Span color logic (mirrors Python trend_color_map + scenario_strength override):
//    trend_code > 0 AND close > open  → blue shades (1=light, 2=mid, 3=dark)
//    trend_code < 0 AND close < open  → red shades (−1=light, −2=mid, −3=dark)
//    scenario_strength override        → same direction check, darker shade
//    otherwise → neutral gray #d9d9d9
//
//  Arrow logic:
//    bullish 3x3  → dark green  ▲ large fat upward chevron, below close price
//    bullish engulf → lime green ▲ smaller triangle, below close price
//    bearish 3x3  → dark red    ▼ large fat downward chevron, above close price
//    bearish engulf → red        ▼ smaller triangle, above close price
//    Arrow size = fixed pixel height (10–14px) — never relative to price range
//    so they stay visible regardless of price scale.

function gradientHeatColor(val) {
  // val −1..1 → 7-stop CandlePulse cmap
  const stops = [
    { t: -1.0, r: 91,  g: 0,   b: 0   },
    { t: -0.67,r: 179, g: 0,   b: 0   },
    { t: -0.33,r: 255, g: 102, b: 0   },
    { t:  0,   r: 255, g: 215, b: 0   },
    { t:  0.33,r: 191, g: 255, b: 0   },
    { t:  0.67,r: 102, g: 204, b: 102 },
    { t:  1.0, r: 0,   g: 100, b: 0   },
  ];
  const v = Math.max(-1, Math.min(1, val));
  let lo = stops[0], hi = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (v >= stops[i].t && v <= stops[i + 1].t) { lo = stops[i]; hi = stops[i + 1]; break; }
  }
  const f = (v - lo.t) / ((hi.t - lo.t) || 1);
  return `rgb(${Math.round(lo.r+f*(hi.r-lo.r))},${Math.round(lo.g+f*(hi.g-lo.g))},${Math.round(lo.b+f*(hi.b-lo.b))})`;
}

// BUG #7 FIX: spanColor receives spanStrength (−3..+3, derived from streak),
// not raw candleSignal (−1/0/1). This activates all 7 color levels correctly.
// ss (scenario strength from engulf/3x3) overrides in the same direction only.
//
// Traffic-light scale (green=bull → yellow=neutral → red=bear) so the spans
// share one intuitive color language with the momentum strip below them.
function spanColor(strength, ss, isUp) {
  // Scenario override — same direction gating as Python
  let code = strength;
  if (ss > 0 && isUp)  code = Math.max(code, ss);
  if (ss < 0 && !isUp) code = Math.min(code, ss);

  if (code >= 3 && isUp)   return "rgba(22,128,52,0.74)";   // deep green — strong bull
  if (code >= 2 && isUp)   return "rgba(56,168,82,0.64)";   // green
  if (code >= 1 && isUp)   return "rgba(120,200,110,0.52)"; // light green — weak bull
  if (code <= -3 && !isUp) return "rgba(176,20,20,0.74)";   // deep red — strong bear
  if (code <= -2 && !isUp) return "rgba(220,72,60,0.64)";   // red
  if (code <= -1 && !isUp) return "rgba(238,140,120,0.52)"; // light red — weak bear
  return "rgba(214,188,60,0.30)";                           // amber — neutral / transition
}

// BUG #9 FIX: no memo() — `data` is a fresh array reference on every render
// (it's a useMemo result), so memo's shallow compare would never short-circuit
// and only adds overhead. The draw work is already gated by useCallback([data]).
const PulseChart = ({ data }) => {
  const canvasRef    = useRef(null);
  const stripRef     = useRef(null);
  const containerRef = useRef(null);

  const draw = useCallback(() => {
    if (!data || data.length === 0) return;
    const canvas = canvasRef.current;
    const strip  = stripRef.current;
    const cont   = containerRef.current;
    if (!canvas || !strip || !cont) return;

    const W       = cont.clientWidth  || 800;
    const TOTAL_H = cont.clientHeight;
    // BUG #10 FIX: bail out if layout hasn't settled (height ~0); ResizeObserver
    // will call draw() again once the flex container has a real height.
    if (!TOTAL_H || TOTAL_H < 40) return;
    const STRIP_H = 34;                       // taller heat strip — easier to read
    const PRICE_H = TOTAL_H - STRIP_H - 4;     // 4px gap

    const PAD_L = 64, PAD_R = 56, PAD_T = 22, PAD_B = 34; // room for labels both sides
    const chartW = W - PAD_L - PAD_R;
    const chartH = PRICE_H - PAD_T - PAD_B;

    const dpr = window.devicePixelRatio || 1;
    canvas.style.width  = W + "px";
    canvas.style.height = PRICE_H + "px";
    canvas.width  = W * dpr;
    canvas.height = PRICE_H * dpr;
    strip.style.width  = W + "px";
    strip.style.height = STRIP_H + "px";
    strip.width  = W * dpr;
    strip.height = STRIP_H * dpr;

    const pc = canvas.getContext("2d");
    const sc = strip.getContext("2d");
    // BUG #3 FIX: setTransform resets the matrix each draw so the dpr scale
    // never accumulates. (Plain scale(dpr,dpr) compounds on every redraw,
    // shrinking the chart into the top-left corner after the first resize.)
    pc.setTransform(dpr, 0, 0, dpr, 0, 0);
    sc.setTransform(dpr, 0, 0, dpr, 0, 0);

    // ── price range (use close values so line stays centred; pad 8%) ──
    // BUG #4 FIX: reduce instead of Math.max(...spread) — spread overflows
    // the call stack for large series (500+ bars common here).
    let priceMax = -Infinity, priceMin = Infinity;
    for (const d of data) {
      if (d.close > priceMax) priceMax = d.close;
      if (d.close < priceMin) priceMin = d.close;
    }
    priceMax *= 1.04;
    priceMin *= 0.96;
    const priceSpan = priceMax - priceMin || 1;

    const n    = data.length;
    const barW = chartW / n;

    const toX  = (i) => PAD_L + (i + 0.5) * barW;
    const toY  = (p) => PAD_T + chartH * (1 - (p - priceMin) / priceSpan);
    const spanX = (i) => PAD_L + i * barW;

    // ── clear + dark bg ──
    pc.fillStyle = "#0d0f18";
    pc.fillRect(0, 0, W, PRICE_H);
    sc.fillStyle = "#0d0f18";
    sc.fillRect(0, 0, W, STRIP_H);

    // ── faint horizontal grid ──
    pc.save();
    pc.strokeStyle = "#2a2f4566";
    pc.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = PAD_T + (i / 4) * chartH;
      pc.beginPath(); pc.moveTo(PAD_L, y); pc.lineTo(W - PAD_R, y); pc.stroke();
    }
    pc.restore();

    // ── Y-axis price labels (left) — bigger, aligned to the 4 gridlines ──
    pc.save();
    pc.fillStyle = "#9aa6d4";
    pc.font = "12px 'Courier New', monospace";
    pc.textAlign = "right";
    pc.textBaseline = "middle";
    for (let i = 0; i <= 4; i++) {
      const p = priceMax - (i / 4) * priceSpan;   // top→bottom
      const y = PAD_T + (i / 4) * chartH;
      const fmt = p >= 1000 ? p.toFixed(0) : p >= 10 ? p.toFixed(1) : p.toFixed(2);
      pc.fillText(fmt, PAD_L - 8, y);
    }
    pc.restore();

    // ── LAYER 1: full-height colored vertical spans (axvspan) ──
    for (let i = 0; i < n; i++) {
      const d = data[i];
      const isUp = d.close >= d.open;
      // BUG #7 FIX: use spanStrength (streak-based, −3..+3) not candleSignal (−1/0/1)
      const ss = d.bullish3x3 ? 3 : d.bearish3x3 ? -3 : d.bullishEngulf ? 1 : d.bearishEngulf ? -1 : 0;
      pc.fillStyle = spanColor(d.spanStrength ?? 0, ss, isUp);
      pc.fillRect(spanX(i), PAD_T, Math.ceil(barW) + 0.5, chartH);
    }

    // ── LAYER 2: close-price LINE (bold white, on top of spans) ──
    pc.save();
    pc.strokeStyle = "#ffffff";
    pc.lineWidth   = 2.5;
    pc.lineJoin    = "round";
    pc.lineCap     = "round";
    pc.shadowColor = "rgba(0,0,0,0.7)";
    pc.shadowBlur  = 4;
    pc.beginPath();
    for (let i = 0; i < n; i++) {
      const x = toX(i);
      const y = toY(data[i].close);
      if (i === 0) pc.moveTo(x, y);
      else         pc.lineTo(x, y);
    }
    pc.stroke();
    pc.restore();

    // ── Last-price marker: dot + price tag pinned to right edge ──
    {
      const lastClose = data[n - 1].close;
      const ly = toY(lastClose);
      const lx = toX(n - 1);
      // dot
      pc.save();
      pc.fillStyle = "#ffffff";
      pc.beginPath(); pc.arc(lx, ly, 3.5, 0, Math.PI * 2); pc.fill();
      pc.restore();
      // tag in the right gutter
      const tag = lastClose >= 1000 ? lastClose.toFixed(0) : lastClose >= 10 ? lastClose.toFixed(1) : lastClose.toFixed(2);
      pc.save();
      pc.font = "bold 12px 'Courier New', monospace";
      const tw = pc.measureText(tag).width;
      const tagX = W - PAD_R + 4;
      pc.fillStyle = "#1b8f5a";
      pc.beginPath();
      const ry = Math.max(PAD_T + 8, Math.min(PRICE_H - PAD_B - 8, ly));
      if (pc.roundRect) pc.roundRect(tagX, ry - 9, tw + 12, 18, 4);
      else              pc.rect(tagX, ry - 9, tw + 12, 18);
      pc.fill();
      pc.fillStyle = "#ffffff";
      pc.textAlign = "left";
      pc.textBaseline = "middle";
      pc.fillText(tag, tagX + 6, ry);
      pc.restore();
    }

    // ── X-axis date labels ──
    pc.save();
    pc.fillStyle = "#9aa6d4";
    pc.textAlign = "center";
    pc.textBaseline = "alphabetic";
    // On narrow screens show fewer labels and use a shorter date string
    const maxLabels = W < 500 ? 4 : W < 800 ? 5 : 7;
    const labelStep = Math.max(1, Math.floor(n / maxLabels));
    pc.font = W < 500 ? "10px 'Courier New', monospace" : "11px 'Courier New', monospace";
    for (let i = 0; i < n; i += labelStep) {
      const raw = data[i].dateStr || "";
      // On mobile shorten "Jan 5, 21" → "Jan '21" to avoid overlap
      const label = W < 500 ? raw.replace(/\s+\d+,\s+/, " '") : raw;
      pc.fillText(label, toX(i), PRICE_H - 10);
    }
    pc.restore();

    // ── GRADIENT STRIP ──
    // Full-width, one rect per bar (aligned with price spans)
    for (let i = 0; i < n; i++) {
      sc.fillStyle = gradientHeatColor(data[i].gradientScore ?? 0);
      sc.fillRect(PAD_L + i * barW, 0, Math.ceil(barW) + 0.5, STRIP_H);
    }
    // Left gutter — solid dark with a readable label
    sc.fillStyle = "#0d0f18";
    sc.fillRect(0, 0, PAD_L, STRIP_H);
    sc.fillStyle = "#9aa6d4";
    sc.font = "bold 10px 'Courier New', monospace";
    sc.textAlign = "right";
    sc.textBaseline = "middle";
    sc.fillText("MOMENTUM", PAD_L - 8, STRIP_H / 2);
    // Right gutter — solid dark
    sc.fillStyle = "#0d0f18";
    sc.fillRect(W - PAD_R, 0, PAD_R, STRIP_H);

  }, [data]);

  // BUG #2 FIX: a single draw path. The ResizeObserver fires once on mount
  // (initial observe) AND on every resize, so we don't need a separate
  // useEffect(draw) that would double-fire. rAF-debounce coalesces bursts of
  // resize callbacks into one paint per frame.
  useEffect(() => {
    let raf = 0;
    const schedule = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => draw());
    };
    const obs = new ResizeObserver(schedule);
    if (containerRef.current) obs.observe(containerRef.current);
    schedule(); // initial paint (also covers data changes via draw's [data] dep)
    return () => { cancelAnimationFrame(raf); obs.disconnect(); };
  }, [draw]);

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column",
               background: "#0d0f18", gap: 4, borderRadius: 10, overflow: "hidden",
               border: "1px solid #1c2030" }}
    >
      <canvas ref={canvasRef}  style={{ width: "100%", flex: 1, display: "block" }} />
      <canvas ref={stripRef}   style={{ width: "100%", height: 34, display: "block", flexShrink: 0 }} />
    </div>
  );
};

// ─── Main Component ───────────────────────────────────────────────────────────

// ═══════════════════════════════════════════════════════════════
// DOMAIN INTELLIGENCE ENGINE
// Adapted from ChatChunkCompiler — graph topology, cohesion,
// semantic alignment, and singularity synthesis applied to
// cup-and-handle detection domains instead of transcript concepts.
// ═══════════════════════════════════════════════════════════════

/* Fixed domain definitions — Cup & Handle */
const DOMAIN_DEFS_CUP = [
  { id: "cup_shape",    label: "Cup Shape",      desc: "How cleanly the price traces a rounded bottom. Composite of area symmetry and span symmetry.",    field: (d) => d ? (d.areaSymmetry * 0.5 + d.spanSymmetry * 0.5) : 0,         relatedIds: ["gradient", "breakout"] },
  { id: "handle",       label: "Handle",         desc: "Quality of the handle consolidation — depth of retrace and tightness of the drift.",              field: (d) => d ? Math.max(0, 1 - Math.abs(d.handleRetrace - 0.3) / 0.3) : 0, relatedIds: ["pulse", "breakout"] },
  { id: "gradient",     label: "Gradient Arc",   desc: "Smoothness of the price recovery arc from cup bottom toward the rim.",                            field: (d) => d ? d.gradConf : 0,                                             relatedIds: ["cup_shape", "volume"] },
  { id: "pulse",        label: "Pulse / Streak", desc: "Recent candle momentum — consecutive bullish bars in the handle zone.",                           field: (d) => d ? d.pulseBonus : 0,                                           relatedIds: ["handle", "volume"] },
  { id: "breakout",     label: "Breakout Prox",  desc: "How close price is to the pivot / breakout level relative to the cup rim.",                       field: (d) => d ? d.breakoutProx : 0,                                         relatedIds: ["cup_shape", "handle"] },
  { id: "volume",       label: "Volume",         desc: "Volume confirmation — expansion on up days, contraction on handle drift.",                        field: (d) => d ? Math.min(1, d.pulseBonus * 1.2) : 0,                        relatedIds: ["gradient", "pulse"] },
  { id: "sector",       label: "Sector",         desc: "Sector-level momentum from the pulse heatmap — tailwind or headwind for this pattern.",           field: null,                                                                   relatedIds: ["volume", "pulse"] },
];

/* Fixed domain definitions — Reverse Head & Shoulders */
const DOMAIN_DEFS_RHS = [
  { id: "shoulder_sym", label: "Shoulder Sym",   desc: "How symmetrical the two shoulders are in height and distance from the head.",  field: (d) => d?.shoulderSym   ?? d?.radar?.rimSymmetry  ?? 0, relatedIds: ["neckline", "breakout"] },
  { id: "head_depth",   label: "Head Depth",     desc: "Depth of the head trough — ideal 10-30%.",                                     field: (d) => d?.radar?.depthScore   ?? 0, relatedIds: ["shape_fit", "volume"] },
  { id: "shape_fit",    label: "Shape Fit",      desc: "How well the W-shape conforms to textbook reverse H&S.",                       field: (d) => d?.areaFit ?? d?.radar?.areaSymmetry ?? d?.radar?.handleQuality ?? 0, relatedIds: ["head_depth", "neckline"] },
  { id: "neckline",     label: "Neckline",       desc: "Neckline quality — near-horizontal scores highest.",                           field: (d) => d?.necklineScore ?? d?.radar?.spanSymmetry  ?? 0, relatedIds: ["shoulder_sym", "breakout"] },
  { id: "breakout",     label: "Breakout Prox",  desc: "Closeness to neckline breakout trigger.",                                      field: (d) => d?.breakoutProx ?? d?.radar?.breakoutProx  ?? 0, relatedIds: ["neckline", "volume"] },
  { id: "volume",       label: "Vol Surge",      desc: "Volume expansion confirming the neckline break.",                              field: (d) => d?.volSurge ?? d?.radar?.gradConf      ?? 0, relatedIds: ["head_depth", "pulse"] },
  { id: "pulse",        label: "Momentum",       desc: "Recent candle momentum approaching the neckline.",                             field: (d) => d?.recentMomentum ?? d?.radar?.recentMomentum ?? 0, relatedIds: ["volume", "breakout"] },
  { id: "sector",       label: "Sector",         desc: "Sector-level momentum — tailwind or headwind for this reversal.",              field: null,                                relatedIds: ["volume", "pulse"] },
];

/* Domain definitions — Head & Shoulders (bearish) */
const DOMAIN_DEFS_HS = [
  { id: "head_dom",     label: "Head Dominance", desc: "How much higher the head is above both shoulders — the primary H&S quality signal.",   field: (d) => d?.radar?.depthScore ?? 0,                                        relatedIds: ["neckline", "volume"] },
  { id: "neckline",     label: "Neckline",       desc: "Neckline quality through the two inner troughs — near-horizontal scores highest.",      field: (d) => d?.necklineScore ?? d?.radar?.spanSymmetry ?? 0,                  relatedIds: ["head_dom", "breakdown"] },
  { id: "shape_fit",    label: "Shape Fit",      desc: "How closely price hugs the textbook ∩-Λ-∩ outline.",                                   field: (d) => d?.areaFit ?? d?.radar?.areaSymmetry ?? 0,                        relatedIds: ["head_dom", "volume"] },
  { id: "shoulder_sym", label: "Shoulder Sym",   desc: "Similarity of shoulders above the neckline. Real H&S can have asymmetric shoulders.",  field: (d) => d?.shoulderSym ?? d?.radar?.rimSymmetry ?? 0,                     relatedIds: ["neckline", "shape_fit"] },
  { id: "breakdown",    label: "Breakdown Prox", desc: "How close price is to breaking the neckline — or has it already broken?",              field: (d) => d?.breakoutProx ?? d?.radar?.breakoutProx ?? 0,                   relatedIds: ["neckline", "volume"] },
  { id: "volume",       label: "Volume Profile", desc: "Vol declining left→head→right shoulder, then surging on breakdown. Classic H&S tell.", field: (d) => d?.volConf ?? d?.radar?.volumeConf ?? 0,                          relatedIds: ["head_dom", "pulse"] },
  { id: "pulse",        label: "Momentum",       desc: "Recent bearish candle momentum — confirms distribution phase.",                        field: (d) => d ? Math.max(0, 1 - (d.recentMomentum ?? 0.5) * 2) : 0,         relatedIds: ["volume", "breakdown"] },
  { id: "sector",       label: "Sector",         desc: "Sector-level momentum — bearish sector tailwind strengthens the setup.",               field: null,                                                                     relatedIds: ["volume", "pulse"] },
];

/* Domain definitions — Rounded Top (bearish) */
const DOMAIN_DEFS_RT = [
  { id: "arc_shape",    label: "Arc Shape",      desc: "How cleanly price traces the inverted rounded top (∩). Area and span symmetry of the arc.", field: (d) => d ? ((d.areaSymmetry ?? 0) * 0.5 + (d.spanSymmetry ?? 0) * 0.5) : 0, relatedIds: ["gradient", "breakdown"] },
  { id: "dist_shelf",   label: "Dist. Shelf",    desc: "Quality of the distribution shelf after the right rim — the bearish 'handle'.",            field: (d) => d ? Math.max(0, 1 - Math.abs((d.handleRetrace ?? 0.25) - 0.25) / 0.25) : 0, relatedIds: ["pulse", "breakdown"] },
  { id: "gradient",     label: "Gradient Arc",   desc: "Smoothness of the descent arc — rising then falling, like an inverted cup.",               field: (d) => d ? (d.gradConf ?? 0) : 0,                                           relatedIds: ["arc_shape", "volume"] },
  { id: "pulse",        label: "Bear Streak",    desc: "Recent bearish candle momentum in the distribution shelf zone.",                            field: (d) => d ? (d.pulseBonus ?? 0) : 0,                                         relatedIds: ["dist_shelf", "volume"] },
  { id: "breakdown",    label: "Breakdown Prox", desc: "How close price is to breaking below the right rim level.",                                 field: (d) => d ? (d.breakoutProx ?? 0) : 0,                                       relatedIds: ["arc_shape", "dist_shelf"] },
  { id: "volume",       label: "Volume",         desc: "Volume confirmation — high at rims, low at arc top (same bowl shape as cup, inverted).",    field: (d) => d ? Math.min(1, (d.volConf ?? 0)) : 0,                               relatedIds: ["gradient", "pulse"] },
  { id: "sector",       label: "Sector",         desc: "Sector-level momentum — bearish sector tailwind strengthens the breakdown signal.",         field: null,                                                                        relatedIds: ["volume", "pulse"] },
];

/* Legacy alias */
const DOMAIN_DEFS = DOMAIN_DEFS_CUP;

/* Compute a recency multiplier [0.25 – 1.0] based on how far the right rim
   is from the end of the price series.
   - 0 bars ago (fresh breakout) → 1.0
   - ~half the series away       → ~0.5
   - at series start (very old)  → 0.25 (floor)
   barsFromEnd: number of bars between rightRim and the last bar (0 = current).
   totalBars:   full length of the ohlcv series for that ticker. */
function computeRecencyMultiplier(barsFromEnd, totalBars) {
  if (barsFromEnd == null || totalBars == null || totalBars <= 0) return 1.0;
  // fraction of the series that has elapsed AFTER the pattern's right rim
  const ageFraction = Math.min(1, barsFromEnd / totalBars);
  // Exponential decay: fresh = 1.0, halfway through = ~0.6, full 5-year ago = 0.25
  return Math.max(0.25, Math.pow(1 - ageFraction, 1.5));
}

/* Build graph nodes from a detection object + optional sector pulse score.
   recencyMult (0.25–1.0): older patterns get proportionally lower node scores.
   Routes to the correct domain definition set based on setupType. */
function buildDomainNodes(detection, sectorPulse, recencyMult = 1.0) {
  const type = detection?.setupType;
  const defs = type === "rhs" ? DOMAIN_DEFS_RHS
    : type === "hs" ? DOMAIN_DEFS_HS
    : type === "rt" ? DOMAIN_DEFS_RT
    : DOMAIN_DEFS_CUP;
  return defs.map(def => {
    const rawScore = def.field
      ? Math.max(0, Math.min(1, def.field(detection)))
      : (sectorPulse != null ? Math.max(0, Math.min(1, 0.5 + sectorPulse)) : 0.5);
    // Pulse / Sector nodes represent CURRENT momentum — don't decay them
    const isCurrentMomentum = def.id === "pulse" || def.id === "sector";
    const score = isCurrentMomentum ? rawScore : Math.max(0, Math.min(1, rawScore * recencyMult));
    return {
      id: def.id,
      label: def.label,
      summary: def.desc,
      score,
      relatedIds: def.relatedIds,
      tokenCount: Math.round(score * 80 + 10),
    };
  });
}

/* Graph topology — degree centrality, clustering, cohesion, edge heat */
function analyzeDomainGraph(nodes) {
  const degree = {};
  nodes.forEach(n => { degree[n.id] = 0; });
  nodes.forEach(n => {
    n.relatedIds.forEach(rid => {
      if (rid in degree) {
        degree[n.id] = (degree[n.id] || 0) + 1;
        degree[rid]  = (degree[rid]  || 0) + 1;
      }
    });
  });

  // Union-Find
  const parent = {};
  nodes.forEach(n => { parent[n.id] = n.id; });
  function find(x) { return parent[x] === x ? x : (parent[x] = find(parent[x])); }
  function union(a, b) { parent[find(a)] = find(b); }
  nodes.forEach(n => { n.relatedIds.forEach(rid => { if (rid in parent) union(n.id, rid); }); });

  const clusterRoots = [...new Set(nodes.map(n => find(n.id)))];
  const clusterIndex = {};
  clusterRoots.forEach((root, i) => { clusterIndex[root] = i; });
  const nodeCluster = {};
  nodes.forEach(n => { nodeCluster[n.id] = clusterIndex[find(n.id)]; });

  const clusters = {};
  nodes.forEach(n => {
    const root = find(n.id);
    if (!clusters[root]) clusters[root] = [];
    clusters[root].push(n.id);
  });

  const edgeList = [];
  const seen = new Set();
  nodes.forEach(n => {
    n.relatedIds.forEach(rid => {
      if (!(rid in degree)) return;
      const key = [n.id, rid].sort().join("--");
      if (seen.has(key)) return;
      seen.add(key);
      const strength = (degree[n.id] || 0) + (degree[rid] || 0);
      edgeList.push({ from: n.id, to: rid, strength, isBridge: nodeCluster[n.id] !== nodeCluster[rid] });
    });
  });
  const maxStrength = Math.max(1, ...edgeList.map(e => e.strength));
  edgeList.forEach(e => { e.heat = e.strength / maxStrength; });

  // Adjacency
  const adj = {};
  nodes.forEach(n => { adj[n.id] = new Set(); });
  nodes.forEach(n => {
    n.relatedIds.forEach(rid => {
      if (adj[rid] !== undefined) { adj[n.id].add(rid); adj[rid].add(n.id); }
    });
  });

  // Cohesion: score-weighted local clustering coefficient
  // For domain graphs we blend structural LCC with how well the node's score
  // aligns with its neighbors' scores (a "momentum agreement" proxy)
  const cohesionScore = {};
  nodes.forEach(n => {
    const neighbors = [...adj[n.id]];
    if (neighbors.length === 0) { cohesionScore[n.id] = 0; return; }
    const triPossible = neighbors.length * (neighbors.length - 1) / 2;
    let triActual = 0;
    for (let i = 0; i < neighbors.length; i++)
      for (let j = i + 1; j < neighbors.length; j++)
        if (adj[neighbors[i]] && adj[neighbors[i]].has(neighbors[j])) triActual++;
    const lcc = triPossible > 0 ? triActual / triPossible : 0;
    // Score agreement with neighbors
    const neighborScores = neighbors.map(rid => nodes.find(x => x.id === rid)?.score ?? 0.5);
    const avgNeighbor = neighborScores.reduce((a, b) => a + b, 0) / neighborScores.length;
    const agreement = 1 - Math.abs(n.score - avgNeighbor);
    cohesionScore[n.id] = Math.round((0.5 * lcc + 0.5 * agreement) * 100) / 100;
  });

  return { degree, nodeCluster, clusters, edgeList, cohesionScore, adj };
}

/* Quadrant diagnosis — same logic as compiler but reframed for trading domains */
function domainQuadrant(structural, score) {
  const hiS = structural >= 0.5, hiD = score >= 0.5;
  if (hiS && hiD)  return { label: "Confirmed",    color: "#34d399", desc: "Well-connected to peer domains and scoring high. This dimension is a genuine setup contributor." };
  if (hiS && !hiD) return { label: "Gap Risk",     color: "#fbbf24", desc: "Structurally linked but underperforming. Neighbor domains are stronger — this is a weak link in the setup." };
  if (!hiS && hiD) return { label: "Isolated Win", color: "#7c9fff", desc: "Scoring high but loosely connected. This dimension looks good in isolation; confirm it's not a coincidence." };
  return              { label: "Drag",           color: "#f87171", desc: "Peripheral and underperforming. Reduces conviction in the overall setup." };
}

function dgHeatColor(t) {
  const stops = [[0,[74,85,128]],[0.35,[100,80,180]],[0.65,[251,191,36]],[1,[249,115,22]]];
  let lo = stops[0], hi = stops[stops.length-1];
  for (let i = 0; i < stops.length-1; i++) { if (t >= stops[i][0] && t <= stops[i+1][0]) { lo = stops[i]; hi = stops[i+1]; break; } }
  const f = (t - lo[0]) / (hi[0] - lo[0] || 1);
  const r = Math.round(lo[1][0] + (hi[1][0]-lo[1][0])*f);
  const g = Math.round(lo[1][1] + (hi[1][1]-lo[1][1])*f);
  const b = Math.round(lo[1][2] + (hi[1][2]-lo[1][2])*f);
  return `rgb(${r},${g},${b})`;
}

function dgConfColor(t) {
  if (t >= 0.65) return "#34d399";
  if (t >= 0.35) return "#fbbf24";
  return "#f87171";
}

const DG_CLUSTER_COLORS = ["#7c9fff","#34d399","#f472b6","#fbbf24","#a78bfa","#38bdf8"];
function dgClusterColor(idx) { return DG_CLUSTER_COLORS[idx % DG_CLUSTER_COLORS.length]; }

/* ── Force-directed SVG Domain Graph ── */
const SVGDomainGraph = memo(function SVGDomainGraph({ nodes, selectedId, onSelectNode }) {
  const posRef = useRef({});
  const velRef = useRef({});
  const frameRef = useRef();
  const [, setTick] = useState(0);

  useEffect(() => {
    posRef.current = {};
    velRef.current = {};
    nodes.forEach((n, i) => {
      const angle = (i / nodes.length) * 2 * Math.PI;
      posRef.current[n.id] = { x: 340 + 140 * Math.cos(angle), y: 230 + 110 * Math.sin(angle) };
      velRef.current[n.id] = { vx: 0, vy: 0 };
    });
  }, [nodes.map(n => n.id).join(",")]);

  useEffect(() => {
    if (!nodes.length) return;
    let running = true;
    const W = 680, H = 460;
    function step() {
      if (!running) return;
      const ids = nodes.map(n => n.id);
      const pos = posRef.current, vel = velRef.current;
      // Repulsion
      for (let i = 0; i < ids.length; i++) {
        for (let j = i+1; j < ids.length; j++) {
          const a = pos[ids[i]], b = pos[ids[j]];
          if (!a || !b) continue;
          const dx = a.x-b.x, dy = a.y-b.y;
          const dist = Math.sqrt(dx*dx+dy*dy) || 1;
          const force = 2800 / (dist*dist);
          const fx = (dx/dist)*force, fy = (dy/dist)*force;
          vel[ids[i]].vx += fx; vel[ids[i]].vy += fy;
          vel[ids[j]].vx -= fx; vel[ids[j]].vy -= fy;
        }
      }
      // Attraction along edges
      for (const n of nodes) {
        for (const rid of n.relatedIds) {
          const a = pos[n.id], b = pos[rid];
          if (!a || !b) continue;
          const dx = b.x-a.x, dy = b.y-a.y;
          const dist = Math.sqrt(dx*dx+dy*dy) || 1;
          const force = (dist - 90) * 0.018;
          const fx = (dx/dist)*force, fy = (dy/dist)*force;
          vel[n.id].vx += fx; vel[n.id].vy += fy;
          if (vel[rid]) { vel[rid].vx -= fx; vel[rid].vy -= fy; }
        }
      }
      // Center gravity + damping
      for (const id of ids) {
        const p = pos[id], v = vel[id];
        if (!p || !v) continue;
        v.vx += (W/2 - p.x) * 0.002; v.vy += (H/2 - p.y) * 0.002;
        v.vx *= 0.86; v.vy *= 0.86;
        p.x = Math.max(55, Math.min(W-55, p.x + v.vx));
        p.y = Math.max(35, Math.min(H-35, p.y + v.vy));
      }
      setTick(t => t+1);
      frameRef.current = requestAnimationFrame(step);
    }
    frameRef.current = requestAnimationFrame(step);
    const stop = setTimeout(() => { running = false; }, 3000);
    return () => { running = false; cancelAnimationFrame(frameRef.current); clearTimeout(stop); };
  }, [nodes.map(n => n.id).join(",")]);

  const { edgeList, nodeCluster, cohesionScore } = analyzeDomainGraph(nodes);
  const pos = posRef.current;

  return (
    <svg viewBox="0 0 680 460" style={{ width: "100%", height: "100%" }} preserveAspectRatio="xMidYMid meet">
      {/* Edges */}
      {edgeList.map(e => {
        const a = pos[e.from], b = pos[e.to];
        if (!a || !b) return null;
        const col = dgHeatColor(e.heat);
        const sw = 1.2 + e.heat * 2.4;
        return (
          <line key={e.from+"-"+e.to}
            x1={a.x} y1={a.y} x2={b.x} y2={b.y}
            stroke={e.isBridge ? "#6b7280" : col}
            strokeWidth={sw}
            strokeDasharray={e.isBridge ? "5 4" : undefined}
            opacity={0.5 + e.heat * 0.35}
          />
        );
      })}

      {/* Nodes */}
      {nodes.map(n => {
        const p = pos[n.id];
        if (!p) return null;
        const isSelected = n.id === selectedId;
        const cColor = dgClusterColor(nodeCluster[n.id] ?? 0);
        const conf = cohesionScore[n.id] ?? 0.5;
        const confCol = dgConfColor(conf);
        const r = 20 + n.score * 14; // bigger = higher domain score
        const ringR = r + 5;
        const circ = 2 * Math.PI * ringR;
        const dash = conf * circ;
        const scoreColor = n.score >= 0.65 ? "#34d399" : n.score >= 0.35 ? "#fbbf24" : "#f87171";
        return (
          <g key={n.id}
            transform={`translate(${p.x},${p.y})`}
            onClick={() => onSelectNode(n)}
            style={{ cursor: "pointer" }}
          >
            {/* Score fill ring */}
            <circle r={r}
              fill={isSelected ? cColor : "rgba(26,29,39,0.92)"}
              stroke={cColor}
              strokeWidth={isSelected ? 2.5 : 2}
              fillOpacity={isSelected ? 0.9 : 1}
            />
            {/* Score tint — inner fill proportional to domain score */}
            {!isSelected && (
              <circle r={r - 2}
                fill={scoreColor}
                fillOpacity={0.06 + n.score * 0.18}
                stroke="none"
              />
            )}
            {/* Cohesion confidence arc */}
            <circle r={ringR} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth={2.5} />
            <circle r={ringR} fill="none" stroke={confCol} strokeWidth={2.5}
              strokeDasharray={`${dash} ${circ - dash}`}
              strokeDashoffset={circ * 0.25}
              opacity={0.8}
              strokeLinecap="round"
            />
            {/* Score badge top-right */}
            <text
              x={r * 0.6} y={-(r * 0.6)}
              textAnchor="middle" dominantBaseline="middle"
              style={{ fill: scoreColor, fontFamily: "monospace", fontSize: 10, fontWeight: 700, pointerEvents: "none" }}
            >{Math.round(n.score * 100)}</text>
            {/* Label */}
            <text textAnchor="middle" dy="0.35em"
              style={{
                fill: isSelected ? "#0f1117" : "#e8eaf6",
                fontSize: Math.max(10, 13 - n.label.length * 0.12),
                fontWeight: 700, pointerEvents: "none",
                fontFamily: "'Inter', system-ui, sans-serif",
              }}
            >{n.label.length > 12 ? n.label.slice(0, 11) + "…" : n.label}</text>
          </g>
        );
      })}

    </svg>
  );
});

/* ── Domain Node Detail Drawer ── */
function DomainDrawer({ node, allNodes, onClose, isMobile }) {
  useEffect(() => {
    const hk = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", hk);
    return () => window.removeEventListener("keydown", hk);
  }, [onClose]);

  if (!node) return null;
  const { cohesionScore } = analyzeDomainGraph(allNodes);
  const conf = cohesionScore[node.id] ?? 0.5;
  const confCol = dgConfColor(conf);
  const confLbl = conf >= 0.65 ? "core" : conf >= 0.35 ? "mid" : "fringe";
  const quad = domainQuadrant(conf, node.score);
  const related = allNodes.filter(n => node.relatedIds.includes(n.id));

  const scoreColor = node.score >= 0.65 ? "#34d399" : node.score >= 0.35 ? "#fbbf24" : "#f87171";
  const scoreLbl = node.score >= 0.65 ? "Strong" : node.score >= 0.35 ? "Moderate" : "Weak";

  return (
    <div style={isMobile ? {
      position: "fixed", inset: 0, zIndex: 400,
      background: "#1a1d27", display: "flex", flexDirection: "column",
      overflowY: "auto", width: "100%",
    } : {
      width: 300, flexShrink: 0, borderLeft: "1px solid #2a2f45",
      background: "#1a1d27", display: "flex", flexDirection: "column",
      overflowY: "auto",
    }}>
      <div style={{ display: "flex", alignItems: "center", padding: "16px 18px 12px", borderBottom: "1px solid #2a2f45", gap: 10, flexShrink: 0 }}>
        <span style={{ flex: 1, fontSize: 17, fontWeight: 800, color: "#e8eaf6", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{node.label}</span>
        <button onClick={onClose} style={{ background: "none", border: "none", color: "#7986cb", cursor: "pointer", fontSize: 22, lineHeight: 1, padding: "2px 4px" }}>×</button>
      </div>
      <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 16 }}>
        {/* Description */}
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", color: "#7986cb", marginBottom: 7 }}>What this measures</div>
          <p style={{ fontSize: 13, lineHeight: 1.6, color: "#e8eaf6" }}>{node.summary}</p>
        </div>
        {/* Domain score + cohesion side by side */}
        <div style={{ display: "flex", gap: 10 }}>
          <div style={{ flex: 1, background: "#0f1117", border: "1px solid #2a2f45", borderRadius: 7, padding: "11px 12px" }}>
            <div style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".07em", color: "#7986cb", marginBottom: 7 }}>Domain Score</div>
            <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
              <svg width={42} height={42} viewBox="0 0 42 42">
                <circle cx={21} cy={21} r={15} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth={4} />
                <circle cx={21} cy={21} r={15} fill="none" stroke={scoreColor} strokeWidth={4}
                  strokeDasharray={`${node.score * 94.25} 94.25`} strokeDashoffset={23.56} strokeLinecap="round" />
                <text x={21} y={21} textAnchor="middle" dominantBaseline="middle"
                  style={{ fill: scoreColor, fontFamily: "monospace", fontSize: 12, fontWeight: 700 }}>{Math.round(node.score * 100)}</text>
              </svg>
              <div>
                <div style={{ fontWeight: 700, fontSize: 14, color: scoreColor }}>{scoreLbl}</div>
                <div style={{ fontSize: 11, color: "#7986cb", marginTop: 2 }}>0–100 scale</div>
              </div>
            </div>
          </div>
          <div style={{ flex: 1, background: "#0f1117", border: "1px solid #2a2f45", borderRadius: 7, padding: "11px 12px" }}>
            <div style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".07em", color: "#7986cb", marginBottom: 7 }}>Cohesion</div>
            <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
              <svg width={42} height={42} viewBox="0 0 42 42">
                <circle cx={21} cy={21} r={15} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth={4} />
                <circle cx={21} cy={21} r={15} fill="none" stroke={confCol} strokeWidth={4}
                  strokeDasharray={`${conf * 94.25} 94.25`} strokeDashoffset={23.56} strokeLinecap="round" />
                <text x={21} y={21} textAnchor="middle" dominantBaseline="middle"
                  style={{ fill: confCol, fontFamily: "monospace", fontSize: 12, fontWeight: 700 }}>{Math.round(conf * 100)}</text>
              </svg>
              <div>
                <div style={{ fontWeight: 700, fontSize: 14, color: confCol }}>{confLbl}</div>
                <div style={{ fontSize: 11, color: "#7986cb", marginTop: 2 }}>
                  {confLbl === "core" && "Strongly wired"}
                  {confLbl === "mid" && "Partial linkage"}
                  {confLbl === "fringe" && "Weakly connected"}
                </div>
              </div>
            </div>
          </div>
        </div>
        {/* Quadrant diagnosis */}
        <div style={{ background: "#0f1117", border: `1px solid ${quad.color}44`, borderRadius: 7, padding: "11px 13px" }}>
          <div style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".07em", color: "#7986cb", marginBottom: 5 }}>Diagnosis</div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 9, height: 9, borderRadius: "50%", background: quad.color, flexShrink: 0, boxShadow: `0 0 6px ${quad.color}` }} />
            <span style={{ fontWeight: 700, fontSize: 14, color: quad.color }}>{quad.label}</span>
          </div>
          <p style={{ fontSize: 12, color: "#9aa6d4", marginTop: 5, lineHeight: 1.5 }}>{quad.desc}</p>
        </div>
        {/* Related domains */}
        {related.length > 0 && (
          <div>
            <div style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", color: "#7986cb", marginBottom: 7 }}>Connected Domains</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
              {related.map(r => (
                <span key={r.id} style={{
                  display: "inline-block", padding: "4px 11px", borderRadius: 20,
                  fontSize: 12, fontWeight: 500,
                  background: "rgba(108,143,255,0.12)", color: "#7c9fff",
                  border: "1px solid rgba(108,143,255,0.25)",
                }}>{r.label}</span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}



// ═══════════════════════════════════════════════════════════════
// END DOMAIN INTELLIGENCE ENGINE
// ═══════════════════════════════════════════════════════════════

export default function App() {
  const [rawData, setRawData] = useState(null);
  const [allScores, setAllScores] = useState([]);
  const [scores, setScores] = useState([]);
  const [selectedTicker, setSelectedTicker] = useState(null);
  const [tolerance, setTolerance] = useState({
    cupDepth: [0.12, 0.50],
    handleRetrace: [0.10, 0.50],
    smoothing: 0.08,
    minBars: MIN_BARS,
    activeSetup: "cup",   // OPTION A: scan detects this setup only ("cup" | "rhs")
  });
  const [scanStatus, setScanStatus] = useState("idle");
  const [scanError, setScanError] = useState(null);
  const [scanProgress, setScanProgress] = useState(0);
  const [parseWarnings, setParseWarnings] = useState([]);
  const [showWarnings, setShowWarnings] = useState(false);




  const [activeTab, setActiveTab] = useState("leaderboard");
  const [chartSubTab, setChartSubTab] = useState("pattern");
  const [chartSetup, setChartSetup] = useState(null); // null = follow active setup; "cup"|"rhs" = chart override
  const [scoreFilter, setScoreFilter] = useState(0);
  const [searchQuery, setSearchQuery] = useState("");
  const [recencyFilter, setRecencyFilter] = useState("all");
  const [windowMode, setWindowMode] = useState("auto");
  const [hmWindow, setHmWindow] = useState(60);   // heatmap lookback bars: 30 | 60 | 90
  const [hmSearch, setHmSearch] = useState("");    // heatmap ticker search
  const [hmHover, setHmHover] = useState(null);    // hovered ticker key
  const [elapsed, setElapsed] = useState(0);

  // ── Sector classification state ──
  const [sectorMap, setSectorMap] = useState({});         // ticker → sector string
  const [hmSectorFilter, setHmSectorFilter] = useState("All"); // "All" | sector name
  const sectorCacheRef = useRef({});                      // persists across re-renders

  const [prevScores, setPrevScores] = useState({});      // ticker → prev rank for delta
  const [leaderSectorFilter, setLeaderSectorFilter] = useState("All"); // sector filter on leaderboard
  const [hmSort, setHmSort] = useState("score");          // "score" | "alpha" | "streak"
  const [pinnedTicker, setPinnedTicker] = useState(null); // pinned heatmap tooltip
  const [darkMode, setDarkMode] = useState(true);         // dark/light toggle

  // ── Domain Intelligence state ──
  const [selectedDomainNode, setSelectedDomainNode] = useState(null);

  // ── Mobile responsiveness ──
  const [isMobile, setIsMobile] = useState(() => typeof window !== "undefined" && window.innerWidth <= 768);
  const [showSettings, setShowSettings] = useState(false); // mobile settings drawer (upload + tolerances)
  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth <= 768);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const cancelRef = useRef(false);

  const scanStartRef = useRef(null);
  const listRef = useRef(null);

  // ── Reset domain state when ticker changes ──
  useEffect(() => {
    setSelectedDomainNode(null);
  }, [selectedTicker]);

  // ── Auto-load bundled default CSV on startup ──
  useEffect(() => {
    const loadDefault = async () => {
      try {
        setScanStatus("scanning");
        setScanError(null);
        setScanProgress(0);
        setScores([]);
        setAllScores([]);
        setSelectedTicker(null);
        cancelRef.current = false;

        const res = await fetch("/ohlcv_data_fixed.csv");
        if (!res.ok) throw new Error("Default data not found");

        const text = await res.text();
        const blob = new Blob([text], { type: "text/csv" });
        const file = new File([blob], "ohlcv_data_fixed.csv", { type: "text/csv" });

        const parsed = await parseCSV(file);
        setRawData(parsed.map);
        setParseWarnings(parsed.warnings);

        if (parsed.map.size === 0) {
          setScanStatus("error");
          setScanError("Default data loaded but no valid tickers found.");
          return;
        }

        const results = await runScan(parsed.map, tolerance, setScanProgress, cancelRef, windowMode);
        if (!cancelRef.current) {
          setAllScores(results);
          setScores(results);
          setScanStatus("done");
          setScanProgress(100);
          if (results.length > 0) {
            setSelectedTicker(results[0].ticker);
            setActiveTab("chart");
          }
        }
      } catch (err) {
        // Silently fall back to manual upload if default data fails
        setScanStatus("idle");
        setScanError(null);
      }
    };
    loadDefault();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // run once on mount

  // ── Elapsed timer ──
  useEffect(() => {
    if (scanStatus !== "scanning") return;
    scanStartRef.current = Date.now();
    const id = setInterval(() => setElapsed(Math.round((Date.now() - scanStartRef.current) / 1000)), 500);
    return () => clearInterval(id);
  }, [scanStatus]);


  // ── Sector classification: static lookup + AI batch fallback ──
  useEffect(() => {
    if (!rawData) return;
    const tickers = [...rawData.keys()];
    const resolved = {};
    const unknown = [];

    // Step 1: static lookup
    for (const t of tickers) {
      const cached = sectorCacheRef.current[t];
      if (cached) { resolved[t] = cached; continue; }
      const s = lookupSector(t);
      if (s) { resolved[t] = s; sectorCacheRef.current[t] = s; }
      else unknown.push(t);
    }

    // Update immediately with what we have
    setSectorMap(prev => ({ ...prev, ...resolved }));

    if (unknown.length === 0) return;
    // Unknown tickers default to ETF/Other
    const fallback = {};
    unknown.forEach(t => { fallback[t] = "ETF/Other"; sectorCacheRef.current[t] = "ETF/Other"; });
    setSectorMap(prev => ({ ...prev, ...fallback }));
  }, [rawData]);

  // The detected setup for the selected ticker (cup or flag per tolerance.activeSetup).
  // Declared BEFORE chartData because chartData's memo reads it on first render.
  // Uses the chart's setup choice (override) or the active setup, pulling the
  // already-computed detection from the stored scan row when available.
  const selectedDetection = useMemo(() => {
    if (!selectedTicker || !rawData) return null;
    const wantSetup = chartSetup || tolerance.activeSetup || "cup";
    const row = scores.find(s => s.ticker === selectedTicker);
    if (row) {
      const stored = wantSetup === "rhs" ? row.rhs : wantSetup === "hs" ? row.hs : wantSetup === "rt" ? row.rt : row.cup;
      if (stored && stored.detection) return stored.detection;
    }
    // Fallback: detect live for the wanted setup
    const tolForSetup = { ...tolerance, activeSetup: wantSetup };
    return detectBestWindow(rawData.get(selectedTicker) || [], tolForSetup, windowMode, wantSetup).detection;
  }, [selectedTicker, rawData, tolerance, windowMode, chartSetup, scores]);

  // When the selected ticker changes, drop any chart-local setup override so
  // the chart defaults to the leaderboard's active setup for the new ticker.
  useEffect(() => { setChartSetup(null); }, [selectedTicker]);

  // ── Derived chart data ──
  const chartData = useMemo(() => {
    if (!selectedTicker || !rawData) return [];
    const rows = rawData.get(selectedTicker) || [];
    const result = selectedDetection; // honor the chart's setup choice (cup/rhs)
    const ghostMap = new Map((result?.ghostCurve || []).map(g => [g.idx, g.ghost]));
    // Reverse-H&S / H&S neckline (only present when the detected setup is rhs or hs)
    const isRHS = result?.setupType === "rhs" || result?.setupType === "hs";
    const necklineMap = new Map();
    const rhsShapeMap = new Map();
    if (isRHS) {
      for (const g of (result.ghostCurve || [])) {
        if (g.ghostNeck != null)  necklineMap.set(g.idx, g.ghostNeck);
        if (g.ghostShape != null) rhsShapeMap.set(g.idx, g.ghostShape);
      }
    }

    // BUG #8 FIX: O(n) momentum pass — was O(n²) via rows.slice(0,i+1) per bar
    const MOM_WIN = 10;
    const candleSignals = rows.map(r => classifyCandle(r));
    const momentum = new Array(rows.length).fill(0.5);
    {
      // Sliding window: maintain running bull/bear counts
      let bull = 0, bear = 0;
      for (let i = 0; i < rows.length; i++) {
        const s = candleSignals[i];
        if (s > 0) bull++; else if (s < 0) bear++;
        if (i >= MOM_WIN) {
          const old = candleSignals[i - MOM_WIN];
          if (old > 0) bull--; else if (old < 0) bear--;
        }
        const winLen = Math.min(i + 1, MOM_WIN);
        momentum[i] = (bull - bear + winLen) / (2 * winLen); // 0..1
      }
    }

    // Per-bar streak
    const streaks = [];
    let streak = 0;
    for (const s of candleSignals) {
      if (s === 1)       streak = streak > 0 ? streak + 1 : 1;
      else if (s === -1) streak = streak < 0 ? streak - 1 : -1;
      else               streak = 0;
      streaks.push(streak);
    }

    // BUG #7 FIX: spanStrength uses streak (−N..+N) not raw candleSignal (−1/0/1).
    // Clamp streak to ±3 buckets matching the Python trend_code levels.
    const spanStrength = streaks.map(sk => Math.max(-3, Math.min(3, sk)));

    // Gradient score per bar: O(n) rolling 20-bar window
    // BUG #8 FIX continued: single-pass using running bull/bear counters
    const GRAD_WIN = 20;
    const gradientScores = new Array(rows.length).fill(0);
    {
      let bull = 0, bear = 0;
      for (let i = 0; i < rows.length; i++) {
        const s = candleSignals[i];
        if (s > 0) bull++; else if (s < 0) bear++;
        if (i >= GRAD_WIN) {
          const old = candleSignals[i - GRAD_WIN];
          if (old > 0) bull--; else if (old < 0) bear--;
        }
        // Only score once a full window is available so early bars don't skew the strip
        if (i >= GRAD_WIN - 1) {
          gradientScores[i] = Math.max(-1, Math.min(1, (bull - bear) / (GRAD_WIN * 0.5)));
        }
      }
    }

    // Engulf + 3x3 markers — full-series scan
    // BUG #4 FIX: replace Math.max(...arr.map()) spread with reduce (no stack overflow)
    const N = 3;
    const bullishEngulf = new Uint8Array(rows.length); // typed arrays — faster + no GC
    const bearishEngulf = new Uint8Array(rows.length);
    const bullish3x3    = new Uint8Array(rows.length);
    const bearish3x3    = new Uint8Array(rows.length);

    // 3-bar engulfing
    for (let i = 2 * N; i < rows.length; i++) {
      const first  = rows.slice(i - 2 * N, i - N);
      const second = rows.slice(i - N, i);
      const fOpen = first[0].open,  fClose = first[N - 1].close;
      const sOpen = second[0].open, sClose = second[N - 1].close;
      // BUG #4 FIX: reduce instead of spread
      const fHigh = first.reduce((m, r)  => r.high > m ? r.high : m, -Infinity);
      const fLow  = first.reduce((m, r)  => r.low  < m ? r.low  : m, Infinity);
      const fBody = Math.abs(fClose - fOpen);
      const sBody = Math.abs(sClose - sOpen);
      const fRange = fHigh - fLow;
      const sHigh = second.reduce((m, r) => r.high > m ? r.high : m, -Infinity);
      const sLow  = second.reduce((m, r) => r.low  < m ? r.low  : m, Infinity);
      const sRange = sHigh - sLow;
      const avgFVol = first.reduce((a, r)  => a + r.volume, 0) / N;
      const sumSVol = second.reduce((a, r) => a + r.volume, 0);
      const volOk = sumSVol >= 1.25 * avgFVol * N;
      if (fClose < fOpen && fBody >= 0.5 * fRange &&
          sClose > sOpen && sBody >= 0.7 * sRange &&
          sOpen <= fOpen && sClose >= fClose && volOk) {
        bullishEngulf[i - 1] = 1;
      }
      if (fClose > fOpen && fBody >= 0.5 * fRange &&
          sClose < sOpen && sBody >= 0.7 * sRange &&
          sOpen >= fOpen && sClose <= fClose && volOk) {
        bearishEngulf[i - 1] = 1;
      }
    }

    // 3x3 patterns
    // BUG #6 FIX: mark index i (last bar of third window), not i-1
    for (let i = 8; i < rows.length; i++) {
      // first=bars[i-8..i-5], second=bars[i-5..i-2], third=bars[i-2..i+1]
      // Use fixed 3-bar slices aligned to loop index
      const f0 = rows[i - 8], f1 = rows[i - 7], f2 = rows[i - 6];
      const s0 = rows[i - 5], s1 = rows[i - 4], s2 = rows[i - 3];
      const t0 = rows[i - 2], t1 = rows[i - 1], t2 = rows[i];
      if (!t2) continue;

      const fOpen = f0.open, fClose = f2.close;
      const fBody  = Math.abs(fClose - fOpen);
      // BUG #4 FIX: inline max/min without spread
      const fHigh  = Math.max(f0.high, f1.high, f2.high);
      const fLow   = Math.min(f0.low,  f1.low,  f2.low);
      const fRange = fHigh - fLow;

      const sBody  = Math.abs(s2.close - s0.open);

      const tOpen  = t0.open, tClose = t2.close;
      const tBody  = Math.abs(tClose - tOpen);
      const tHigh  = Math.max(t0.high, t1.high, t2.high);
      const tLow   = Math.min(t0.low,  t1.low,  t2.low);
      const tRange = tHigh - tLow;

      if (fClose < fOpen && fBody >= 0.7 * fRange && sBody <= 0.25 * fBody &&
          tClose > tOpen && tBody >= 0.7 * tRange) {
        bullish3x3[i] = 1; // BUG #6 FIX: mark i, the last bar of third window
      }
      if (fClose > fOpen && fBody >= 0.7 * fRange && sBody <= 0.25 * fBody &&
          tClose < tOpen && tBody >= 0.7 * tRange) {
        bearish3x3[i] = 1;
      }
    }

    return rows.map((r, i) => ({
      idx: i,
      dateStr: r.date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "2-digit" }),
      open: r.open, high: r.high, low: r.low, close: r.close, volume: r.volume,
      ghost: ghostMap.has(i) ? ghostMap.get(i) : undefined,
      neckline: necklineMap.has(i) ? necklineMap.get(i) : undefined,
      rhsShape: rhsShapeMap.has(i) ? rhsShapeMap.get(i) : undefined,
      isCupBottom: result?.cupBottom === i,
      isLeftRim: result?.leftRim === i,
      isRightRim: result?.rightRim === i,
      isHandleMin: result?.handleMinIdx === i,
      momentum: momentum[i],             // O(n) computed above
      candleSignal: candleSignals[i],    // −1/0/1
      spanStrength: spanStrength[i],     // −3..+3 for span color (BUG #7 FIX)
      streak: streaks[i],
      gradientScore: gradientScores[i],  // −1..1
      bullishEngulf:  !!bullishEngulf[i],
      bearishEngulf:  !!bearishEngulf[i],
      bullish3x3:     !!bullish3x3[i],
      bearish3x3:     !!bearish3x3[i],
    }));
  }, [selectedTicker, rawData, tolerance, windowMode, selectedDetection]);

  const radarData = useMemo(() => {
    if (!selectedDetection?.radar) return [];
    const setupType = selectedDetection.setupType;
    const isRHSFamily = setupType === "rhs" || setupType === "hs"; // both shoulder setups
    const isRTFamily  = setupType === "rt"; // rounded top (cup family, bearish)
    // Metric sets per chart tab and setup:
    //  • Gradient view → momentum-only signals (geometry irrelevant)
    //  • Cup pattern   → cup geometry + confirmation
    //  • RHS pattern   → reverse-H&S geometry + confirmation
    // Each entry: { key: radar field, label: display label }
    let entries;
    if (chartSubTab === "gradient") {
      entries = [
        { key: "gradConf",       label: "Gradient",  direct: "gradConf" },
        { key: "pulseStr",       label: "Pulse",     direct: "pulseBonus" },
        { key: "recentMomentum", label: "Momentum",  direct: "recentMomentum" },
        { key: "breakoutProx",   label: "Breakout",  direct: "breakoutProx" },
        { key: "volumeConf",     label: "Volume",    direct: null },
      ];
    } else if (isRHSFamily) {
      entries = [
        { key: "rimSymmetry",  label: "Shoulder Sym", direct: "shoulderSym" },
        { key: "areaSymmetry", label: "Shape Fit",    direct: "areaFit" },
        { key: "spanSymmetry", label: "Width Sym",    direct: "widthSym" },
        { key: "depthScore",   label: "Head Depth",   direct: null },
        { key: "handleQuality",label: setupType === "hs" ? "∩/Λ Shape" : "U/V Shape", direct: "shapeScore" },
        { key: "breakoutProx", label: setupType === "hs" ? "Brkdn Prox" : "Brk Prox", direct: "breakoutProx" },
        { key: "volumeConf",   label: "Volume",       direct: "volConf" },
        { key: "gradConf",     label: setupType === "hs" ? "Brkdn Vol" : "Brk Vol", direct: "volSurge" },
        { key: "recentMomentum", label: "Momentum",   direct: "recentMomentum" },
        { key: "necklineScore",  label: "Neckline",   direct: "necklineScore" },
      ];
    } else if (isRTFamily) {
      // Rounded top reuses cup-family axes but labels differ
      entries = [
        { key: "rimSymmetry",  label: "Rim Sym",   direct: null },
        { key: "areaSymmetry", label: "Area Sym",  direct: "areaSymmetry" },
        { key: "spanSymmetry", label: "Span Sym",  direct: "spanSymmetry" },
        { key: "depthScore",   label: "Arc Depth", direct: null },
        { key: "handleQuality",label: "Dist Shelf",direct: null },
        { key: "breakoutProx", label: "Breakdown", direct: "breakoutProx" },
        { key: "volumeConf",   label: "Volume",    direct: null },
        { key: "gradConf",     label: "Gradient",  direct: "gradConf" },
      ];
    } else {
      entries = [
        { key: "rimSymmetry",  label: "Rim Sym",  direct: null },
        { key: "areaSymmetry", label: "Area Sym", direct: "areaSymmetry" },
        { key: "spanSymmetry", label: "Span Sym", direct: "spanSymmetry" },
        { key: "depthScore",   label: "Depth",    direct: null },
        { key: "handleQuality",label: "Handle",   direct: null },
        { key: "breakoutProx", label: "Breakout", direct: "breakoutProx" },
        { key: "volumeConf",   label: "Volume",   direct: null },
        { key: "gradConf",     label: "Gradient", direct: "gradConf" },
      ];
    }
    return entries.map(({ key, label, direct }) => {
      const radarVal = selectedDetection.radar?.[key] ?? 0;
      const directVal = direct ? (selectedDetection[direct] ?? 0) : 0;
      const val = radarVal > 0 ? radarVal : directVal;
      return { metric: label, value: Math.round(val * 100) };
    });
  }, [selectedDetection, chartSubTab]);

  // ── Filtered scores ──
  const filteredScores = useMemo(() => {
    return scores.filter(s => {
      if (s.score < scoreFilter / 100) return false;
      if (!s.ticker.includes(searchQuery.toUpperCase().trim())) return false;
      // Sector filter (always applied)
      if (leaderSectorFilter !== "All") {
        const sec = sectorMap[s.ticker] || "ETF/Other";
        if (sec !== leaderSectorFilter) return false;
      }
      // Recency filter
      if (recencyFilter === "forming") return !!s.forming;
      if (recencyFilter === "active") return s.barsFromEnd != null && s.barsFromEnd <= 30;
      if (recencyFilter === "recent") return s.barsFromEnd != null && s.barsFromEnd <= 90;
      return true;
    });
  }, [scores, scoreFilter, searchQuery, recencyFilter, leaderSectorFilter, sectorMap]);

  // ── Keyboard shortcuts: J/K navigate leaderboard, Enter open chart, H/L/A switch tabs ──
  useEffect(() => {
    const handler = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      if (e.key === "h" || e.key === "H") { setActiveTab("heatmap"); return; }
      if (e.key === "l" || e.key === "L") { setActiveTab("leaderboard"); return; }
      if (e.key === "d" || e.key === "D") { setActiveTab("domain"); return; }
      if ((e.key === "j" || e.key === "k") && filteredScores.length > 0) {
        setSelectedTicker(prev => {
          const idx = filteredScores.findIndex(s => s.ticker === prev);
          const next = e.key === "j"
            ? Math.min(filteredScores.length - 1, idx + 1)
            : Math.max(0, idx - 1);
          return filteredScores[next]?.ticker || prev;
        });
        // Stay on current view tab (chart/domain) when using J/K
        if (activeTab !== "chart" && activeTab !== "domain") setActiveTab("chart");
      }
      if (e.key === "Enter" && selectedTicker) { setActiveTab("chart"); }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [filteredScores, selectedTicker]);

  const VIRTUAL_ROW_H = 54;

  // ── Upload ──
  const handleFileUpload = useCallback(async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = "";

    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setScanError(`File exceeds ${MAX_FILE_MB} MB limit.`);
      return;
    }

    setScanStatus("scanning");
    setScanError(null);
    setScanProgress(0);
    setParseWarnings([]);
    setScores([]);
    setAllScores([]);
    setSelectedTicker(null);
    cancelRef.current = false;

    let parsed;
    try {
      parsed = await parseCSV(file);
    } catch (err) {
      setScanStatus("error");
      setScanError(err.message);
      return;
    }

    setRawData(parsed.map);
    setParseWarnings(parsed.warnings);
    if (parsed.warnings.length) setShowWarnings(true);

    if (parsed.map.size === 0) {
      setScanStatus("error");
      setScanError(`No valid tickers passed the ${MIN_BARS}-bar minimum. ${parsed.warnings.length} tickers were excluded — check warnings for details.`);
      return;
    }

    try {
      const results = await runScan(parsed.map, tolerance, setScanProgress, cancelRef, windowMode);
      if (!cancelRef.current) {
        // Save current rank positions for delta tracking before overwriting
        if (scores.length > 0) {
          const rankMap = {};
          scores.forEach((s, i) => { rankMap[s.ticker] = i + 1; });
          setPrevScores(rankMap);
        }
        setAllScores(results);
        setScores(results);
        setScanStatus("done");
        setScanProgress(100);
        if (results.length > 0) {
          setSelectedTicker(results[0].ticker);
          setActiveTab("chart");
        }
      } else {
        setScanStatus("idle");
      }
    } catch (err) {
      setScanStatus("error");
      setScanError(`Scan failed: ${err.message}`);
    }
  }, [tolerance, windowMode]);

  // ── Drag-and-drop ──
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.name.endsWith(".csv")) {
      handleFileUpload({ target: { files: [file], value: "" } });
    }
  }, [handleFileUpload]);

  // ── Tolerance change ──
  const handleToleranceChange = useCallback((key, value) => {
    const newTol = { ...tolerance, [key]: value };
    setTolerance(newTol);
    if (rawData && allScores.length > 0) {
      const reranked = rerank(allScores, newTol, rawData, windowMode);
      setScores(reranked);
    }
  }, [tolerance, rawData, allScores, windowMode]);

  // ── Setup switch: both setups are already stored per ticker, so just
  // re-point the leaderboard at the chosen setup and re-rank. No re-scan.
  // CRITICAL: repoint from allScores (the FULL set, every row carries both
  // cup+flag sub-objects), not from the already-filtered `scores` — otherwise
  // toggling progressively shrinks the list to the intersection. ──
  const handleSetupChange = useCallback((setup, keepTicker = false) => {
    if (setup === tolerance.activeSetup) return;
    setTolerance(prev => ({ ...prev, activeSetup: setup }));
    const repointed = repointSetup(allScores, setup);
    setScores(repointed);
    // If not keeping the current ticker, jump to the top-ranked ticker of the
    // new setup so the chart immediately shows a real example of that pattern.
    if (!keepTicker) setSelectedTicker(repointed.length > 0 ? repointed[0].ticker : null);
  }, [tolerance.activeSetup, allScores]);


  // ─── Heatmap data ─────────────────────────────────────────────────────────────
  const hmData = useMemo(() => {
    if (!rawData) return [];
    const q = hmSearch.toUpperCase().trim();
    const results = [];
    for (const [ticker, ohlcv] of rawData) {
      if (q && !ticker.includes(q)) continue;
      if (ohlcv.length < hmWindow) continue;
      const pw = computePulseWave(ohlcv, hmWindow);
      const sector = sectorMap[ticker] || "ETF/Other";
      results.push({ ticker, ...pw, ohlcv, sector });
    }
    return results.sort((a, b) => b.total - a.total);
  }, [rawData, hmWindow, hmSearch, sectorMap]);

  // ─── Sector grouped heatmap data ─────────────────────────────────────────────
  const hmBySector = useMemo(() => {
    const map = {};
    for (const d of hmData) {
      const s = d.sector;
      if (!map[s]) map[s] = [];
      map[s].push(d);
    }
    // Sort sectors by average pulse score descending
    const sorted = Object.entries(map).sort((a, b) => {
      const avg = arr => arr.reduce((s, d) => s + d.total, 0) / arr.length;
      return avg(b[1]) - avg(a[1]);
    });
    return sorted; // [[sectorName, [tickers...]], ...]
  }, [hmData]);


  const cancelScan = useCallback(() => {
    cancelRef.current = true;
    setScanStatus("idle");
  }, []);

  // ── Export ──
  const handleExport = useCallback(() => {
    if (!scores.length) return;
    const rows = scores.map(s => ({
      Ticker: s.ticker,
      SetupType: s.detection?.setupType === "rhs" ? "Reverse H&S"
        : s.detection?.setupType === "hs"  ? "Head & Shoulders"
        : s.detection?.setupType === "rt"  ? "Rounded Top"
        : "Cup & Handle",
      Score: (s.score * 100).toFixed(1),
      Bars: s.bars,
      Triggered: s.detection?.triggered ? "Yes" : "No",
      BreakoutProximity: s.detection ? (s.detection.breakoutProx * 100).toFixed(1) + "%" : "",
      // Cup-specific
      CupDepth: s.detection && s.detection.setupType !== "rhs" ? (s.detection.cupDepth * 100).toFixed(1) + "%" : "",
      HandleRetrace: s.detection && s.detection.setupType !== "rhs" ? (s.detection.handleRetrace * 100).toFixed(1) + "%" : "",
      AreaSymmetry: s.detection && s.detection.setupType !== "rhs" ? (s.detection.areaSymmetry * 100).toFixed(1) + "%" : "",
      SpanSymmetry: s.detection && s.detection.setupType !== "rhs" ? (s.detection.spanSymmetry * 100).toFixed(1) + "%" : "",
      // Flag-specific
      HeadDepth: s.detection?.setupType === "rhs" ? (s.detection.headDepth * 100).toFixed(1) + "%" : "",
      ShoulderSym: s.detection?.setupType === "rhs" ? (s.detection.shoulderSym * 100).toFixed(1) + "%" : "",
      NecklineScore: s.detection?.setupType === "rhs" ? (s.detection.necklineScore * 100).toFixed(1) + "%" : "",
      VolSurge: s.detection?.setupType === "rhs" ? (s.detection.volSurge * 100).toFixed(1) + "%" : "",
      VolumeConf: s.detection ? (s.detection.volConf * 100).toFixed(1) + "%" : "",
    }));
    const csv = Papa.unparse(rows);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `cup_handle_scan_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [scores]);

  // ── Heatmap export ──
  const handleExportHeatmap = useCallback(() => {
    if (!hmData.length) return;
    const rows = hmData.map(d => ({
      Ticker: d.ticker,
      Sector: d.sector,
      PulseScore: (d.total * 100).toFixed(1),
      GradientArc: (d.gradient * 100).toFixed(1),
      CandleStreak: (d.streak * 100).toFixed(1),
      StreakBars: d.streakVal,
      VolumeRatio: d.volume.toFixed(2),
    }));
    const csv = Papa.unparse(rows);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `pulse_heatmap_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [hmData]);


  // ── Dynamic colors based on darkMode (shadows module-level COLORS inside component) ──
  // eslint-disable-next-line no-shadow
  const COLORS = darkMode ? COLORS_DARK : {
    bg: "#f4f5f8",
    surface: "#ffffff",
    surfaceHover: "#eef0f8",
    border: "#d0d5ea",
    accent: "#3a5bd9",
    accentDim: "#c5d0f5",
    green: "#1a8c82",
    red: "#c62828",
    gold: "#b8860b",
    text: "#1a1d2e",
    textDim: "#4a5080",
    textMuted: "#8890b0",
    cup: "#3a5bd9",
    handle: "#b8860b",
    ghost: "#ff4dd2",
    warning: "#e65100",
  };

  // ─── Styles ──────────────────────────────────────────────────────────────────
  const S = {
    app: {
      background: COLORS.bg, color: COLORS.text,
      height: "100%",
      fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
      display: "flex", flexDirection: "column", fontSize: 14,
      overflow: "hidden",
      width: "100%", maxWidth: "100vw",
      boxSizing: "border-box",
    },
    header: {
      background: COLORS.surface, borderBottom: `1px solid ${COLORS.border}`,
      padding: "0 20px", display: "flex", alignItems: "center",
      justifyContent: "space-between", flexShrink: 0, height: 52, gap: 12
    },
    logo: { fontSize: 17, fontWeight: 800, color: COLORS.accent, letterSpacing: "-0.5px", whiteSpace: "nowrap" },
    logoSub: { fontSize: 11, color: COLORS.textMuted, marginLeft: 8 },
    main: { display: "flex", flex: 1, minHeight: 0, overflow: "hidden", flexDirection: isMobile ? "column" : "row" },
    sidebar: isMobile ? { display: "none" } : {
      width: 272, flexShrink: 0, background: COLORS.surface,
      borderRight: `1px solid ${COLORS.border}`,
      display: "flex", flexDirection: "column", overflow: "hidden"
    },
    sidebarScroll: { flex: 1, overflowY: "auto", padding: 14 },
    content: { flex: 1, minWidth: 0, display: "flex", flexDirection: "column", overflow: "hidden" },
    tabBar: {
      display: isMobile ? "none" : "flex", borderBottom: `1px solid ${COLORS.border}`,
      background: COLORS.surface, flexShrink: 0, padding: "0 4px"
    },
    tab: (active) => ({
      padding: "14px 18px", fontSize: 13, fontWeight: active ? 700 : 500,
      color: active ? COLORS.accent : COLORS.textDim,
      borderBottom: active ? `2px solid ${COLORS.accent}` : "2px solid transparent",
      cursor: "pointer", background: "none", border: "none",
      outline: "none", transition: "color 0.15s", whiteSpace: "nowrap"
    }),
    panel: { flex: 1, overflowY: "auto", overflowX: "hidden", minHeight: 0, minWidth: 0, paddingBottom: isMobile ? 80 : 0, overscrollBehavior: "contain", WebkitOverflowScrolling: "touch" },
    card: {
      background: COLORS.surfaceHover, border: `1px solid ${COLORS.border}`,
      borderRadius: 10, padding: 14, marginBottom: 10
    },
    sectionTitle: {
      fontSize: 10, fontWeight: 700, color: COLORS.textMuted,
      letterSpacing: "1px", textTransform: "uppercase", marginBottom: 10
    },
    uploadZone: (dragging) => ({
      border: `2px dashed ${dragging ? COLORS.accent : COLORS.accentDim}`,
      borderRadius: 10, padding: "24px 14px", textAlign: "center",
      cursor: "pointer", background: dragging ? "rgba(108,143,255,0.08)" : "rgba(108,143,255,0.02)",
      transition: "all 0.15s"
    }),
    btn: (variant = "primary", disabled = false) => ({
      padding: "8px 14px", borderRadius: 8, fontSize: 12, fontWeight: 600,
      cursor: disabled ? "not-allowed" : "pointer", border: "none", outline: "none",
      opacity: disabled ? 0.45 : 1, transition: "opacity 0.15s",
      background: variant === "primary" ? COLORS.accent
        : variant === "danger" ? COLORS.red
        : variant === "ghost" ? "transparent"
        : COLORS.border,
      color: variant === "ghost" ? COLORS.textDim : COLORS.bg,
    }),
    input: {
      background: COLORS.surface, border: `1px solid ${COLORS.border}`,
      borderRadius: 8, color: COLORS.text, padding: "9px 13px",
      fontSize: 13, outline: "none", width: "100%", boxSizing: "border-box",
      transition: "border-color 0.15s"
    },
    leaderRow: (selected, isMobile) => ({
      display: "flex", alignItems: isMobile ? "flex-start" : "center", gap: 10,
      padding: "9px 14px", borderRadius: 8, cursor: "pointer",
      background: selected ? COLORS.accentDim : "transparent",
      border: `1px solid ${selected ? COLORS.accent : "transparent"}`,
      transition: "background 0.1s",
      height: isMobile ? "auto" : VIRTUAL_ROW_H - 6,
      minHeight: isMobile ? VIRTUAL_ROW_H - 6 : undefined,
      boxSizing: "border-box", marginBottom: isMobile ? 6 : 4,
    }),
    chatBubble: (role) => ({
      maxWidth: "82%", padding: "10px 14px",
      borderRadius: role === "user" ? "14px 14px 4px 14px" : "14px 14px 14px 4px",
      background: role === "user" ? COLORS.accentDim : COLORS.surfaceHover,
      color: COLORS.text, fontSize: 13, lineHeight: 1.65,
      alignSelf: role === "user" ? "flex-end" : "flex-start",
      border: `1px solid ${COLORS.border}`,
      whiteSpace: "pre-wrap", wordBreak: "break-word"
    }),
  };

  // ─── Upload area ─────────────────────────────────────────────────────────────
  const [dragging, setDragging] = useState(false);

  const renderUpload = () => (
    <div style={S.card}>
      <div style={S.sectionTitle}>Data Source</div>
      {(scanStatus === "idle" || scanStatus === "done") && rawData && (
        <div style={{
          padding: "8px 12px", borderRadius: 8, marginBottom: 10,
          background: "rgba(38,166,154,0.1)", border: "1px solid rgba(38,166,154,0.3)",
          fontSize: 11, color: COLORS.green, lineHeight: 1.5
        }}>
          ✓ Default dataset loaded ({rawData.size} tickers)
        </div>
      )}
      <label
        style={{ ...S.uploadZone(dragging), padding: "12px 10px" }}
        htmlFor="csv-upload"
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => { setDragging(false); handleDrop(e); }}
      >
        <div style={{ fontSize: 18, marginBottom: 4 }}>📂</div>
        <div style={{ fontSize: 12, fontWeight: 600, color: COLORS.textDim, marginBottom: 2 }}>
          Upload your own CSV
        </div>
        <div style={{ fontSize: 10, color: COLORS.textMuted, lineHeight: 1.5 }}>
          date · ticker · open · high · low · close · volume
        </div>
        <input
          id="csv-upload" type="file" accept=".csv" style={{ display: "none" }}
          onChange={handleFileUpload} disabled={scanStatus === "scanning"}
        />
      </label>
      {scanStatus === "error" && scanError && (
        <div style={{
          marginTop: 10, padding: "10px 12px", borderRadius: 8,
          background: "rgba(239,83,80,0.12)", border: `1px solid ${COLORS.red}`,
          fontSize: 12, color: COLORS.red, lineHeight: 1.5
        }}>
          ⚠ {scanError}
        </div>
      )}
    </div>
  );

  // ─── Progress ────────────────────────────────────────────────────────────────
  const renderProgress = () => (
    <div style={{ ...S.card, borderColor: COLORS.accentDim }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <span style={{ fontSize: 13, fontWeight: 600 }}>
          {cancelRef.current ? "Cancelling…" : `Scanning… ${elapsed}s`}
        </span>
        <span style={{ fontSize: 12, color: COLORS.textDim }}>{scanProgress}%</span>
      </div>
      <div style={{ height: 5, background: COLORS.border, borderRadius: 3, overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${scanProgress}%`,
          background: `linear-gradient(90deg, ${COLORS.accent}, ${COLORS.green})`,
          borderRadius: 3, transition: "width 0.3s"
        }} />
      </div>
      {elapsed > 5 && (
        <button style={{ ...S.btn("danger"), marginTop: 10, width: "100%", fontSize: 12 }} onClick={cancelScan}>
          Cancel Scan
        </button>
      )}
    </div>
  );

  // ─── Tolerances ──────────────────────────────────────────────────────────────
  const renderTolerances = () => {
    const params = [
      {
        label: "Cup Depth Min", value: tolerance.cupDepth[0], min: 0.05, max: 0.40, step: 0.01,
        fmt: v => (v * 100).toFixed(0) + "%",
        onChange: v => handleToleranceChange("cupDepth", [v, tolerance.cupDepth[1]])
      },
      {
        label: "Cup Depth Max", value: tolerance.cupDepth[1], min: 0.15, max: 0.70, step: 0.01,
        fmt: v => (v * 100).toFixed(0) + "%",
        onChange: v => handleToleranceChange("cupDepth", [tolerance.cupDepth[0], v])
      },
      {
        label: "Handle Retrace Min", value: tolerance.handleRetrace[0], min: 0.05, max: 0.35, step: 0.01,
        fmt: v => (v * 100).toFixed(0) + "%",
        onChange: v => handleToleranceChange("handleRetrace", [v, tolerance.handleRetrace[1]])
      },
      {
        label: "Handle Retrace Max", value: tolerance.handleRetrace[1], min: 0.15, max: 0.70, step: 0.01,
        fmt: v => (v * 100).toFixed(0) + "%",
        onChange: v => handleToleranceChange("handleRetrace", [tolerance.handleRetrace[0], v])
      },
      {
        label: "Smoothing", value: tolerance.smoothing, min: 0.02, max: 0.20, step: 0.01,
        fmt: v => v.toFixed(2),
        onChange: v => handleToleranceChange("smoothing", v)
      },
    ];

    return (
      <div style={S.card}>
        <div style={{ ...S.sectionTitle, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span>Detection Parameters</span>
          <button
            style={{ ...S.btn("ghost"), fontSize: 10, padding: "1px 7px", color: COLORS.textMuted }}
            onClick={() => {
              const defaults = { cupDepth: [0.12, 0.50], handleRetrace: [0.10, 0.50], smoothing: 0.08, minBars: MIN_BARS };
              setTolerance(defaults);
              if (rawData && allScores.length > 0) setScores(rerank(allScores, defaults, rawData, windowMode));
            }}
            title="Reset all parameters to defaults"
          >
            ↺ Reset
          </button>
        </div>

        {/* Scan Window */}
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 11, color: COLORS.textDim, marginBottom: 6 }}>
            Scan Window
            <span style={{ float: "right", fontSize: 10, color: COLORS.textMuted }}>
              helps find patterns hidden by later spikes
            </span>
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            {[
              { id: "auto", label: "Auto", title: "Try full, 75%, 50% — keep best scoring window" },
              { id: "full", label: "Full", title: "Scan full series only (original behavior)" },
              { id: "75",   label: "75%",  title: "Scan last 75% of bars only" },
              { id: "50",   label: "50%",  title: "Scan last 50% of bars only" },
            ].map(({ id, label, title }) => {
              const active = windowMode === id;
              return (
                <button
                  key={id}
                  title={title}
                  disabled={scanStatus === "scanning"}
                  onClick={() => {
                    setWindowMode(id);
                    if (rawData && allScores.length > 0) {
                      setScores(rerank(allScores, tolerance, rawData, id));
                    }
                  }}
                  style={{
                    flex: 1, fontSize: 11, fontWeight: active ? 700 : 500,
                    padding: "5px 0", borderRadius: 6, cursor: "pointer",
                    border: `1px solid ${active ? COLORS.accent : COLORS.border}`,
                    background: active ? "rgba(108,143,255,0.15)" : "transparent",
                    color: active ? COLORS.accent : COLORS.textDim,
                    transition: "all 0.15s",
                  }}
                >
                  {label}
                </button>
              );
            })}
          </div>
        </div>
        {params.map(({ label, value, min, max, step, fmt, onChange }, i) => (
          <div key={i} style={{ marginBottom: 13 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
              <span style={{ fontSize: 11, color: COLORS.textDim }}>{label}</span>
              <span style={{ fontSize: 11, fontWeight: 700, color: COLORS.accent }}>{fmt(value)}</span>
            </div>
            <input
              type="range" min={min} max={max} step={step} value={value}
              onChange={e => onChange(parseFloat(e.target.value))}
              disabled={scanStatus === "scanning"}
              style={{ width: "100%", accentColor: COLORS.accent, cursor: "pointer" }}
            />
          </div>
        ))}
      </div>
    );
  };

  // ─── Scan summary ─────────────────────────────────────────────────────────────
  const renderScanSummary = () => {
    if (!rawData) return null;
    return (
      <div style={{ ...S.card, display: "flex", gap: 8, flexWrap: "wrap" }}>
        <div style={{ flex: 1, minWidth: 60 }}>
          <div style={{ fontSize: 10, color: COLORS.textMuted, textTransform: "uppercase", letterSpacing: "0.5px" }}>Tickers</div>
          <div style={{ fontSize: 18, fontWeight: 800, color: COLORS.text }}>{rawData.size}</div>
        </div>
        <div style={{ flex: 1, minWidth: 60 }}>
          <div style={{ fontSize: 10, color: COLORS.textMuted, textTransform: "uppercase", letterSpacing: "0.5px" }}>Matched</div>
          <div style={{ fontSize: 18, fontWeight: 800, color: COLORS.accent }}>{scores.length}</div>
        </div>
        {parseWarnings.length > 0 && (
          <button
            style={{ ...S.btn("ghost"), fontSize: 11, padding: "4px 8px", color: COLORS.warning }}
            onClick={() => setShowWarnings(v => !v)}
          >
            {parseWarnings.length} warning{parseWarnings.length !== 1 ? "s" : ""} {showWarnings ? "▲" : "▼"}
          </button>
        )}
      </div>
    );
  };

  // ─── Leaderboard tab ─────────────────────────────────────────────────────────
  const renderLeaderboard = () => {
    const totalItems = filteredScores.length;
    const visibleStart = 0;
    const visibleEnd = totalItems;
    const visible = filteredScores;

    return (
      <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
        {/* Filter bar */}
        <div style={{
          padding: "10px 16px", borderBottom: `1px solid ${COLORS.border}`,
          display: "flex", gap: 10, alignItems: "center", flexShrink: 0, background: COLORS.surface,
          flexWrap: "wrap"
        }}>
          <input
            style={{ ...S.input, maxWidth: 160 }}
            placeholder="Search ticker…"
            value={searchQuery}
            onChange={e => { setSearchQuery(e.target.value); }}
          />
          <div style={{ flex: 1, minWidth: 100 }}>
            <div style={{ fontSize: 10, color: COLORS.textMuted, marginBottom: 3 }}>
              Min score: {scoreFilter}%
            </div>
            <input
              type="range" min={0} max={90} step={5} value={scoreFilter}
              onChange={e => { setScoreFilter(parseInt(e.target.value)); }}
              style={{ width: "100%", accentColor: COLORS.accent }}
            />
          </div>
          <button
            style={S.btn("secondary", !scores.length)}
            onClick={handleExport}
            disabled={!scores.length}
            title="Export CSV"
          >
            ↓ CSV
          </button>
        </div>

        {/* Recency filter strip */}
        <div style={{
          padding: "8px 16px", borderBottom: `1px solid ${COLORS.border}`,
          display: "flex", gap: 6, alignItems: "center", flexShrink: 0,
          background: COLORS.bg, overflowX: "auto", WebkitOverflowScrolling: "touch"
        }}>
          <span style={{ fontSize: 10, color: COLORS.textMuted, fontWeight: 600, letterSpacing: "0.5px", textTransform: "uppercase", marginRight: 4, whiteSpace: "nowrap" }}>
            Recency
          </span>
          {[
            { id: "all",     label: "All",           title: "Show all detected patterns regardless of age" },
            { id: "recent",  label: "≤ 90 bars",     title: "Right rim formed within last 90 bars" },
            { id: "active",  label: "≤ 30 bars",     title: "Right rim formed within last 30 bars — handle forming or just broke out" },
            { id: "forming", label: "Forming only",  title: "Show only partial patterns still building the right side" },
          ].map(({ id, label, title }) => {
            const active = recencyFilter === id;
            return (
              <button
                key={id}
                title={title}
                onClick={() => setRecencyFilter(id)}
                style={{
                  fontSize: 11, fontWeight: active ? 700 : 500,
                  padding: "4px 10px", borderRadius: 20, cursor: "pointer",
                  border: `1px solid ${active ? (
                    id === "forming" ? COLORS.gold :
                    id === "active"  ? COLORS.green :
                    id === "recent"  ? COLORS.accent : COLORS.border
                  ) : COLORS.border}`,
                  background: active ? (
                    id === "forming" ? "rgba(255,213,79,0.12)" :
                    id === "active"  ? "rgba(38,166,154,0.12)" :
                    id === "recent"  ? "rgba(108,143,255,0.12)" : COLORS.surfaceHover
                  ) : "transparent",
                  color: active ? (
                    id === "forming" ? COLORS.gold :
                    id === "active"  ? COLORS.green :
                    id === "recent"  ? COLORS.accent : COLORS.text
                  ) : COLORS.textDim,
                  transition: "all 0.15s",
                  whiteSpace: "nowrap",
                }}
              >
                {id === "forming" ? "🔄 " : id === "active" ? "🟢 " : id === "recent" ? "⏱ " : ""}{label}
              </button>
            );
          })}
        </div>

        {/* Setup selector: 2 bullish + 2 bearish */}
        {rawData && rawData.size > 0 && (() => {
          const bullOpts = [
            { id: "cup", label: "⌣ Cup & Handle", color: COLORS.cup },
            { id: "rhs", label: "⋎ Reverse H&S",  color: COLORS.green },
          ];
          const bearOpts = [
            { id: "hs",  label: "⋏ H&S",           color: COLORS.red },
            { id: "rt",  label: "⌢ Rounded Top",    color: COLORS.warning },
          ];
          const renderBtn = ({ id, label, color }) => {
            const active = tolerance.activeSetup === id;
            return (
              <button
                key={id}
                onClick={() => handleSetupChange(id)}
                disabled={scanStatus === "scanning"}
                style={{
                  fontSize: isMobile ? 11 : 13, fontWeight: active ? 800 : 600,
                  padding: isMobile ? "6px 10px" : "8px 16px", borderRadius: 9,
                  cursor: scanStatus === "scanning" ? "wait" : "pointer",
                  border: `1px solid ${active ? color : COLORS.border}`,
                  background: active ? `${color}22` : COLORS.surfaceHover,
                  color: active ? color : COLORS.text,
                  transition: "all 0.15s", whiteSpace: "nowrap",
                }}
              >
                {label}
              </button>
            );
          };
          return (
            <div style={{
              padding: "10px 16px", borderBottom: `1px solid ${COLORS.border}`,
              display: "flex", gap: 6, alignItems: "center", flexShrink: 0, background: COLORS.bg,
              flexWrap: "nowrap", overflowX: "auto", WebkitOverflowScrolling: "touch",
            }}>
              {/* Bullish group */}
              <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "nowrap" }}>
                <span style={{ fontSize: 9, color: COLORS.green, fontWeight: 700, letterSpacing: "0.5px", textTransform: "uppercase", whiteSpace: "nowrap" }}>
                  Bull
                </span>
                {bullOpts.map(renderBtn)}
              </div>
              <div style={{ width: 1, height: 24, background: COLORS.border, flexShrink: 0, margin: "0 2px" }} />
              {/* Bearish group */}
              <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "nowrap" }}>
                <span style={{ fontSize: 9, color: COLORS.red, fontWeight: 700, letterSpacing: "0.5px", textTransform: "uppercase", whiteSpace: "nowrap" }}>
                  Bear
                </span>
                {bearOpts.map(renderBtn)}
              </div>
              {!isMobile && (
                <span style={{ fontSize: 10, color: COLORS.textMuted, marginLeft: 4 }}>
                  {tolerance.activeSetup === "rhs" ? "Reverse H&S (bullish)" :
                   tolerance.activeSetup === "hs"  ? "Head & Shoulders (bearish)" :
                   tolerance.activeSetup === "rt"  ? "Rounded Top (bearish)" :
                   "Cup & Handle (bullish)"} · all 4 detected, toggle freely
                </span>
              )}
            </div>
          );
        })()}
        {scores.length > 0 && (() => {
          const sectorCounts = {};
          for (const s of scores) {
            const sec = sectorMap[s.ticker] || "ETF/Other";
            sectorCounts[sec] = (sectorCounts[sec] || 0) + 1;
          }
          const sectors = ["All", ...Object.keys(sectorCounts).sort()];
          return (
            <div style={{
              padding: "6px 16px", borderBottom: `1px solid ${COLORS.border}`,
              display: "flex", gap: 5, alignItems: "center", flexShrink: 0,
              background: COLORS.bg, overflowX: "auto", flexWrap: "nowrap",
            }}>
              <span style={{ fontSize: 10, color: COLORS.textMuted, fontWeight: 600, letterSpacing: "0.5px", textTransform: "uppercase", marginRight: 4, whiteSpace: "nowrap", flexShrink: 0 }}>
                Sector
              </span>
              {sectors.map(sec => {
                const active = leaderSectorFilter === sec;
                const count = sec === "All" ? scores.length : (sectorCounts[sec] || 0);
                return (
                  <button
                    key={sec}
                    onClick={() => setLeaderSectorFilter(sec)}
                    style={{
                      fontSize: 10, fontWeight: active ? 700 : 500,
                      padding: "3px 9px", borderRadius: 20, cursor: "pointer", flexShrink: 0,
                      border: `1px solid ${active ? COLORS.accent : COLORS.border}`,
                      background: active ? "rgba(108,143,255,0.15)" : "transparent",
                      color: active ? COLORS.accent : COLORS.textDim,
                      transition: "all 0.15s", whiteSpace: "nowrap",
                    }}
                  >
                    {sec} <span style={{ opacity: 0.6 }}>({count})</span>
                  </button>
                );
              })}
            </div>
          );
        })()}

        {/* Results count */}
        <div style={{ padding: "8px 16px", fontSize: 11, color: COLORS.textMuted, flexShrink: 0 }}>
          {filteredScores.length} result{filteredScores.length !== 1 ? "s" : ""}
          {filteredScores.length !== scores.length ? ` (filtered from ${scores.length})` : ""}
        </div>

        {/* Empty states */}
        {scanStatus === "idle" && scores.length === 0 && (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12, color: COLORS.textMuted }}>
            <div style={{ fontSize: 40 }}>📂</div>
            <div style={{ fontSize: 14, fontWeight: 600 }}>Upload a CSV to begin screening</div>
            <div style={{ fontSize: 12, textAlign: "center", maxWidth: 260, lineHeight: 1.6 }}>
              The engine will detect cup-and-handle patterns across all tickers and rank them by pattern quality score.
            </div>
          </div>
        )}

        {scanStatus === "scanning" && (
          <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <div style={{ textAlign: "center", color: COLORS.textDim }}>
              <div style={{ fontSize: 28, marginBottom: 10 }}>⚙️</div>
              <div style={{ fontSize: 13 }}>Scanning… {scanProgress}%</div>
            </div>
          </div>
        )}

        {scanStatus === "done" && filteredScores.length === 0 && (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 8, color: COLORS.textMuted }}>
            <div style={{ fontSize: 28 }}>🔍</div>
            <div style={{ fontSize: 13 }}>No patterns match current filters</div>
            <button style={S.btn("ghost")} onClick={() => { setScoreFilter(0); setSearchQuery(""); setRecencyFilter("all"); }}>
              Clear filters
            </button>
          </div>
        )}

        {/* Virtual list */}
        {filteredScores.length > 0 && (
          <div
            ref={listRef}
            style={{ flex: 1, overflowY: "auto", padding: "0 12px 12px" }}
          >
            {visible.map((item, localIdx) => {
              const globalIdx = visibleStart + localIdx;
              const selected = item.ticker === selectedTicker;
              const pct = item.score * 100;
              const currentRank = globalIdx + 1;
              const prevRank = prevScores[item.ticker];
              const delta = prevRank != null ? prevRank - currentRank : null;
              const sector = sectorMap[item.ticker];
              return (
                <div
                  key={item.ticker}
                  style={S.leaderRow(selected, isMobile)}
                  onClick={() => {
                    setSelectedTicker(item.ticker);
                    // Stay on chart or domain if already there; otherwise go to chart
                    if (activeTab !== "chart" && activeTab !== "domain") setActiveTab("chart");
                  }}
                >
                  <div style={{ fontSize: 11, color: COLORS.textMuted, width: 22, flexShrink: 0, textAlign: "right" }}>
                    {currentRank}
                  </div>
                  {/* Delta rank */}
                  <div style={{ width: 22, flexShrink: 0, textAlign: "center", fontSize: 10, fontWeight: 700 }}>
                    {delta != null && delta !== 0 ? (
                      <span style={{ color: delta > 0 ? COLORS.green : COLORS.red }}>
                        {delta > 0 ? `▲${delta}` : `▼${Math.abs(delta)}`}
                      </span>
                    ) : delta === 0 ? (
                      <span style={{ color: COLORS.textMuted }}>—</span>
                    ) : null}
                  </div>
                  <ScoreRing score={item.score} size={42} dashed={!!item.forming} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                      <span style={{ fontWeight: 700, fontSize: 14, color: selected ? COLORS.accent : COLORS.text }}>
                        {item.ticker}
                      </span>
                      {sector && (
                        <span style={{
                          fontSize: 9, fontWeight: 600, padding: "1px 5px", borderRadius: 4,
                          background: "rgba(108,143,255,0.08)", color: COLORS.textDim,
                          border: `1px solid ${COLORS.border}`, letterSpacing: "0.3px",
                          maxWidth: 90, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                        }} title={sector}>
                          {sector}
                        </span>
                      )}
                      {item.detection?.setupType === "rhs" && (
                        <span style={{
                          fontSize: 9, fontWeight: 700, padding: "1px 5px", borderRadius: 4,
                          background: "rgba(38,166,154,0.15)", color: COLORS.green,
                          border: `1px solid ${COLORS.green}`, letterSpacing: "0.5px"
                        }}>
                          ⋎ REV H&S
                        </span>
                      )}
                      {item.detection?.setupType === "hs" && (
                        <span style={{
                          fontSize: 9, fontWeight: 700, padding: "1px 5px", borderRadius: 4,
                          background: "rgba(239,83,80,0.15)", color: COLORS.red,
                          border: `1px solid ${COLORS.red}`, letterSpacing: "0.5px"
                        }}>
                          ⋏ H&S
                        </span>
                      )}
                      {item.detection?.setupType === "rt" && (
                        <span style={{
                          fontSize: 9, fontWeight: 700, padding: "1px 5px", borderRadius: 4,
                          background: "rgba(255,152,0,0.15)", color: COLORS.warning,
                          border: `1px solid ${COLORS.warning}`, letterSpacing: "0.5px"
                        }}>
                          ⌢ ROUNDED TOP
                        </span>
                      )}
                      {item.detection && item.detection.setupType !== "rhs" && item.detection.setupType !== "hs" && item.detection.setupType !== "rt" && !item.forming && (
                        <span style={{
                          fontSize: 9, fontWeight: 700, padding: "1px 5px", borderRadius: 4,
                          background: "rgba(108,143,255,0.12)", color: COLORS.cup,
                          border: `1px solid ${COLORS.cup}`, letterSpacing: "0.5px"
                        }}>
                          ⌣ CUP
                        </span>
                      )}
                      {item.detection?.triggered && (
                        <span style={{
                          fontSize: 9, fontWeight: 700, padding: "1px 5px", borderRadius: 4,
                          background: (item.detection.setupType === "hs" || item.detection.setupType === "rt")
                            ? "rgba(239,83,80,0.22)" : "rgba(38,166,154,0.22)",
                          color: (item.detection.setupType === "hs" || item.detection.setupType === "rt")
                            ? COLORS.red : COLORS.green,
                          border: `1px solid ${(item.detection.setupType === "hs" || item.detection.setupType === "rt") ? COLORS.red : COLORS.green}`,
                          letterSpacing: "0.5px"
                        }}>
                          {(item.detection.setupType === "hs" || item.detection.setupType === "rt") ? "BROKE DOWN ▼" : "BROKE OUT ▲"}
                        </span>
                      )}
                      {item.forming && (
                        <span style={{
                          fontSize: 9, fontWeight: 700, padding: "1px 5px", borderRadius: 4,
                          background: "rgba(255,213,79,0.15)", color: COLORS.gold,
                          border: `1px solid ${COLORS.gold}`, letterSpacing: "0.5px"
                        }}>
                          FORMING · S{item.forming.stage}
                        </span>
                      )}
                      {item.detection?.rimSignalBullish && (
                        <span style={{
                          fontSize: 9, fontWeight: 700, padding: "1px 5px", borderRadius: 4,
                          background: "rgba(38,166,154,0.15)", color: COLORS.green,
                          border: `1px solid ${COLORS.green}`, letterSpacing: "0.5px"
                        }}>
                          {item.detection.rimSignalStrength === 2 ? "3×3 ▲" : "ENGULF ▲"}
                        </span>
                      )}
                      {item.window && item.window !== "full" && (
                        <span style={{
                          fontSize: 9, fontWeight: 600, padding: "1px 5px", borderRadius: 4,
                          background: "rgba(108,143,255,0.10)", color: COLORS.textDim,
                          border: `1px solid ${COLORS.border}`, letterSpacing: "0.3px"
                        }}>
                          ⊞ {item.window}
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: 10, color: COLORS.textMuted, marginTop: 2 }}>
                      {item.bars} bars
                      {(item.detection?.setupType === "rhs" || item.detection?.setupType === "hs") ? (
                        <>
                          {` · head ${(item.detection.headDepth * 100).toFixed(0)}% ${item.detection.setupType === "hs" ? "above" : "below"} shoulders`}
                          {` · sym ${(item.detection.shoulderSym * 100).toFixed(0)}%`}
                          {item.detection.volSurge > 0.3 && ` · vol surge`}
                        </>
                      ) : (
                        <>
                          {item.detection && ` · depth ${(item.detection.cupDepth * 100).toFixed(0)}%`}
                          {item.detection && item.detection.setupType !== "rt" && ` · handle ${(item.detection.handleRetrace * 100).toFixed(0)}%`}
                          {item.detection && item.detection.setupType === "rt" && ` · bounce ${(item.detection.handleRetrace * 100).toFixed(0)}%`}
                          {item.detection && (item.detection.handleStreakVal > 0) && item.detection.setupType !== "rt" && ` · streak +${item.detection.handleStreakVal}`}
                          {item.detection && (item.detection.handleStreakVal < 0) && item.detection.setupType === "rt" && ` · streak ${item.detection.handleStreakVal}`}
                        </>
                      )}
                      {item.forming && ` · recovery ${(item.forming.recoveryPct * 100).toFixed(0)}%`}
                      {item.barsFromEnd != null && (
                        <span style={{
                          marginLeft: 6,
                          color: item.barsFromEnd <= 30 ? COLORS.green : item.barsFromEnd <= 90 ? COLORS.gold : COLORS.textMuted,
                          fontWeight: item.barsFromEnd <= 30 ? 700 : 500,
                        }}>
                          · {item.barsFromEnd === 0 ? "now" : `${item.barsFromEnd}b ago`}
                        </span>
                      )}
                    </div>
                    <div style={{
                      height: 3, borderRadius: 2, marginTop: 5,
                      background: `linear-gradient(90deg, ${
                        item.forming ? COLORS.gold :
                        pct >= 70 ? COLORS.green : pct >= 50 ? COLORS.gold : COLORS.accent
                      } ${pct}%, ${COLORS.border} ${pct}%)`
                    }} />
                  </div>
                </div>
              );
            })}


          </div>
        )}
      </div>
    );
  };

  // ─── Chart tab ───────────────────────────────────────────────────────────────
  const renderChart = () => {
    if (!selectedTicker || chartData.length === 0) {
      return (
        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 12, color: COLORS.textMuted }}>
          <div style={{ fontSize: 36 }}>📈</div>
          <div style={{ fontSize: 14 }}>Select a ticker from the leaderboard to view its chart</div>
        </div>
      );
    }

    const det = selectedDetection;

    // For recharts ComposedChart we represent each candle as a bar using close price
    // and overlay high/low via a custom shape that reads from payload
    const yMin = Math.min(...chartData.map(d => d.low)) * 0.98;
    const yMax = Math.max(...chartData.map(d => d.high)) * 1.02;

    // Sample every N ticks for x-axis labels to avoid crowding
    // On mobile use far fewer labels (4) since screen is narrow
    const totalBars = chartData.length;
    const labelCount = isMobile ? 4 : 8;
    const labelStep = Math.max(1, Math.floor(totalBars / labelCount));

    const xTicks = chartData
      .filter((_, i) => i % labelStep === 0)
      .map(d => d.idx);

    return (
      <div style={{ display: "flex", flexDirection: "column", minHeight: isMobile ? "auto" : "100%" }}>
        {/* Ticker header */}
        <div style={{
          padding: isMobile ? "6px 10px" : "12px 20px", borderBottom: `1px solid ${COLORS.border}`,
          display: "flex", alignItems: "center", gap: isMobile ? 6 : 16, flexShrink: 0,
          background: COLORS.surface, flexWrap: "wrap"
        }}>
          {/* Ticker + prev/next nav */}
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            {filteredScores.length > 1 && (() => {
              const idx = filteredScores.findIndex(s => s.ticker === selectedTicker);
              const hasPrev = idx > 0;
              const hasNext = idx < filteredScores.length - 1;
              const navBtn = (enabled, onClick, label) => (
                <button onClick={onClick} disabled={!enabled} title={label} style={{
                  width: 26, height: 26, borderRadius: 6, border: `1px solid ${COLORS.border}`,
                  background: enabled ? COLORS.surfaceHover : "transparent",
                  color: enabled ? COLORS.textDim : COLORS.textMuted,
                  cursor: enabled ? "pointer" : "default", fontSize: 13,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  opacity: enabled ? 1 : 0.3, flexShrink: 0,
                }}>{label}</button>
              );
              return (<>
                {navBtn(hasPrev, () => setSelectedTicker(filteredScores[idx - 1].ticker), "‹")}
                {navBtn(hasNext, () => setSelectedTicker(filteredScores[idx + 1].ticker), "›")}
                <span style={{ fontSize: 10, color: COLORS.textMuted }}>{idx + 1} / {filteredScores.length}</span>
              </>);
            })()}
            <div>
              <div style={{ fontSize: 20, fontWeight: 800, color: COLORS.accent }}>{selectedTicker}</div>
              <div style={{ fontSize: 11, color: COLORS.textMuted }}>
                {chartData.length} bars · {chartData[0]?.dateStr} – {chartData[chartData.length - 1]?.dateStr}
              </div>
            </div>
          </div>
          {det && (
            <div style={{ display: "flex", gap: isMobile ? 6 : 12,
              flexWrap: "nowrap", overflowX: "auto", WebkitOverflowScrolling: "touch",
              paddingBottom: 2, msOverflowStyle: "none", scrollbarWidth: "none" }}>
              <StatPill label="Score" value={`${(det.score * 100).toFixed(1)}%`} color={COLORS.accent} />
              {/* Cup geometry — only on the Cup & Handle view, cup setups */}
              {chartSubTab === "pattern" && det.setupType !== "rhs" && det.setupType !== "hs" && (
                <>
                  <StatPill label="Arc Depth" value={`${(det.cupDepth * 100).toFixed(1)}%`} color={det.setupType === "rt" ? COLORS.warning : COLORS.cup} />
                  {det.setupType !== "rt" && <StatPill label="Handle" value={`${(det.handleRetrace * 100).toFixed(1)}%`} color={COLORS.handle} />}
                  {det.setupType === "rt" && <StatPill label="Dist Shelf" value={`${(det.handleRetrace * 100).toFixed(1)}%`} color={COLORS.warning} />}
                </>
              )}
              {/* H&S / Rev H&S geometry */}
              {chartSubTab === "pattern" && (det.setupType === "rhs" || det.setupType === "hs") && (
                <>
                  <StatPill label="Head Depth" value={`${(det.headDepth * 100).toFixed(0)}%`}
                    color={det.headDepth >= 0.1 && det.headDepth <= 0.3 ? (det.setupType === "hs" ? COLORS.red : COLORS.green) : det.headDepth <= 0.5 ? COLORS.gold : COLORS.text} />
                  <StatPill label="Shoulder Sym" value={`${(det.shoulderSym * 100).toFixed(0)}%`}
                    color={det.shoulderSym > 0.9 ? (det.setupType === "hs" ? COLORS.red : COLORS.green) : det.shoulderSym > 0.8 ? COLORS.gold : COLORS.text} />
                </>
              )}
              {/* Momentum signals — shown on both views */}
              <StatPill label="Momentum (10d)" value={`${(det.recentMomentum * 100).toFixed(0)}%`} color={det.recentMomentum > 0.6 ? COLORS.green : det.recentMomentum < 0.4 ? COLORS.red : COLORS.textDim} />
              <StatPill label="Breakout" value={`${(det.breakoutProx * 100).toFixed(1)}%`}
                color={det.breakoutProx > 0.8 ? COLORS.green : COLORS.text} />
              {(det.setupType === "rhs" || det.setupType === "hs")
                ? <StatPill label="Vol Surge" value={`${(det.volSurge * 100).toFixed(0)}%`}
                    color={det.volSurge > 0.5 ? (det.setupType === "hs" ? COLORS.red : COLORS.green) : det.volSurge > 0.25 ? COLORS.gold : COLORS.textDim} />
                : <StatPill label="Gradient" value={`${(det.gradConf * 100).toFixed(1)}%`}
                    color={det.gradConf > 0.7 ? COLORS.green : det.gradConf > 0.45 ? COLORS.gold : COLORS.textDim} />
              }
              {(det.setupType !== "rhs" && det.setupType !== "hs") && (
                <StatPill label="Pulse" value={
                  det.handleStreakVal > 0 ? `+${det.handleStreakVal}` :
                  det.handleStreakVal < 0 ? `${det.handleStreakVal}` : "0"
                } color={
                  det.setupType === "rt"
                    ? (det.pulseBonus > 0.6 ? COLORS.red : det.pulseBonus > 0.35 ? COLORS.gold : COLORS.textDim)
                    : (det.pulseBonus > 0.6 ? COLORS.green : det.pulseBonus > 0.35 ? COLORS.gold : COLORS.textDim)
                } />
              )}
              {/* Symmetry — cup / rt pattern view */}
              {chartSubTab === "pattern" && det.setupType !== "rhs" && det.setupType !== "hs" && (
                <>
                  <StatPill label="Area Sym" value={`${(det.areaSymmetry * 100).toFixed(1)}%`}
                    color={det.areaSymmetry > 0.75 ? COLORS.green : det.areaSymmetry > 0.5 ? COLORS.gold : COLORS.textDim} />
                  <StatPill label="Span Sym" value={`${(det.spanSymmetry * 100).toFixed(1)}%`}
                    color={det.spanSymmetry > 0.65 ? COLORS.green : det.spanSymmetry > 0.4 ? COLORS.gold : COLORS.textDim} />
                </>
              )}
              {/* H&S extra geometry */}
              {chartSubTab === "pattern" && (det.setupType === "rhs" || det.setupType === "hs") && (
                <>
                  <StatPill label="Neckline" value={`${(det.necklineScore * 100).toFixed(0)}%`}
                    color={det.necklineScore > 0.7 ? COLORS.green : det.necklineScore > 0.45 ? COLORS.gold : COLORS.textDim} />
                  <StatPill label="Shape" value={`${(det.shapeScore * 100).toFixed(0)}%`}
                    color={det.shapeScore > 0.7 ? COLORS.green : det.shapeScore > 0.45 ? COLORS.gold : COLORS.textDim} />
                </>
              )}
            </div>
          )}
          {!det && (
            <div style={{ fontSize: 12, color: COLORS.warning, padding: "6px 12px", background: "rgba(255,152,0,0.1)", borderRadius: 8, border: `1px solid ${COLORS.warning}` }}>
              No pattern detected with current parameters
            </div>
          )}

          {/* Unified toggle: Cup | Flag | Momentum Gradient.
              Cup/Flag pick the SETUP (and sync the leaderboard ranking);
              Momentum Gradient is the confirmation view. No separate
              "Pattern" button — the setup buttons imply the pattern view. */}
          <div style={{
            display: "flex", gap: 8, alignItems: "center",
            marginLeft: isMobile ? 0 : "auto",
            width: isMobile ? "100%" : "auto",
            overflowX: isMobile ? "auto" : "visible",
            WebkitOverflowScrolling: "touch",
          }}>
            {(() => {
              const row = scores.find(s => s.ticker === selectedTicker);
              const hasCup = !!row?.cup?.detection;
              const hasRHS = !!row?.rhs?.detection;
              const hasHS  = !!row?.hs?.detection;
              const hasRT  = !!row?.rt?.detection;
              // On the pattern view, the active pattern is chartSetup (override)
              // or the leaderboard's active setup. On gradient view, none is "active".
              const activePattern = chartSubTab === "pattern"
                ? (chartSetup || tolerance.activeSetup || "cup")
                : null;

              const setupBtn = (id, label, has, color) => (
                <button
                  key={id}
                  onClick={() => {
                    setChartSubTab("pattern");
                    setChartSetup(id);
                    handleSetupChange(id, has);
                  }}
                  style={{
                    padding: isMobile ? "5px 8px" : "11px 16px", borderRadius: 9,
                    fontSize: isMobile ? 10 : 13, fontWeight: 700,
                    border: `1px solid ${activePattern === id ? color : COLORS.border}`,
                    background: activePattern === id ? `${color}22` : COLORS.surfaceHover,
                    color: activePattern === id ? color : (has ? COLORS.text : COLORS.textDim),
                    cursor: "pointer", whiteSpace: "nowrap", flexShrink: 0,
                  }}
                >
                  {label}
                </button>
              );

              return (
                <>
                  {setupBtn("cup", "⌣ Cup",        hasCup, COLORS.cup)}
                  {setupBtn("rhs", "⋎ Rev H&S",    hasRHS, COLORS.green)}
                  {setupBtn("hs",  "⋏ H&S",        hasHS,  COLORS.red)}
                  {setupBtn("rt",  "⌢ Rnd Top",    hasRT,  COLORS.warning)}
                  <button
                    onClick={() => setChartSubTab("gradient")}
                    style={{
                      padding: isMobile ? "5px 8px" : "11px 16px", borderRadius: 9,
                      fontSize: isMobile ? 10 : 13, fontWeight: 700, flexShrink: 0,
                      border: `1px solid ${chartSubTab === "gradient" ? COLORS.accent : COLORS.border}`,
                      background: chartSubTab === "gradient" ? COLORS.accentDim : COLORS.surfaceHover,
                      color: chartSubTab === "gradient" ? COLORS.accent : COLORS.text,
                      cursor: "pointer", whiteSpace: "nowrap",
                    }}
                  >
                    ≈ Gradient
                  </button>
                </>
              );
            })()}
          </div>
        </div>

        <div style={{ padding: chartSubTab === "gradient" ? "8px 0 0" : "16px 12px 8px", display: "flex", flexDirection: "column", gap: 12 }}>

          {/* ── Momentum Gradient view: canvas-based CandlePulse chart ── */}
          {chartSubTab === "gradient" && (
            <>
              {/* Legend row — trend-code swatches + gradient scale */}
              <div style={{
                padding: "10px 20px", display: "flex", gap: 24, alignItems: "center",
                flexShrink: 0, flexWrap: "wrap",
                borderBottom: `1px solid ${COLORS.border}`, background: COLORS.surface,
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", rowGap: 6 }}>
                  <span style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", color: COLORS.textDim }}>
                    Price trend
                  </span>
                  {[["rgba(22,128,52,0.9)","Strong bull"],["rgba(56,168,82,0.9)","Bullish"],["rgba(120,200,110,0.9)","Weak bull"],
                    ["rgba(214,188,60,0.8)","Neutral"],
                    ["rgba(238,140,120,0.9)","Weak bear"],["rgba(220,72,60,0.9)","Bearish"],["rgba(176,20,20,0.9)","Strong bear"]].map(([c,l]) => (
                    <div key={l} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                      <div style={{ width: 13, height: 13, borderRadius: 3, background: c, border: "1px solid rgba(255,255,255,0.15)", flexShrink: 0 }} />
                      <span style={{ fontSize: 11, color: COLORS.textDim, whiteSpace: "nowrap" }}>{l}</span>
                    </div>
                  ))}
                </div>
                <div style={{ marginLeft: isMobile ? 0 : "auto", display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                  <span style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", color: COLORS.textDim }}>
                    Momentum
                  </span>
                  <span style={{ fontSize: 11, color: "#b30000", fontWeight: 700 }}>Bear</span>
                  <div style={{ width: 120, height: 12, borderRadius: 4, border: "1px solid rgba(255,255,255,0.15)", flexShrink: 0,
                    background: "linear-gradient(90deg,#5b0000,#b30000,#ff6600,#ffd700,#bfff00,#66cc66,#006400)" }} />
                  <span style={{ fontSize: 11, color: "#1b8f5a", fontWeight: 700 }}>Bull</span>
                </div>
              </div>
              {/* Canvas chart fills remainder */}
              <div style={{ height: isMobile ? 360 : 420, padding: "12px 16px 8px", boxSizing: "border-box" }}>
                <PulseChart data={chartData} />
              </div>
            </>
          )}

          {/* ── Cup & Handle view: Recharts ── */}
          {chartSubTab === "pattern" && (
            <div key={`pattern-${selectedTicker}`} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <div style={{ height: isMobile ? "45vh" : "52vh", minHeight: isMobile ? 280 : 340 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={chartData} margin={{ top: 8, right: 16, bottom: 0, left: 8 }}>
                    <CartesianGrid stroke={COLORS.border} strokeDasharray="2 4" vertical={false} />
                    <XAxis
                      dataKey="idx"
                      type="number"
                      domain={["dataMin", "dataMax"]}
                      ticks={xTicks}
                      tickFormatter={i => chartData[i]?.dateStr || ""}
                      tick={{ fill: COLORS.textMuted, fontSize: 10 }}
                      axisLine={{ stroke: COLORS.border }}
                      tickLine={false}
                    />
                    <YAxis
                      domain={[yMin, yMax]}
                      tick={{ fill: COLORS.textMuted, fontSize: 10 }}
                      axisLine={{ stroke: COLORS.border }}
                      tickLine={false}
                      tickFormatter={v => v.toFixed(0)}
                      width={48}
                    />
                    <Tooltip content={<ChartTooltip />} />
                    <Bar
                      dataKey="close"
                      shape={(props) => {
                        const { x, y, width, height, payload } = props;
                        if (!payload) return null;
                        const isUp = payload.close >= payload.open;
                        const color = isUp ? COLORS.green : COLORS.red;
                        return (
                          <g>
                            <rect
                              x={(x || 0) + 1} y={y || 0}
                              width={Math.max(1, (width || 4) - 2)}
                              height={Math.abs(height || 1)}
                              fill={isUp ? color : "transparent"}
                              stroke={color} strokeWidth={1}
                            />
                          </g>
                        );
                      }}
                      isAnimationActive={false}
                    >
                      {chartData.map((d, i) => (
                        <Cell key={i} fill={d.close >= d.open ? COLORS.green : COLORS.red} />
                      ))}
                    </Bar>
                    {/* Cup / Rounded Top: single ideal ghost curve */}
                    {(det?.setupType !== "rhs" && det?.setupType !== "hs") && (
                      <Line
                        dataKey="ghost"
                        stroke={det?.setupType === "rt" ? COLORS.warning : COLORS.ghost}
                        strokeWidth={2.5}
                        dot={false}
                        strokeDasharray="6 4"
                        isAnimationActive={false}
                        connectNulls={false}
                      />
                    )}
                    {/* Reverse H&S / H&S: outline tracing the swings */}
                    {(det?.setupType === "rhs" || det?.setupType === "hs") && (
                      <Line
                        dataKey="rhsShape"
                        stroke={det?.setupType === "hs" ? COLORS.red : "#a855f7"}
                        strokeWidth={2.5}
                        dot={false}
                        isAnimationActive={false}
                        connectNulls
                      />
                    )}
                    {/* Neckline for rhs/hs */}
                    {(det?.setupType === "rhs" || det?.setupType === "hs") && det.necklineLeftPrice != null && (
                      <ReferenceLine
                        segment={[
                          { x: det.necklineLeftIdx,  y: det.necklineLeftPrice },
                          { x: det.necklineRightIdx, y: det.necklineRightPrice },
                        ]}
                        stroke={det?.setupType === "hs" ? COLORS.red : COLORS.green}
                        strokeWidth={2}
                        strokeDasharray="6 4"
                        label={isMobile ? undefined : { value: "Neckline", fill: det?.setupType === "hs" ? COLORS.red : COLORS.green, fontSize: 10, position: "insideTopLeft" }}
                      />
                    )}
                    {/* Reference lines — keyLevels for rhs/hs, named indices for cup/rt. */}
                    {det && ((det.setupType === "rhs" || det.setupType === "hs")
                      ? (det.keyLevels || [])
                      : det.setupType === "rt"
                        ? [
                          { idx: det.leftRim,      label: "L Rim",  color: COLORS.warning },
                          { idx: det.cupBottom,    label: "Top",    color: COLORS.red },
                          { idx: det.rightRim,     label: "R Rim",  color: COLORS.warning },
                          { idx: det.handleMinIdx, label: "Bounce", color: COLORS.gold },
                        ]
                        : [
                          { idx: det.leftRim,      label: "L Rim",  color: COLORS.cup },
                          { idx: det.cupBottom,    label: "Bottom", color: COLORS.accent },
                          { idx: det.rightRim,     label: "R Rim",  color: COLORS.cup },
                          { idx: det.handleMinIdx, label: "Handle", color: COLORS.handle },
                        ]
                    ).map(({ idx, label, color }, i) => (
                      <ReferenceLine
                        key={label} x={idx}
                        stroke={color} strokeDasharray="4 3" strokeWidth={1.5}
                        label={isMobile ? undefined : {
                          value: label, fill: color, fontSize: 10,
                          position: i % 2 === 0 ? "insideTopRight" : "insideBottomRight"
                        }}
                      />
                    ))}
                  </ComposedChart>
                </ResponsiveContainer>
              </div>

              {/* Mobile-only label legend (replaces overlapping in-chart labels) */}
              {isMobile && det && (
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap", padding: "2px 8px" }}>
                  {((det.setupType === "rhs" || det.setupType === "hs")
                    ? (det.keyLevels || [])
                    : det.setupType === "rt"
                      ? [
                          { label: "L Rim",  color: COLORS.warning },
                          { label: "Top",    color: COLORS.red },
                          { label: "R Rim",  color: COLORS.warning },
                          { label: "Bounce", color: COLORS.gold },
                        ]
                      : [
                          { label: "L Rim",  color: COLORS.cup },
                          { label: "Bottom", color: COLORS.accent },
                          { label: "R Rim",  color: COLORS.cup },
                          { label: "Handle", color: COLORS.handle },
                        ]
                  ).map(({ label, color }) => (
                    <div key={label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <div style={{ width: 2, height: 12, background: color, borderRadius: 1, opacity: 0.9 }} />
                      <span style={{ fontSize: 10, color, fontWeight: 600 }}>{label}</span>
                    </div>
                  ))}
                  {(det.setupType === "rhs" || det.setupType === "hs") && (
                    <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <div style={{ width: 12, height: 2, background: det.setupType === "hs" ? COLORS.red : COLORS.green, borderRadius: 1 }} />
                      <span style={{ fontSize: 10, color: det.setupType === "hs" ? COLORS.red : COLORS.green, fontWeight: 600 }}>Neckline</span>
                    </div>
                  )}
                </div>
              )}

              {/* Volume bars */}
              <div style={{ height: 80 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={chartData} margin={{ top: 0, right: 16, bottom: 0, left: 8 }}>
                    <XAxis dataKey="idx" hide />
                    <YAxis
                      tick={{ fill: COLORS.textMuted, fontSize: 9 }}
                      axisLine={false} tickLine={false}
                      tickFormatter={v => v >= 1e6 ? (v / 1e6).toFixed(0) + "M" : v >= 1e3 ? (v / 1e3).toFixed(0) + "K" : v}
                      width={48}
                    />
                    <Bar dataKey="volume" isAnimationActive={false}>
                      {chartData.map((d, i) => (
                        <Cell key={i} fill={`${d.close >= d.open ? COLORS.green : COLORS.red}88`} />
                      ))}
                    </Bar>
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>

        {/* Metric cards — radar only for pattern views; gradient view gets stat strip */}
        {det && chartSubTab !== "gradient" && radarData.length > 0 && (
          <div style={{
            borderTop: `1px solid ${COLORS.border}`, padding: "12px 16px",
            display: "flex", alignItems: isMobile ? "stretch" : "center", gap: 18, flexShrink: 0,
            background: COLORS.surface, flexDirection: isMobile ? "column" : "row"
          }}>
            <div style={{ width: isMobile ? "100%" : 210, height: isMobile ? 220 : 190, flexShrink: 0, display: "flex", justifyContent: "center" }}>
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData} outerRadius={isMobile ? 62 : 58} margin={{ top: 18, right: 22, bottom: 18, left: 22 }}>
                  <PolarGrid stroke={COLORS.border} />
                  <PolarAngleAxis
                    dataKey="metric"
                    tick={(props) => {
                      const { x, y, payload, cx, cy } = props;
                      // Shorten labels on mobile to prevent clipping
                      let label = payload.value;
                      if (isMobile && label.length > 8) label = label.slice(0, 7) + "…";
                      const dx = x - cx, dy = y - cy;
                      const angle = Math.atan2(dy, dx) * 180 / Math.PI;
                      const anchor = Math.abs(angle) > 90 ? "end" : "start";
                      // Top/bottom labels centered
                      const textAnchor = Math.abs(dx) < 10 ? "middle" : anchor;
                      return (
                        <text x={x} y={y} textAnchor={textAnchor} dominantBaseline="middle"
                          style={{ fill: COLORS.textDim, fontSize: isMobile ? 9 : 10, fontWeight: 600 }}>
                          {label}
                        </text>
                      );
                    }}
                  />
                  <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                  <Radar dataKey="value" stroke={COLORS.accent} fill={COLORS.accent} fillOpacity={0.28} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", color: COLORS.textDim, marginBottom: 8 }}>
                {det?.setupType === "rhs" ? "Reverse H&S structure"
                  : det?.setupType === "hs" ? "Head & Shoulders structure"
                  : det?.setupType === "rt" ? "Rounded Top structure"
                  : "Cup & Handle structure"}
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {radarData.map(({ metric, value }, idx) => (
                  <div key={idx} style={{ flex: "1 1 90px", minWidth: 80, background: COLORS.surfaceHover, border: `1px solid ${COLORS.border}`, borderRadius: 8, padding: "7px 10px" }}>
                    <div style={{ fontSize: 10, color: COLORS.textDim, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.3px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{metric}</div>
                    <div style={{ fontSize: 17, fontWeight: 800, marginTop: 2, color: value >= 70 ? COLORS.green : value >= 50 ? COLORS.gold : COLORS.text }}>{value}</div>
                    <div style={{ height: 3, borderRadius: 2, marginTop: 4, background: `linear-gradient(90deg, ${value >= 70 ? COLORS.green : value >= 50 ? COLORS.gold : COLORS.accent} ${value}%, ${COLORS.border} ${value}%)` }} />
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
        {/* Gradient view — show current momentum stats, works for all setup types */}
        {det && chartSubTab === "gradient" && (() => {
          const isHSFamily = det.setupType === "hs" || det.setupType === "rt";
          // Pull from radar first (always populated), then direct fields as fallback
          // Read the last bar's rolling 20-bar gradient from chartData — same source as the strip
          const lastBarGrad = chartData.length > 0 ? (chartData[chartData.length - 1].gradientScore ?? 0) : 0;
          const gradVal  = Math.round(((lastBarGrad + 1) / 2) * 100); // remap −1..1 → 0..100
          const momVal   = Math.round((det.recentMomentum ?? 0.5) * 100);
          const brkVal   = Math.round((det.breakoutProx ?? det.radar?.breakoutProx ?? 0) * 100);
          const volVal   = Math.round((det.radar?.volumeConf ?? det.volConf ?? det.volScore ?? 0) * 100);
          // Pulse: cup/RT have pulseBonus; H&S/RHS use volSurge as the confirmation signal
          const pulseVal = Math.round(((det.setupType === "hs" || det.setupType === "rhs")
            ? (det.radar?.gradConf ?? det.volSurge ?? 0)
            : (det.pulseBonus ?? det.radar?.pulseStr ?? 0)) * 100);
          const stats = [
            { label: "Gradient",        value: gradVal, desc: "20-bar momentum gradient — how cleanly price is trending" },
            { label: "Momentum (10d)",  value: momVal,  desc: "10-bar candle momentum (bull vs bear ratio)" },
            { label: isHSFamily ? "Brkdn Prox" : "Breakout Prox",
                                        value: brkVal,  desc: isHSFamily ? "Closeness to neckline breakdown level" : "Closeness to breakout pivot level" },
            { label: "Volume",          value: volVal,  desc: "Volume confirmation signal" },
            { label: isHSFamily ? "Vol Surge" : "Pulse / Streak",
                                        value: pulseVal, desc: isHSFamily ? "Volume surge on breakdown" : "Consecutive bullish candle streak bonus" },
          ];
          return (
            <div style={{ borderTop: `1px solid ${COLORS.border}`, padding: "14px 16px", background: COLORS.surface, flexShrink: 0 }}>
              <div style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", color: COLORS.textDim, marginBottom: 10 }}>
                Current Momentum Readings
              </div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {stats.map(({ label, value, desc }) => {
                  const color = value >= 70 ? COLORS.green : value >= 50 ? COLORS.gold : value >= 30 ? COLORS.textDim : COLORS.red;
                  return (
                    <div key={label} title={desc} style={{ flex: "1 1 100px", minWidth: 90, background: COLORS.surfaceHover, border: `1px solid ${COLORS.border}`, borderRadius: 10, padding: "10px 12px" }}>
                      <div style={{ fontSize: 10, color: COLORS.textDim, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.4px", marginBottom: 4 }}>{label}</div>
                      <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
                        <span style={{ fontSize: 22, fontWeight: 900, color, lineHeight: 1 }}>{value}</span>
                        <span style={{ fontSize: 11, color: COLORS.textMuted }}>/100</span>
                      </div>
                      <div style={{ height: 4, borderRadius: 2, marginTop: 6, background: COLORS.border, overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${value}%`, background: color, borderRadius: 2, transition: "width 0.4s" }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })()}
      </div>
    );
  };

  // ─── Chat tab ─────────────────────────────────────────────────────────────────
  // ─── Heatmap tab ──────────────────────────────────────────────────────────────
  const renderHeatmap = () => {
    const isEmpty = !rawData;
    const noResults = rawData && hmData.length === 0;

    // Tile renderer — shared between swim-lane and flat view
    const renderTile = (d) => {
      const score = d.total;
      const bg = pulseColor(score, 0.85);
      const border = pulseColor(score, 1);
      const isHovered = hmHover === d.ticker;
      const isPinned = pinnedTicker === d.ticker;
      const pctLabel = `${score >= 0 ? "+" : ""}${(score * 100).toFixed(0)}`;
      const sparkSlice = d.signals.slice(-20);
      return (
        <div
          key={d.ticker}
          onMouseEnter={() => { if (!pinnedTicker) setHmHover(d.ticker); }}
          onMouseLeave={() => { if (!pinnedTicker) setHmHover(null); }}
          onClick={() => {
            if (isPinned) {
              setPinnedTicker(null);
              setHmHover(null);
            } else {
              setPinnedTicker(d.ticker);
              setHmHover(d.ticker);
            }
          }}
          style={{
            background: bg,
            border: `1px solid ${isPinned ? COLORS.gold : isHovered ? COLORS.text : border}`,
            borderRadius: 8, padding: "8px 8px 6px",
            cursor: "pointer",
            transform: isHovered || isPinned ? "scale(1.04)" : "scale(1)",
            transition: "transform 0.12s, border-color 0.12s",
            boxShadow: isPinned ? `0 4px 20px ${COLORS.gold}44` : isHovered ? `0 4px 16px rgba(0,0,0,0.4)` : "none",
            userSelect: "none",
            outline: isPinned ? `2px solid ${COLORS.gold}` : "none",
          }}
        >
          <div style={{
            fontSize: 11, fontWeight: 800, color: "#fff",
            letterSpacing: "0.3px", lineHeight: 1, marginBottom: 4,
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap"
          }}>{d.ticker}</div>
          <div style={{
            fontSize: 15, fontWeight: 900, color: "#fff",
            lineHeight: 1, marginBottom: 5,
            textShadow: "0 1px 3px rgba(0,0,0,0.4)"
          }}>{pctLabel}</div>
          <svg width="100%" height={16} style={{ display: "block" }}>
            {sparkSlice.map((s, i) => {
              const x = (i / (sparkSlice.length - 1 || 1)) * 92;
              const color = s === 1 ? "rgba(255,255,255,0.9)" : s === -1 ? "rgba(255,100,100,0.9)" : "rgba(255,255,255,0.3)";
              return <rect key={i} x={x} y={s === 1 ? 2 : s === -1 ? 10 : 6} width={3} height={s === 0 ? 4 : 6} fill={color} rx={1} />;
            })}
          </svg>
          {d.streakVal !== 0 && (
            <div style={{
              marginTop: 4, fontSize: 9, fontWeight: 700,
              color: d.streakVal > 0 ? "rgba(255,255,255,0.95)" : "rgba(255,180,180,0.95)",
              letterSpacing: "0.3px"
            }}>
              {d.streakVal > 0 ? `▲ +${d.streakVal}` : `▼ ${d.streakVal}`}
            </div>
          )}
        </div>
      );
    };

    // Sectors present in current data
    const activeSectors = ["All", ...hmBySector.map(([s]) => s)];

    // Data to render: filtered by hmSectorFilter
    const visibleSectors = hmSectorFilter === "All"
      ? hmBySector
      : hmBySector.filter(([s]) => s === hmSectorFilter);

    return (
      <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>

        {/* Controls bar */}
        <div style={{
          padding: "10px 16px", borderBottom: `1px solid ${COLORS.border}`,
          display: "flex", gap: 10, alignItems: "center", flexShrink: 0,
          background: COLORS.surface, flexWrap: "wrap"
        }}>
          <input
            style={{ ...S.input, maxWidth: 140 }}
            placeholder="Filter ticker…"
            value={hmSearch}
            onChange={e => setHmSearch(e.target.value)}
          />
          <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <span style={{ fontSize: 10, color: COLORS.textMuted, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.5px", whiteSpace: "nowrap" }}>
              Window
            </span>
            {[30, 60, 90].map(w => (
              <button
                key={w}
                onClick={() => setHmWindow(w)}
                style={{
                  fontSize: 11, fontWeight: hmWindow === w ? 700 : 500,
                  padding: "4px 10px", borderRadius: 20, cursor: "pointer",
                  border: `1px solid ${hmWindow === w ? COLORS.accent : COLORS.border}`,
                  background: hmWindow === w ? "rgba(108,143,255,0.15)" : "transparent",
                  color: hmWindow === w ? COLORS.accent : COLORS.textDim,
                  transition: "all 0.15s",
                }}
              >{w}b</button>
            ))}
          </div>

          {/* Legend */}
          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
            {/* Sort controls */}
            <div style={{ display: "flex", gap: 3, alignItems: "center" }}>
              <span style={{ fontSize: 10, color: COLORS.textMuted, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.5px" }}>Sort</span>
              {[
                { id: "score", label: "Score" },
                { id: "alpha", label: "A–Z" },
                { id: "streak", label: "Streak" },
              ].map(({ id, label }) => (
                <button
                  key={id}
                  onClick={() => setHmSort(id)}
                  style={{
                    fontSize: 10, fontWeight: hmSort === id ? 700 : 500,
                    padding: "3px 8px", borderRadius: 12, cursor: "pointer",
                    border: `1px solid ${hmSort === id ? COLORS.accent : COLORS.border}`,
                    background: hmSort === id ? "rgba(108,143,255,0.15)" : "transparent",
                    color: hmSort === id ? COLORS.accent : COLORS.textDim,
                    transition: "all 0.15s",
                  }}
                >{label}</button>
              ))}
            </div>
            {/* Heatmap CSV export */}
            <button
              style={{ ...S.btn("secondary", !hmData.length), fontSize: 10, padding: "4px 10px" }}
              onClick={handleExportHeatmap}
              disabled={!hmData.length}
              title="Export heatmap pulse data as CSV"
            >
              ↓ CSV
            </button>
            <span style={{ fontSize: 10, color: COLORS.red, fontWeight: 600 }}>◀ Bear</span>
            <div style={{
              width: 80, height: 8, borderRadius: 4,
              background: "linear-gradient(90deg, #ef5350, #2a2f45, #26a69a)"
            }} />
            <span style={{ fontSize: 10, color: COLORS.green, fontWeight: 600 }}>Bull ▶</span>
          </div>
        </div>

        {/* Sector leaderboard / filter strip */}
        {hmBySector.length > 0 && (
          <div style={{
            padding: "8px 16px", borderBottom: `1px solid ${COLORS.border}`,
            display: "flex", gap: 6, alignItems: "center", flexShrink: 0,
            background: COLORS.bg, overflowX: "auto", flexWrap: "nowrap",
          }}>
            <span style={{ fontSize: 10, color: COLORS.textMuted, fontWeight: 600, letterSpacing: "0.5px", textTransform: "uppercase", marginRight: 4, whiteSpace: "nowrap", flexShrink: 0 }}>
              Sector
            </span>
            {activeSectors.map(s => {
              const isAll = s === "All";
              const active = hmSectorFilter === s;
              const sectorTickers = isAll ? hmData : (hmBySector.find(([n]) => n === s)?.[1] || []);
              const avg = isAll ? null : sectorTickers.reduce((acc, d) => acc + d.total, 0) / (sectorTickers.length || 1);
              const avgColor = avg == null ? COLORS.textDim : pulseColor(avg, 1);
              return (
                <button
                  key={s}
                  onClick={() => setHmSectorFilter(s)}
                  style={{
                    fontSize: 10, fontWeight: active ? 700 : 500,
                    padding: "3px 9px", borderRadius: 20, cursor: "pointer", flexShrink: 0,
                    border: `1px solid ${active ? (isAll ? COLORS.accent : avgColor) : COLORS.border}`,
                    background: active ? (isAll ? "rgba(108,143,255,0.15)" : `${avgColor}22`) : "transparent",
                    color: active ? (isAll ? COLORS.accent : avgColor) : COLORS.textDim,
                    transition: "all 0.15s", display: "flex", alignItems: "center", gap: 5,
                  }}
                >
                  {!isAll && (
                    <span style={{
                      display: "inline-block", width: 6, height: 6, borderRadius: "50%",
                      background: avgColor, flexShrink: 0,
                    }} />
                  )}
                  {isAll ? "All Sectors" : s}
                  {!isAll && (
                    <span style={{ color: avgColor, fontWeight: 700, fontSize: 9 }}>
                      {avg >= 0 ? "+" : ""}{(avg * 100).toFixed(0)}
                    </span>
                  )}
                  <span style={{ color: COLORS.textMuted, fontSize: 9 }}>
                    ({sectorTickers.length})
                  </span>
                </button>
              );
            })}
          </div>
        )}

        {/* Score formula bar */}
        <div style={{
          padding: "5px 16px", borderBottom: `1px solid ${COLORS.border}`,
          display: "flex", gap: 12, alignItems: "center", flexShrink: 0,
          background: COLORS.bg, fontSize: 10, color: COLORS.textMuted
        }}>
          <span style={{ fontWeight: 600, color: COLORS.textDim }}>Score = </span>
          <span><span style={{ color: COLORS.accent, fontWeight: 600 }}>Gradient</span> × 40%</span>
          <span>+</span>
          <span><span style={{ color: COLORS.gold, fontWeight: 600 }}>Streak</span> × 35%</span>
          <span>+</span>
          <span><span style={{ color: COLORS.green, fontWeight: 600 }}>Volume</span> × 25%</span>
          {hmData.length > 0 && (
            <span style={{ marginLeft: "auto" }}>
              {hmData.length} tickers · {hmBySector.length} sector{hmBySector.length !== 1 ? "s" : ""}
            </span>
          )}
        </div>

        {/* Empty states */}
        {isEmpty && (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12, color: COLORS.textMuted }}>
            <div style={{ fontSize: 40 }}>🌡️</div>
            <div style={{ fontSize: 14, fontWeight: 600 }}>Upload a CSV to generate the pulse heatmap</div>
            <div style={{ fontSize: 12, maxWidth: 300, textAlign: "center", lineHeight: 1.6 }}>
              Tickers are grouped by GICS sector with momentum-ranked swim lanes.
            </div>
          </div>
        )}
        {noResults && (
          <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: COLORS.textMuted }}>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 28, marginBottom: 8 }}>🔍</div>
              <div>No tickers match — try a shorter window or clear the filter</div>
            </div>
          </div>
        )}

        {/* Heatmap: sector swim lanes */}
        {hmData.length > 0 && (
          <div style={{ flex: 1, overflowY: "auto", padding: "12px 16px 24px" }}>

            {/* Hover / pinned tooltip */}
            {(hmHover || pinnedTicker) && (() => {
              const activeTicker = pinnedTicker || hmHover;
              const d = hmData.find(x => x.ticker === activeTicker);
              if (!d) return null;
              const scoreColor = pulseColor(d.total);
              const isPinned = !!pinnedTicker;
              return (
                <div style={isMobile ? {
                  position: "fixed", bottom: 0, left: 0, right: 0, top: "auto", zIndex: 1000,
                  background: COLORS.surface, border: `2px solid ${isPinned ? COLORS.gold : COLORS.border}`,
                  borderRadius: "16px 16px 0 0", padding: "14px 16px", maxHeight: "70vh", overflowY: "auto",
                  boxShadow: isPinned ? `0 8px 32px ${COLORS.gold}33` : "0 -8px 32px rgba(0,0,0,0.6)",
                } : {
                  position: "fixed", top: 80, right: 24, zIndex: 1000,
                  background: COLORS.surface, border: `2px solid ${isPinned ? COLORS.gold : COLORS.border}`,
                  borderRadius: 12, padding: "14px 16px", minWidth: 220,
                  boxShadow: isPinned ? `0 8px 32px ${COLORS.gold}33` : "0 8px 32px rgba(0,0,0,0.6)",
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                    <div style={{ width: 10, height: 10, borderRadius: "50%", background: scoreColor, flexShrink: 0 }} />
                    <span style={{ fontWeight: 800, fontSize: 16, color: COLORS.text }}>{d.ticker}</span>
                    <span style={{ fontSize: 9, color: COLORS.textMuted, marginLeft: 2 }}>{d.sector}</span>
                    <span style={{ marginLeft: "auto", fontWeight: 700, fontSize: 14, color: scoreColor }}>
                      {d.total >= 0 ? "+" : ""}{(d.total * 100).toFixed(0)}
                    </span>
                    {isPinned && (
                      <button
                        style={{ ...S.btn("ghost"), fontSize: 11, padding: "0 4px", color: COLORS.gold }}
                        onClick={() => { setPinnedTicker(null); setHmHover(null); }}
                        title="Unpin tooltip"
                      >📌</button>
                    )}
                  </div>
                  {isPinned && (
                    <div style={{ fontSize: 9, color: COLORS.gold, marginBottom: 8, opacity: 0.8 }}>
                      📌 Pinned — click tile again to unpin
                    </div>
                  )}
                  {[
                    { label: "Gradient Arc", val: d.gradient, color: COLORS.accent, pct: 40 },
                    { label: "Candle Streak", val: d.streak, color: COLORS.gold, pct: 35 },
                    { label: "Volume Conf.", val: d.volume > 1 ? Math.min(0.5,(d.volume-1)*0.5) : Math.max(-0.5,(d.volume-1)*0.5), color: COLORS.green, pct: 25, raw: `${d.volume.toFixed(2)}×` },
                  ].map(({ label, val, color, pct, raw }) => (
                    <div key={label} style={{ marginBottom: 8 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                        <span style={{ fontSize: 10, color: COLORS.textMuted }}>{label} ×{pct}%</span>
                        <span style={{ fontSize: 10, fontWeight: 700, color }}>{raw || `${val >= 0 ? "+" : ""}${(val * 100).toFixed(0)}`}</span>
                      </div>
                      <div style={{ height: 4, background: COLORS.border, borderRadius: 2, overflow: "hidden" }}>
                        <div style={{
                          height: "100%", borderRadius: 2,
                          width: `${Math.abs(val) * 100}%`,
                          marginLeft: val < 0 ? `${(1 - Math.abs(val)) * 100}%` : 0,
                          background: val >= 0 ? color : COLORS.red,
                        }} />
                      </div>
                    </div>
                  ))}
                  <div style={{ marginTop: 10, paddingTop: 8, borderTop: `1px solid ${COLORS.border}`, fontSize: 11, color: COLORS.textDim }}>
                    Streak: <span style={{ color: d.streakVal > 0 ? COLORS.green : d.streakVal < 0 ? COLORS.red : COLORS.textMuted, fontWeight: 700 }}>
                      {d.streakVal > 0 ? `+${d.streakVal}` : d.streakVal} bars
                    </span>
                    <span style={{ float: "right" }}>
                      Vol: <span style={{ color: d.volume > 1.3 ? COLORS.green : d.volume < 0.7 ? COLORS.red : COLORS.textMuted, fontWeight: 700 }}>{d.volume.toFixed(2)}×</span>
                    </span>
                  </div>
                  <div style={{ marginTop: 10 }}>
                    <div style={{ fontSize: 10, color: COLORS.textMuted, marginBottom: 4 }}>Zone gradient (early → recent)</div>
                    <div style={{ display: "flex", gap: 3 }}>
                      {d.zones.map((z, i) => (
                        <div key={i} style={{ flex: 1, textAlign: "center" }}>
                          <div style={{ height: 20, borderRadius: 3, background: pulseColor(z), marginBottom: 2 }} />
                          <div style={{ fontSize: 9, color: COLORS.textMuted }}>Z{i+1}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div style={{ marginTop: 10 }}>
                    <div style={{ fontSize: 10, color: COLORS.textMuted, marginBottom: 4 }}>Per-bar signals</div>
                    <svg width="100%" height={28} viewBox="0 0 188 28" preserveAspectRatio="none" style={{ display: "block" }}>
                      {d.signals.map((s, i) => {
                        const x = (i / (d.signals.length - 1)) * 188;
                        const color = s === 1 ? COLORS.green : s === -1 ? COLORS.red : COLORS.border;
                        return <rect key={i} x={x} y={s === 1 ? 4 : s === -1 ? 16 : 12} width={2} height={s === 0 ? 4 : 8} fill={color} rx={1} />;
                      })}
                    </svg>
                  </div>
                  <button
                    style={{ ...S.btn("ghost"), marginTop: 8, fontSize: 10, width: "100%", color: COLORS.accent }}
                    onClick={() => { setSelectedTicker(d.ticker); setActiveTab("chart"); }}
                  >
                    View chart →
                  </button>
                </div>
              );
            })()}

            {/* Swim lanes */}
            {visibleSectors.map(([sectorName, tickers]) => {
              const avg = tickers.reduce((acc, d) => acc + d.total, 0) / (tickers.length || 1);
              const avgColor = pulseColor(avg, 1);
              const avgPct = `${avg >= 0 ? "+" : ""}${(avg * 100).toFixed(0)}`;
              const bullCount = tickers.filter(d => d.total > 0.1).length;
              const bearCount = tickers.filter(d => d.total < -0.1).length;
              return (
                <div key={sectorName} style={{ marginBottom: 20 }}>
                  {/* Sector header */}
                  <div style={{
                    display: "flex", alignItems: "center", gap: 10,
                    marginBottom: 8, padding: "6px 0",
                    borderBottom: `2px solid ${avgColor}44`,
                  }}>
                    {/* Color swatch */}
                    <div style={{
                      width: 12, height: 12, borderRadius: 3, flexShrink: 0,
                      background: avgColor, boxShadow: `0 0 6px ${avgColor}66`,
                    }} />
                    <span style={{ fontWeight: 800, fontSize: 13, color: COLORS.text }}>{sectorName}</span>
                    <span style={{
                      fontSize: 12, fontWeight: 700, color: avgColor,
                      padding: "1px 7px", borderRadius: 10,
                      background: `${avgColor}18`, border: `1px solid ${avgColor}44`,
                    }}>{avgPct}</span>
                    <span style={{ fontSize: 10, color: COLORS.textMuted }}>
                      avg · {tickers.length} ticker{tickers.length !== 1 ? "s" : ""}
                    </span>
                    {bullCount > 0 && (
                      <span style={{ fontSize: 10, color: COLORS.green }}>▲ {bullCount} bull</span>
                    )}
                    {bearCount > 0 && (
                      <span style={{ fontSize: 10, color: COLORS.red }}>▼ {bearCount} bear</span>
                    )}
                    {/* Mini sector pulse bar */}
                    <div style={{ flex: 1, height: 4, borderRadius: 2, background: COLORS.border, overflow: "hidden", maxWidth: 120, marginLeft: "auto" }}>
                      <div style={{
                        height: "100%", borderRadius: 2,
                        width: `${Math.abs(avg) * 100}%`,
                        marginLeft: avg < 0 ? `${(1 - Math.abs(avg)) * 100}%` : "50%",
                        background: avgColor,
                      }} />
                    </div>
                  </div>

                  {/* Tile grid for this sector */}
                  <div style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fill, minmax(100px, 1fr))",
                    gap: 5,
                  }}>
                    {[...tickers].sort((a, b) => {
                      if (hmSort === "alpha") return a.ticker.localeCompare(b.ticker);
                      if (hmSort === "streak") return b.streakVal - a.streakVal;
                      return b.total - a.total; // default: score
                    }).map(d => renderTile(d))}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  // ─── Warnings panel ──────────────────────────────────────────────────────────
  const renderWarnings = () => {
    if (!showWarnings || parseWarnings.length === 0) return null;
    return (
      <div style={{
        ...S.card, marginTop: 0, borderColor: COLORS.warning,
        background: "rgba(255,152,0,0.06)", maxHeight: 160, overflowY: "auto"
      }}>
        <div style={{ ...S.sectionTitle, color: COLORS.warning }}>
          Parse Warnings ({parseWarnings.length})
          <button
            style={{ ...S.btn("ghost"), fontSize: 10, padding: "0 6px", float: "right" }}
            onClick={() => setShowWarnings(false)}
          >✕</button>
        </div>
        {parseWarnings.map((w, i) => (
          <div key={i} style={{ fontSize: 11, color: COLORS.textDim, padding: "2px 0", lineHeight: 1.5 }}>
            • {w}
          </div>
        ))}
      </div>
    );
  };

  // ── Domain Intelligence Tab ──────────────────────────────────────────────────
  const renderDomainIntelligence = () => {
    const noTicker = !selectedTicker;
    const noDet = selectedTicker && !selectedDetection;
    const sectorPulse = hmData.find(d => d.ticker === selectedTicker)?.total ?? null;
    const tickerRows = (selectedTicker && rawData) ? (rawData.get(selectedTicker) || []) : [];
    const patternBarsFromEnd = selectedDetection
      ? (tickerRows.length - 1 - ((selectedDetection.setupType === "rhs" || selectedDetection.setupType === "hs")
          ? (selectedDetection.rightShoulderIdx ?? selectedDetection.rightRim ?? 0)
          : (selectedDetection.rightRim ?? 0)))
      : null;
    const recencyMult = computeRecencyMultiplier(patternBarsFromEnd, tickerRows.length);
    const domainNodes = (!noTicker && !noDet) ? buildDomainNodes(selectedDetection, sectorPulse, recencyMult) : [];
    const { cohesionScore } = domainNodes.length ? analyzeDomainGraph(domainNodes) : { cohesionScore: {} };
    const confirmed = domainNodes.filter(n => { const c = cohesionScore[n.id] ?? 0.5; return c >= 0.5 && n.score >= 0.5; }).length;
    const gapRisk   = domainNodes.filter(n => { const c = cohesionScore[n.id] ?? 0.5; return c >= 0.5 && n.score < 0.5; }).length;
    const drags     = domainNodes.filter(n => { const c = cohesionScore[n.id] ?? 0.5; return c < 0.5 && n.score < 0.5; }).length;
    const avgScore  = domainNodes.length ? domainNodes.reduce((s, n) => s + n.score, 0) / domainNodes.length : 0;

    // Recency badge label
    const recencyBadge = patternBarsFromEnd == null ? null
      : patternBarsFromEnd <= 10 ? { label: "🟢 Fresh", color: "#34d399" }
      : patternBarsFromEnd <= 60 ? { label: `⚡ ${patternBarsFromEnd}b ago`, color: "#fbbf24" }
      : { label: `⏳ ${patternBarsFromEnd}b ago`, color: "#f87171" };

    return (
      <div style={{ height: isMobile ? "auto" : "100%", minHeight: isMobile ? "85vh" : undefined, display: "flex", flexDirection: "column", overflow: isMobile ? "visible" : "hidden", background: "#0b0d11" }}>
        <style>{`

        `}</style>

        {/* Header bar */}
        <div style={{ padding: "10px 16px", borderBottom: "1px solid #2a2f45", background: "#13161d", display: "flex", alignItems: "center", gap: 12, flexShrink: 0, flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            {filteredScores.length > 1 && selectedTicker && (() => {
              const idx = filteredScores.findIndex(s => s.ticker === selectedTicker);
              const hasPrev = idx > 0;
              const hasNext = idx < filteredScores.length - 1;
              const navBtn = (enabled, onClick, label) => (
                <button onClick={onClick} disabled={!enabled} title={label} style={{
                  width: 22, height: 22, borderRadius: 5, border: "1px solid #2a2f45",
                  background: enabled ? "#1a1d27" : "transparent",
                  color: enabled ? "#7986cb" : "#2a2f45",
                  cursor: enabled ? "pointer" : "default", fontSize: 12,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  opacity: enabled ? 1 : 0.3, flexShrink: 0,
                }}>{label}</button>
              );
              return (<>
                {navBtn(hasPrev, () => setSelectedTicker(filteredScores[idx - 1].ticker), "‹")}
                {navBtn(hasNext, () => setSelectedTicker(filteredScores[idx + 1].ticker), "›")}
                <span style={{ fontSize: 11, color: "#7986cb" }}>{idx + 1}/{filteredScores.length}</span>
              </>);
            })()}
            <div>
              <span style={{ fontSize: 15, fontWeight: 800, color: "#7c9fff" }}>⬡ Pattern Intelligence</span>
              {selectedTicker && (
                <span style={{ fontSize: 12, color: "#7986cb", marginLeft: 10 }}>
                  {selectedTicker}
                  {sectorPulse != null && (
                    <span style={{ marginLeft: 7, color: sectorPulse >= 0 ? "#26a69a" : "#ef5350", fontWeight: 700 }}>
                      · sector {sectorPulse >= 0 ? "+" : ""}{(sectorPulse * 100).toFixed(0)}
                    </span>
                  )}
                  {recencyBadge && (
                    <span style={{ marginLeft: 8, color: recencyBadge.color, fontWeight: 700 }} title={`Recency weight: ${Math.round(recencyMult * 100)}%`}>
                      · {recencyBadge.label}
                    </span>
                  )}
                </span>
              )}
            </div>
          </div>
          {domainNodes.length > 0 && (
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              {[
                { label: "Confirmed", count: confirmed, color: "#34d399" },
                { label: "Gap Risk",  count: gapRisk,   color: "#fbbf24" },
                { label: "Drag",      count: drags,     color: "#f87171" },
              ].map(({ label, count, color }) => (
                <span key={label} style={{ fontSize: 12, fontWeight: 700, padding: "3px 10px", borderRadius: 20, background: `${color}18`, color, border: `1px solid ${color}44` }}>
                  {label} {count}
                </span>
              ))}
              <span style={{ fontSize: 12, color: "#7986cb" }}>
                avg <span style={{ color: avgScore >= 0.65 ? "#34d399" : avgScore >= 0.35 ? "#fbbf24" : "#f87171", fontWeight: 800 }}>{Math.round(avgScore * 100)}</span>
              </span>
            </div>
          )}
          <div style={{ marginLeft: isMobile ? 0 : "auto", display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            {!isMobile && (
              <div style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 11, color: "#7986cb" }}>
                <span>Edge heat:</span>
                <div style={{ width: 40, height: 4, background: "linear-gradient(to right, rgb(120,134,203), rgb(251,191,36), rgb(249,115,22))", borderRadius: 2 }} />
                <svg width={20} height={6}><line x1={0} y1={3} x2={20} y2={3} stroke="#7986cb" strokeWidth={1.5} strokeDasharray="4 3" /></svg>
                <span>bridge</span>
              </div>
            )}

          </div>
        </div>

        {/* Empty states */}
        {noTicker && (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12, color: "#4a5080" }}>
            <div style={{ fontSize: 40, opacity: 0.3 }}>⬡</div>
            <div style={{ fontWeight: 600, fontSize: 14, color: "#7986cb" }}>No ticker selected</div>
            <div style={{ fontSize: 12 }}>Select a ticker from the Leaderboard, then open this tab.</div>
          </div>
        )}
        {noDet && (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12, color: "#4a5080" }}>
            <div style={{ fontSize: 40, opacity: 0.3 }}>⚠️</div>
            <div style={{ fontWeight: 600, fontSize: 14, color: "#fbbf24" }}>{selectedTicker} — no pattern detected</div>
            <div style={{ fontSize: 12 }}>Adjust scan parameters to get detection data for this ticker.</div>
          </div>
        )}

        {/* Main content: graph + optional drawer */}
        {domainNodes.length > 0 && (
          <div style={{ flex: 1, display: "flex", minHeight: isMobile ? 0 : 0, flexDirection: isMobile ? "column" : "row" }}>
            <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", overflow: isMobile ? "visible" : "hidden" }}>
              {/* Legend row */}
              <div style={{ padding: "9px 16px", borderBottom: "1px solid #2a2f45", background: "#13161d", display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap", flexShrink: 0 }}>
                <span style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", color: "#7986cb" }}>Cohesion arc</span>
                {[["#f87171","fringe"],["#fbbf24","mid"],["#34d399","core"]].map(([col, lbl]) => (
                  <div key={lbl} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                    <div style={{ width: 12, height: 12, borderRadius: "50%", border: `2px solid ${col}`, background: "transparent" }} />
                    <span style={{ fontSize: 11, color: "#9aa6d4" }}>{lbl}</span>
                  </div>
                ))}
                {!isMobile && <span style={{ fontSize: 11, color: "#7986cb" }}>· Node size = score · Inner glow = strength</span>}
                {!isMobile && <span style={{ fontSize: 11, color: "#7986cb", marginLeft: "auto" }}>Click node for details</span>}
              </div>
              {/* SVG */}
              <div style={isMobile ? { height: 340, position: "relative" } : { flex: 1, minHeight: 0, position: "relative" }}>
                <SVGDomainGraph
                  nodes={domainNodes}
                  selectedId={selectedDomainNode?.id}
                  onSelectNode={setSelectedDomainNode}
                />
              </div>

            </div>
            {/* Node detail drawer */}
            {selectedDomainNode && (
              <DomainDrawer
                node={selectedDomainNode}
                allNodes={domainNodes}
                onClose={() => setSelectedDomainNode(null)}
                isMobile={isMobile}
              />
            )}
          </div>
        )}
      </div>
    );
  };

  // ─── Root render ─────────────────────────────────────────────────────────────
  return (
    <div style={S.app} className="cupscan-app-root">
      <style>{`
        *, *::before, *::after { box-sizing: border-box; }
        html, body {
          height: 100%; margin: 0; overflow: hidden;
          width: 100%; overscroll-behavior: none;
        }
        #root, #__next { height: 100%; overflow: hidden; }
        .cupscan-app-root { height: 100vh; overflow: hidden; }
        @supports (height: 100dvh) { .cupscan-app-root { height: 100dvh; } }
        /* Mobile: make the panel div scrollable instead of the document */
        @media (max-width: 768px) {
          .pp-panel { overflow-y: auto !important; -webkit-overflow-scrolling: touch !important; }
        }
      `}</style>
      {/* Header */}
      <div style={S.header}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 0 }}>
          <span style={S.logo}>⌖ PatternPulse</span>
          {!isMobile && <span style={S.logoSub}>by AlgoGradient · Cup &amp; Handle · Reverse H&amp;S · H&amp;S · Rounded Top</span>}
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexShrink: 0 }}>
          {scanStatus === "done" && !isMobile && (
            <span style={{
              fontSize: 11, padding: "3px 10px", borderRadius: 20, fontWeight: 600,
              background: "rgba(38,166,154,0.15)", color: COLORS.green, border: `1px solid ${COLORS.green}`
            }}>
              ✓ Scan complete
            </span>
          )}
          <button
            style={{ ...S.btn("ghost"), fontSize: 11, padding: "4px 10px", color: COLORS.textDim }}
            onClick={() => setDarkMode(m => !m)}
            title="Toggle dark/light mode"
          >
            {darkMode ? "☀️" : "🌙"}
          </button>
          {!isMobile && (
            <button
              style={{ ...S.btn("ghost"), fontSize: 11, padding: "4px 10px", color: COLORS.textDim }}
              title="Keyboard shortcuts: J/K navigate · Enter open chart · H heatmap · L leaderboard · D domain"
            >
              ⌨
            </button>
          )}
          {rawData && (
            <button style={S.btn("secondary", !scores.length)} onClick={handleExport} disabled={!scores.length}>
              {isMobile ? "↓ CSV" : "Export CSV"}
            </button>
          )}
        </div>
      </div>

      {/* Main layout */}
      <div style={S.main}>
        {/* Sidebar (desktop only — hidden on mobile, content moves to Settings drawer) */}
        <div style={S.sidebar}>
          <div style={S.sidebarScroll}>
            {renderUpload()}
            {scanStatus === "scanning" && renderProgress()}
            {rawData && renderScanSummary()}
            {renderWarnings()}
            {renderTolerances()}
          </div>
        </div>

        {/* Content */}
        <div style={S.content}>
          <div style={S.tabBar}>
            {[
              { id: "leaderboard", label: `📋 Leaderboard${scores.length ? ` (${scores.length})` : ""}` },
              { id: "chart", label: `📈 Chart${selectedTicker ? ` · ${selectedTicker}` : ""}` },
              { id: "heatmap", label: `🌡️ Pulse Heatmap${rawData ? ` (${hmData.length})` : ""}` },
              { id: "domain", label: `⬡ Pattern Intel${selectedTicker ? ` · ${selectedTicker}` : ""}` },
            ].map(({ id, label }) => (
              <button key={id} style={S.tab(activeTab === id)} onClick={() => setActiveTab(id)}>
                {label}
              </button>
            ))}
          </div>

          <div style={S.panel} className="pp-panel">
            {activeTab === "leaderboard" && renderLeaderboard()}
            {activeTab === "chart" && renderChart()}
            {activeTab === "heatmap" && renderHeatmap()}
            {activeTab === "domain" && renderDomainIntelligence()}
          </div>
        </div>
      </div>

      {/* Mobile bottom nav — replaces top tab bar on small screens */}
      {isMobile && (
        <div style={{
          position: "fixed", bottom: 0, left: 0, right: 0,
          background: COLORS.surface, borderTop: `1px solid ${COLORS.border}`,
          display: "flex", zIndex: 200,
          paddingBottom: "env(safe-area-inset-bottom, 0px)",
        }}>
          {[
            { id: "leaderboard", icon: "📋", label: "Scores" },
            { id: "chart",       icon: "📈", label: "Chart" },
            { id: "heatmap",     icon: "🌡️",  label: "Heat" },
            { id: "domain",      icon: "⬡",   label: "Intel" },
            { id: "settings",    icon: "⚙️",  label: "Settings" },
          ].map(({ id, icon, label }) => (
            <button key={id}
              onClick={() => id === "settings" ? setShowSettings(true) : setActiveTab(id)}
              style={{
                flex: 1, display: "flex", flexDirection: "column", alignItems: "center",
                justifyContent: "center", padding: "8px 4px", gap: 3,
                background: "none", border: "none", cursor: "pointer",
                color: activeTab === id ? COLORS.accent : COLORS.textMuted,
                fontSize: 10, fontWeight: 600,
              }}
            >
              <span style={{ fontSize: 18 }}>{icon}</span>
              <span>{label}</span>
            </button>
          ))}
        </div>
      )}

      {/* Mobile settings drawer — holds Data Source + Detection Parameters
          (the desktop sidebar content) behind a bottom-sheet overlay */}
      {isMobile && showSettings && (
        <>
          <div onClick={() => setShowSettings(false)} style={{
            position: "fixed", inset: 0, background: "rgba(0,0,0,.55)", zIndex: 250,
          }} />
          <div style={{
            position: "fixed", bottom: 0, left: 0, right: 0,
            background: COLORS.surface, borderTop: `1px solid ${COLORS.border}`,
            borderRadius: "16px 16px 0 0", zIndex: 300,
            maxHeight: "82vh", overflowY: "auto",
            paddingBottom: "env(safe-area-inset-bottom, 16px)",
          }}>
            <div style={{ width: 36, height: 4, background: COLORS.border, borderRadius: 2, margin: "12px auto 4px" }} />
            <div style={{ padding: "8px 16px 12px", display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: `1px solid ${COLORS.border}` }}>
              <span style={{ fontSize: 15, fontWeight: 800 }}>Settings</span>
              <button onClick={() => setShowSettings(false)} style={{ background: "none", border: "none", color: COLORS.textDim, fontSize: 22, cursor: "pointer" }}>×</button>
            </div>
            <div style={{ padding: "0 16px 24px" }}>
              {renderUpload()}
              {scanStatus === "scanning" && renderProgress()}
              {rawData && renderScanSummary()}
              {renderWarnings()}
              {renderTolerances()}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
