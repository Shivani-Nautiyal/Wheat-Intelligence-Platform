import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import {
  Activity, AlertTriangle, BrainCircuit, Info,
  TrendingUp, ShieldAlert, Layers, CheckCircle2,
  Wheat, Search, Map, Eye, ChevronRight
} from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, PieChart, Pie, Cell,
  AreaChart, Area, RadarChart, Radar, PolarGrid, PolarAngleAxis
} from "recharts";
import WheatMap from "./components/WheatMap";
import {
  DEFAULT_API_STATS, HEALTH_DISTRIBUTION, DISEASE_DISTRIBUTION,
  PHENOLOGY_DATA, NDVI_HISTOGRAM, ML_RESULTS, FEATURE_IMPORTANCE, BAND_MEANS
} from "./mockData";

// ── Types ─────────────────────────────────────────────────────────────────────
interface ApiStats {
  wheat_area: number;
  avg_ndvi:   number;
  avg_ndre:   number;
  season:     string;
  region:     string;
  health_distribution: { healthy: number; moderate: number; stressed: number };
}

type TabId =
  | "satellite" | "health" | "disease" | "change"
  | "ml" | "xai" | "alerts" | "about";

// ── Tab definition ────────────────────────────────────────────────────────────
const TABS: { id: TabId; label: string; icon: React.FC<any> }[] = [
  { id: "satellite", label: "Satellite Map",       icon: Map          },
  { id: "health",    label: "Crop Health",          icon: Activity     },
  { id: "disease",   label: "Disease Risk",         icon: AlertTriangle},
  { id: "change",    label: "Change Detection",     icon: TrendingUp   },
  { id: "ml",        label: "ML Results",           icon: BrainCircuit },
  { id: "xai",       label: "Explainable AI",       icon: Eye          },
  { id: "alerts",    label: "Alerts",               icon: ShieldAlert  },
  { id: "about",     label: "About",                icon: Info         },
];

// ── Shared chart theme ────────────────────────────────────────────────────────
const TOOLTIP_STYLE = {
  contentStyle: {
    background: "#1e293b", border: "1px solid #334155",
    borderRadius: 8, color: "white", fontSize: 12,
  },
};

// ── Sub-components ────────────────────────────────────────────────────────────
function KpiCard({ label, value, sub, Icon, color, bg }:
  { label: string; value: string; sub: string; Icon: React.FC<any>; color: string; bg: string }) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex items-center gap-4 hover:border-slate-700 transition-colors">
      <div className="rounded-xl flex items-center justify-center w-11 h-11 shrink-0"
           style={{ background: bg }}>
        <Icon size={18} color={color} />
      </div>
      <div className="min-w-0">
        <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest truncate">{label}</p>
        <p className="text-lg font-black text-white leading-tight">{value}</p>
        <p className="text-[10px] text-slate-600 mt-0.5">{sub}</p>
      </div>
    </div>
  );
}

function Card({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 h-full">
      <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">{title}</p>
      {subtitle && <p className="text-sm font-semibold text-white mt-1 mb-5">{subtitle}</p>}
      {children}
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [activeTab, setActiveTab]     = useState<TabId>("satellite");
  const [stats, setStats]             = useState<ApiStats>(DEFAULT_API_STATS);
  const [mlResults, setMlResults]     = useState(ML_RESULTS);
  const [apiOnline, setApiOnline]     = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [currentCoords, setCurrentCoords] = useState("30.9010°N, 75.8573°E");
  const [toast, setToast]             = useState<string | null>(null);
  const [activeLayer, setActiveLayer] = useState<"NDVI"|"NDRE"|"HEALTH"|"RISK">("NDVI");

  const layerUrls: Record<"NDVI"|"NDRE"|"HEALTH"|"RISK", string> = {
    NDVI:   "/api/raster/ndvi_wheat_ludhiana.tif",
    NDRE:   "/api/raster/ndre_wheat_ludhiana.tif",
    HEALTH: "/api/raster/crop_health_ludhiana.tif",
    RISK:   "/api/raster/disease_risk_ludhiana.tif",
  };

  // Fetch from FastAPI — fall back to baked-in data silently
  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [statsRes, mlRes] = await Promise.all([
          axios.get<ApiStats>("/api/stats"),
          axios.get("/api/ml-results"),
        ]);
        setStats(statsRes.data);
        setMlResults(mlRes.data);
        setApiOnline(true);
      } catch {
        // Backend not running — use baked-in real GEE values (already in state)
        setApiOnline(false);
      }
    };
    fetchAll();
  }, []);

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3000);
  }, []);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = searchQuery.trim();
    if (!trimmed) return;
    setIsSearching(true);
    try {
      const parts = trimmed.split(",").map((p) => p.trim());
      let res;
      if (parts.length === 2 && !isNaN(Number(parts[0])) && !isNaN(Number(parts[1]))) {
        res = await axios.post("/api/search", {
          latitude: Number(parts[0]), longitude: Number(parts[1]),
        });
      } else {
        res = await axios.post("/api/search", { query: trimmed });
      }
      const { latitude, longitude, local_ndvi, local_ndre } = res.data;
      setCurrentCoords(`${latitude.toFixed(4)}°N, ${longitude.toFixed(4)}°E`);
      setStats((prev) => ({ ...prev, avg_ndvi: local_ndvi, avg_ndre: local_ndre }));
      showToast(`Updated to ${latitude.toFixed(4)}°N, ${longitude.toFixed(4)}°E`);
    } catch (error: unknown) {
      const msg = axios.isAxiosError(error)
        ? error.response?.data?.detail || error.message
        : "Search failed — is the API running?";
      showToast(msg || "Search failed — is the API running?");
    } finally {
      setIsSearching(false);
      setSearchQuery("");
    }
  };

  const kpis = [
    { label:"Wheat Area",       value:`${stats.wheat_area.toLocaleString()} Ha`, sub:"Sentinel-2 mask",  Icon:Wheat,      color:"#22c55e", bg:"rgba(34,197,94,0.1)"   },
    { label:"Mean NDVI",        value:stats.avg_ndvi.toFixed(4),                 sub:"26,600 samples",  Icon:Activity,   color:"#3b82f6", bg:"rgba(59,130,246,0.1)"  },
    { label:"Mean NDRE",        value:stats.avg_ndre.toFixed(4),                 sub:"Red-edge band",   Icon:TrendingUp, color:"#a78bfa", bg:"rgba(167,139,250,0.1)" },
    { label:"Risk Status",      value:"Moderate",                                sub:"227.42 Ha flagged",Icon:AlertTriangle,color:"#f59e0b",bg:"rgba(245,158,11,0.1)" },
  ];

  // Derived health chart data from API (or mock)
  const healthChartData = [
    { name:"Healthy",  value: stats.health_distribution.healthy,  color:"#22c55e" },
    { name:"Moderate", value: stats.health_distribution.moderate, color:"#eab308" },
    { name:"Stressed", value: stats.health_distribution.stressed, color:"#ef4444" },
  ];

  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200 flex">

      {/* ── Sidebar ── */}
      <aside className="w-56 bg-slate-900 border-r border-slate-800 hidden lg:flex flex-col fixed h-full z-10">
        <div className="p-5 border-b border-slate-800">
          <div className="flex items-center gap-3 mb-1">
            <div className="bg-blue-700 p-1.5 rounded-lg">
              <Wheat className="w-4 h-4 text-white" />
            </div>
            <span className="font-bold text-base text-white">WheatMonitor</span>
          </div>
          <p className="text-[10px] text-slate-600 mt-1">Ludhiana · Punjab · 2023-24</p>
          <div className="flex items-center gap-1.5 mt-2">
            <div className={`w-1.5 h-1.5 rounded-full ${apiOnline ? "bg-emerald-500" : "bg-amber-500"}`} />
            <span className={`text-[10px] ${apiOnline ? "text-emerald-500" : "text-amber-500"}`}>
              {apiOnline ? "API connected" : "Offline mode"}
            </span>
          </div>
        </div>

        <nav className="flex-1 p-3 space-y-0.5 overflow-y-auto">
          {TABS.map((t) => {
            const active = activeTab === t.id;
            return (
              <button key={t.id} onClick={() => setActiveTab(t.id)}
                className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-xs font-medium transition-all text-left ${
                  active
                    ? "bg-blue-600/20 text-blue-400 border-l-2 border-blue-500"
                    : "text-slate-500 hover:text-slate-300 hover:bg-slate-800 border-l-2 border-transparent"
                }`}>
                <t.icon size={13} />
                {t.label}
              </button>
            );
          })}
        </nav>

        <div className="p-4 border-t border-slate-800 text-[10px] text-slate-700">
          Sentinel-2 L2A · 10m res<br />Season: {stats.season}
        </div>
      </aside>

      {/* ── Main ── */}
      <main className="flex-1 lg:ml-56 p-6 lg:p-8 min-w-0">

        {/* Header */}
        <header className="mb-8 flex flex-col md:flex-row md:items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-black text-white tracking-tight">
              🌾 Wheat Analytics Dashboard
            </h1>
            <div className="flex items-center gap-2 mt-1.5 flex-wrap">
              <span className="text-slate-600 text-[10px] font-bold uppercase tracking-widest">Focus:</span>
              <span className="px-2 py-0.5 bg-blue-900/40 text-blue-400 rounded text-[10px] font-mono border border-blue-800/50">
                {currentCoords}
              </span>
              <span className="text-[10px] text-slate-700">{TABS.find(t => t.id === activeTab)?.label}</span>
            </div>
          </div>

          <form onSubmit={handleSearch} className="relative shrink-0">
            <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text" value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Place name or Lat, Lon (e.g. Ludhiana or 30.9, 75.8)"
              disabled={isSearching}
              className="pl-9 pr-4 py-2 bg-slate-900 border border-slate-700 rounded-xl text-xs text-white placeholder:text-slate-600 focus:outline-none focus:ring-1 focus:ring-blue-500 w-60 transition-all disabled:opacity-50"
            />
          </form>
        </header>

        {/* KPI row */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-8">
          {kpis.map((k, i) => <KpiCard key={i} {...k} />)}
        </div>

        {/* Satellite view */}
        {activeTab === "satellite" && (
          <div className="grid lg:grid-cols-[2fr_1fr] gap-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 flex flex-col gap-4">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Satellite Map</p>
                  <p className="text-sm font-semibold text-white mt-1">Explore NDVI / NDRE overlays for Ludhiana</p>
                </div>
                <div className="text-right">
                  <p className="text-[10px] text-slate-500 uppercase tracking-widest">Active layer</p>
                  <p className="text-sm font-semibold text-white mt-1">{activeLayer}</p>
                </div>
              </div>
              <div className="h-[580px] rounded-3xl overflow-hidden border border-slate-800">
                <WheatMap layerUrl={layerUrls[activeLayer]} activeIndex={activeLayer} />
              </div>
            </div>

            <div className="space-y-5">
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">Layer Inspector</p>
                <p className="text-sm font-semibold text-white mb-5">Select raster layer</p>
                <div className="space-y-2">
                  {(Object.keys(layerUrls) as Array<keyof typeof layerUrls>).map((key) => (
                    <button key={key} onClick={() => setActiveLayer(key)}
                      className={`w-full flex items-center justify-between p-3 rounded-xl border transition-all text-left ${
                        activeLayer === key
                          ? "bg-blue-600/20 border-blue-500/40 text-blue-300"
                          : "bg-slate-800/50 border-transparent text-slate-400 hover:border-slate-600 hover:text-slate-300"
                      }`}>
                      <span className="text-xs font-medium">{key} Layer</span>
                      <ChevronRight size={14} />
                    </button>
                  ))}
                </div>
              </div>
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                <p className="text-[10px] text-slate-500 mb-2 font-bold uppercase tracking-wide">TIF Source</p>
                <code className="text-[10px] text-blue-400 break-all">{layerUrls[activeLayer]}</code>
                <p className="text-[10px] text-slate-600 mt-2">Served via FastAPI /api/raster</p>
              </div>
            </div>
          </div>
        )}

        {/* ── CROP HEALTH ───────────────────────────────────────────────── */}
        {activeTab === "health" && (
          <div className="grid lg:grid-cols-2 gap-5">
            <Card title="Health Classification" subtitle="Area by health category (Ha) — crop_health_ludhiana.tif">
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={HEALTH_DISTRIBUTION} dataKey="value" nameKey="name"
                       cx="50%" cy="50%" innerRadius={55} outerRadius={80} paddingAngle={4}>
                    {HEALTH_DISTRIBUTION.map((d, i) => <Cell key={i} fill={d.color} />)}
                  </Pie>
                  <Tooltip {...TOOLTIP_STYLE} formatter={(v: any) => [`${v.toFixed(2)} Ha`]} />
                  <Legend wrapperStyle={{ color:"#94a3b8", fontSize:11 }} />
                </PieChart>
              </ResponsiveContainer>
              <div className="mt-4 space-y-2">
                {HEALTH_DISTRIBUTION.map(d => (
                  <div key={d.name} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-2.5 h-2.5 rounded-sm" style={{ background:d.color }} />
                      <span className="text-xs text-slate-400">{d.name}</span>
                    </div>
                    <div className="flex gap-3">
                      <span className="text-xs text-white font-semibold">{d.value} Ha</span>
                      <span className="text-[10px] text-slate-600">{d.pct}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            <Card title="Spectral Phenology Curve" subtitle="NDVI & NDRE seasonal trajectory — 2023-24 Rabi">
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={PHENOLOGY_DATA}>
                  <defs>
                    <linearGradient id="gNdvi" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#3b82f6" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}   />
                    </linearGradient>
                    <linearGradient id="gNdre" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#a78bfa" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#a78bfa" stopOpacity={0}   />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="month" tick={{ fill:"#64748b", fontSize:11 }} />
                  <YAxis tick={{ fill:"#64748b", fontSize:10 }} domain={[0, 0.7]} />
                  <Tooltip {...TOOLTIP_STYLE} />
                  <Legend wrapperStyle={{ color:"#94a3b8", fontSize:11 }} />
                  <Area type="monotone" dataKey="ndvi" name="NDVI" stroke="#3b82f6" fill="url(#gNdvi)" strokeWidth={2} dot={{ r:3, fill:"#3b82f6" }} />
                  <Area type="monotone" dataKey="ndre" name="NDRE" stroke="#a78bfa" fill="url(#gNdre)" strokeWidth={2} dot={{ r:3, fill:"#a78bfa" }} />
                </AreaChart>
              </ResponsiveContainer>
              <p className="text-[10px] text-slate-600 mt-3">
                Current observation (mean NDVI 0.379) sits in the Feb–Mar peak range.
              </p>
            </Card>

            <Card title="NDVI Distribution" subtitle="Pixel frequency across 26,600 samples — sentinel_data_ludhiana.csv">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={NDVI_HISTOGRAM} margin={{ left:-10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" vertical={false} />
                  <XAxis dataKey="range" tick={{ fill:"#64748b", fontSize:9 }} />
                  <YAxis tick={{ fill:"#64748b", fontSize:10 }} />
                  <Tooltip {...TOOLTIP_STYLE} />
                  <Bar dataKey="count" name="Pixel count" radius={[3,3,0,0]}>
                    {NDVI_HISTOGRAM.map((d, i) => (
                      <Cell key={i} fill={
                        d.range.startsWith("0.35") || d.range.startsWith("0.37") ? "#ef4444" :
                        d.range.startsWith("0.41") || d.range.startsWith("0.44") ? "#eab308" : "#22c55e"
                      } />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Spectral Band Profile" subtitle="Mean reflectance per Sentinel-2 band">
              <ResponsiveContainer width="100%" height={200}>
                <RadarChart data={BAND_MEANS} outerRadius="70%">
                  <PolarGrid stroke="#1e3a5f" />
                  <PolarAngleAxis dataKey="band" tick={{ fill:"#64748b", fontSize:9 }} />
                  <Radar name="Reflectance" dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25} />
                  <Tooltip {...TOOLTIP_STYLE} formatter={(v: any) => [v.toFixed(3), "Reflectance"]} />
                </RadarChart>
              </ResponsiveContainer>
              <p className="text-[10px] text-slate-600 mt-2">B8A (NIR) dominant at 0.197 — characteristic wheat canopy signature.</p>
            </Card>
          </div>
        )}

        {/* ── DISEASE RISK ──────────────────────────────────────────────── */}
        {activeTab === "disease" && (
          <div className="grid lg:grid-cols-2 gap-5">
            <Card title="Disease Risk Zones" subtitle="Area by risk class (Ha) — disease_risk_ludhiana.tif">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={DISEASE_DISTRIBUTION} layout="vertical" margin={{ left:20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" horizontal={false} />
                  <XAxis type="number" tick={{ fill:"#64748b", fontSize:10 }} />
                  <YAxis type="category" dataKey="name" tick={{ fill:"#94a3b8", fontSize:11 }} width={100} />
                  <Tooltip {...TOOLTIP_STYLE} formatter={(v: any) => [`${Number(v).toFixed(2)} Ha`]} />
                  <Bar dataKey="value" radius={[0,4,4,0]}>
                    {DISEASE_DISTRIBUTION.map((d, i) => <Cell key={i} fill={d.color} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="mt-4 p-3 bg-amber-900/20 border border-amber-700/30 rounded-xl">
                <p className="text-xs font-semibold text-amber-400 mb-1">⚠️ Moderate Risk Dominant</p>
                <p className="text-[11px] text-slate-500">
                  227.42 Ha (99.98%) under moderate risk. Zero high-risk pixels.
                  Monitor NDRE trends for rust / blight precursors.
                </p>
              </div>
            </Card>

            <Card title="Risk Factor Analysis" subtitle="Key agronomic indicators from GEE data">
              <div className="space-y-3">
                {[
                  { label:"Rust Fungus Propensity",   value:"Moderate",   detail:"NDRE mean 0.262 below healthy threshold 0.35", color:"#eab308" },
                  { label:"Water Stress (NDWI)",       value:"Low",        detail:"NDWI = −0.326  →  dry canopy, minimal water stress", color:"#22c55e" },
                  { label:"Canopy N Status",           value:"Adequate",   detail:"NDRE-based proxy within acceptable range", color:"#22c55e" },
                  { label:"Biomass Accumulation",      value:"Below Peak", detail:"NDVI 0.379 suggests pre-peak or post-peak stage", color:"#f97316" },
                  { label:"High Disease Pixels",       value:"0.00 Ha",    detail:"No Class-3 pixel in disease_risk TIF", color:"#22c55e" },
                ].map(item => (
                  <div key={item.label} className="flex items-start justify-between p-3 bg-slate-800/50 rounded-xl">
                    <div>
                      <p className="text-xs text-slate-400">{item.label}</p>
                      <p className="text-[10px] text-slate-600 mt-0.5">{item.detail}</p>
                    </div>
                    <span className="text-xs font-bold ml-3 shrink-0" style={{ color:item.color }}>{item.value}</span>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        )}

        {/* ── CHANGE DETECTION ──────────────────────────────────────────── */}
        {activeTab === "change" && (
          <Card title="Change Detection" subtitle="Monthly NDVI delta — 2023-24 Rabi season phenology">
            <div className="mb-4 p-3 bg-blue-900/20 border border-blue-700/30 rounded-xl text-[11px] text-blue-400">
              For live change detection, export two-date Sentinel-2 composites from GEE as
              <code className="mx-1 px-1 bg-slate-800 rounded">ndvi_t1.tif</code> and
              <code className="mx-1 px-1 bg-slate-800 rounded">ndvi_t2.tif</code> into the data/ folder.
              The diff is computed automatically by the backend.
            </div>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={[
                { month:"Oct→Nov", delta:+0.08 }, { month:"Nov→Dec", delta:+0.18 },
                { month:"Dec→Jan", delta:+0.14 }, { month:"Jan→Feb", delta:+0.07 },
                { month:"Feb→Mar", delta:-0.02 }, { month:"Mar→Apr", delta:-0.27 },
              ]}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" vertical={false} />
                <XAxis dataKey="month" tick={{ fill:"#64748b", fontSize:11 }} />
                <YAxis tick={{ fill:"#64748b", fontSize:10 }} />
                <Tooltip {...TOOLTIP_STYLE} formatter={(v: any) => [v.toFixed(2), "ΔNDVI"]} />
                <Bar dataKey="delta" name="ΔNDVI" radius={[4,4,0,0]}>
                  {[+0.08,+0.18,+0.14,+0.07,-0.02,-0.27].map((v, i) => (
                    <Cell key={i} fill={v >= 0 ? "#22c55e" : "#ef4444"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        )}

        {/* ── ML RESULTS ────────────────────────────────────────────────── */}
        {activeTab === "ml" && (
          <div className="space-y-5">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
              <div className="px-6 py-4 border-b border-slate-800">
                <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Model Benchmark</p>
                <p className="text-sm font-semibold text-white mt-1">
                  Trained on 26,600 GEE samples · 16 features · 80/20 stratified split · 3-class health
                </p>
              </div>
              <table className="w-full">
                <thead>
                  <tr className="bg-slate-950/60">
                    {["Model","Accuracy","F1 Score","ROC AUC","Status"].map(h => (
                      <th key={h} className="px-6 py-3 text-left text-[10px] text-slate-500 font-bold uppercase tracking-widest">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {mlResults.map((r: any, i: number) => (
                    <tr key={i} className={r.best ? "bg-blue-600/5" : ""}>
                      <td className="px-6 py-4 font-bold text-white text-sm">
                        {r.best && <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500 mr-2 mb-0.5" />}
                        {r.Model}
                      </td>
                      <td className="px-6 py-4 font-mono text-sm text-emerald-400">
                        {((r.Accuracy || r.accuracy || 0) * 100).toFixed(2)}%
                      </td>
                      <td className="px-6 py-4 font-mono text-sm text-blue-400">
                        {((r.F1_Score || r.F1 || r.f1 || 0) * 100).toFixed(2)}%
                      </td>
                      <td className="px-6 py-4 font-mono text-sm text-violet-400">
                        {((r.ROC_AUC || r.auc || 0) * 100).toFixed(2)}%
                      </td>
                      <td className="px-6 py-4">
                        <span className={`text-[10px] px-2.5 py-1 rounded-full font-semibold border ${
                          r.best
                            ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
                            : "bg-slate-700/30 text-slate-500 border-slate-700/30"
                        }`}>
                          {r.best ? "✓ Best model" : "Trained"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="p-4 bg-emerald-900/20 border border-emerald-700/30 rounded-xl text-[11px] text-emerald-400">
              <strong>Note:</strong> Near-perfect accuracy reflects that the classification target was derived from
              the same NDVI/NDRE thresholds used as features. In deployment, replace synthesised labels with
              independent field ground-truth for an unbiased evaluation.
            </div>
          </div>
        )}

        {/* ── EXPLAINABLE AI ────────────────────────────────────────────── */}
        {activeTab === "xai" && (
          <div className="grid lg:grid-cols-2 gap-5">
            <Card title="Feature Importance" subtitle="Random Forest — Gini importance (real model output)">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={FEATURE_IMPORTANCE} layout="vertical" margin={{ left:40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" horizontal={false} />
                  <XAxis type="number" tick={{ fill:"#64748b", fontSize:10 }} domain={[0, 0.6]} />
                  <YAxis type="category" dataKey="feature" tick={{ fill:"#94a3b8", fontSize:10 }} width={150} />
                  <Tooltip {...TOOLTIP_STYLE} formatter={(v: any) => [`${(v*100).toFixed(1)}%`, "Importance"]} />
                  <Bar dataKey="importance" name="Gini importance" radius={[0,4,4,0]}>
                    {FEATURE_IMPORTANCE.map((_, i) => (
                      <Cell key={i} fill={i===0?"#3b82f6":i===1?"#6366f1":i===2?"#8b5cf6":"#334155"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <p className="text-[10px] text-slate-600 mt-3">
                NDVI (52.3%) + NDRE (18.7%) + Veg Health Score (15.5%) drive 86.5% of decisions.
              </p>
            </Card>

            <Card title="Classification Logic" subtitle="How the Random Forest classifies each pixel">
              <div className="space-y-3">
                {[
                  { cls:"Class 1 — Healthy",  color:"#22c55e", bg:"rgba(34,197,94,0.08)",  border:"rgba(34,197,94,0.2)",  rules:["NDVI > 0.50","NDRE > 0.35","EVI > 0.25"], ha:"9.81 Ha",   pct:"4.3%" },
                  { cls:"Class 2 — Moderate", color:"#eab308", bg:"rgba(234,179,8,0.08)",  border:"rgba(234,179,8,0.2)",  rules:["NDVI 0.38–0.50","NDRE 0.22–0.35"],        ha:"156.33 Ha", pct:"68.7%" },
                  { cls:"Class 3 — Poor",     color:"#ef4444", bg:"rgba(239,68,68,0.08)",  border:"rgba(239,68,68,0.2)",  rules:["NDVI < 0.38","NDRE < 0.22"],               ha:"61.32 Ha",  pct:"27.0%" },
                ].map(c => (
                  <div key={c.cls} className="p-3 rounded-xl border" style={{ background:c.bg, borderColor:c.border }}>
                    <div className="flex justify-between mb-2">
                      <span className="text-xs font-bold" style={{ color:c.color }}>{c.cls}</span>
                      <span className="text-[10px] text-slate-500">{c.ha} · {c.pct}</span>
                    </div>
                    <div className="flex gap-1.5 flex-wrap">
                      {c.rules.map(r => (
                        <code key={r} className="text-[10px] px-1.5 py-0.5 bg-white/5 rounded text-slate-400">{r}</code>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-4 p-3 bg-indigo-900/20 border border-indigo-700/30 rounded-xl">
                <p className="text-[11px] text-indigo-400">
                  💡 68.7% of pixels are Class 2 because NDVI sits at 0.379 — just below the healthy threshold of 0.50.
                  This is characteristic of late-emergence or post-peak wheat.
                </p>
              </div>
            </Card>
          </div>
        )}

        {/* ── ALERTS ────────────────────────────────────────────────────── */}
        {activeTab === "alerts" && (
          <div className="space-y-3">
            {[
              { type:"warning", icon:"⚠️", title:"Moderate Stress Dominant",
                msg:`68.7% of wheat area (156.33 Ha) classified as moderate health. Canopy stress visible in NDRE < 0.35.` },
              { type:"warning", icon:"📉", title:"NDVI Below Peak Vigor",
                msg:`Mean NDVI = ${stats.avg_ndvi.toFixed(4)} is below the optimal threshold (> 0.50). Monitor for biomass stagnation.` },
              { type:"success", icon:"✅", title:"No High Disease Risk Detected",
                msg:"0.00 Ha classified as high risk in disease_risk_ludhiana.tif. Moderate risk spans 227.42 Ha — precautionary monitoring advised." },
              { type:"info",    icon:"🛰️", title:"Sentinel-2 L2A — All Bands Valid",
                msg:"26,600 pixel samples processed at 10m resolution. 7 spectral bands + 6 indices extracted via Google Earth Engine." },
              { type:"info",    icon:"🌾", title:"Wheat Area Confirmed",
                msg:"227.48 Ha of wheat detected via binary mask. Pixel resolution: 6,680 × 5,014 @ 10m (Ludhiana extent)." },
            ].map((a, i) => {
              const styles: Record<string, string> = {
                warning: "bg-amber-900/20 border-amber-700/30 text-amber-400",
                success: "bg-emerald-900/20 border-emerald-700/30 text-emerald-400",
                info:    "bg-blue-900/20 border-blue-700/30 text-blue-400",
              };
              return (
                <div key={i} className={`flex gap-4 p-4 rounded-xl border ${styles[a.type]}`}>
                  <span className="text-xl shrink-0">{a.icon}</span>
                  <div>
                    <p className="text-sm font-bold mb-1">{a.title}</p>
                    <p className="text-[11px] text-slate-400 leading-relaxed">{a.msg}</p>
                  </div>
                </div>
              );
            })}
            <p className="text-[10px] text-slate-700 pt-2">
              All alerts derived from real GEE data —{" "}
              <code className="text-blue-600">crop_health_ludhiana.tif</code>,{" "}
              <code className="text-blue-600">disease_risk_ludhiana.tif</code>,{" "}
              <code className="text-blue-600">area_statistics_ludhiana_2023_24.csv</code>
            </p>
          </div>
        )}

        {/* ── ABOUT ─────────────────────────────────────────────────────── */}
        {activeTab === "about" && (
          <div className="grid lg:grid-cols-2 gap-5">
            <Card title="Data Sources" subtitle="Real GEE exports — 2023-24 Rabi season">
              <div className="space-y-2.5">
                {[
                  ["ndvi_wheat_ludhiana.tif",       "NDVI raster · 6680×5014 px · float32 · range 0.35–0.64"],
                  ["ndre_wheat_ludhiana.tif",       "NDRE raster · red-edge chlorophyll · range 0.15–0.51"],
                  ["wheat_mask_ludhiana.tif",       "Binary wheat mask · 26,592 wheat pixels = 265.92 Ha"],
                  ["crop_health_ludhiana.tif",      "3-class map · 1=Healthy 2=Moderate 3=Poor"],
                  ["disease_risk_ludhiana.tif",     "Risk raster · Class 2 (moderate) = 99.98% pixels"],
                  ["sentinel_data_ludhiana.csv",    "26,600 samples · 13 bands + indices per pixel"],
                  ["area_statistics_ludhiana.csv",  "7 aggregated metrics · GEE reduceRegion output"],
                ].map(([file, desc]) => (
                  <div key={file} className="p-2.5 bg-slate-800/50 rounded-lg">
                    <code className="text-[10px] text-blue-400">{file}</code>
                    <p className="text-[10px] text-slate-600 mt-0.5">{desc}</p>
                  </div>
                ))}
              </div>
            </Card>

            <Card title="Tech Stack & Methodology">
              <div className="grid grid-cols-2 gap-2 mb-5">
                {["Google Earth Engine","Sentinel-2 L2A","scikit-learn RF","FastAPI backend",
                  "React 18 + Recharts","Python · PIL","Streamlit UI","10m resolution"].map(t => (
                  <div key={t} className="flex items-center gap-2 p-2.5 bg-slate-800/50 rounded-lg border border-slate-700">
                    <CheckCircle2 size={11} className="text-emerald-500 shrink-0" />
                    <span className="text-[10px] text-slate-400">{t}</span>
                  </div>
                ))}
              </div>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                Sentinel-2 L2A surface reflectance bands exported from GEE for Ludhiana district.
                Spectral indices computed at 10m resolution. Random Forest trained on 26,600 pixels
                with 16 features (7 bands + 6 indices + 3 derived agronomic scores).
                Health classification thresholds calibrated against the GEE-exported crop_health raster.
              </p>
              <div className="mt-4 p-3 bg-blue-900/20 border border-blue-700/30 rounded-lg">
                <p className="text-[10px] text-blue-400">Final Year Project — Agricultural Remote Sensing © 2024</p>
              </div>
            </Card>
          </div>
        )}
      </main>

      {/* ── Toast ── */}
      {toast && (
        <div className="fixed top-6 right-6 z-50 bg-emerald-700 text-white px-5 py-3 rounded-xl shadow-2xl flex items-center gap-3 animate-pulse">
          <CheckCircle2 size={16} />
          <span className="text-sm font-medium">{toast}</span>
        </div>
      )}
    </div>
  );
}
