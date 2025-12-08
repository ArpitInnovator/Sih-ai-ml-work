// 'use client';

// import React, { useState, useEffect } from 'react';
// import { RefreshCw, AlertCircle, Clock, MapPin, Moon, Sun } from 'lucide-react';
// import { useTheme } from '@/contexts/ThemeContext';
// import { 
//   LivePredictionResponse, 
//   ModelHealthResponse, 
//   PredictResponse,
//   ModelDetailResponse 
// } from '@shared/api';
// import { 
//   generateDummyLivePrediction,
//   generateDummy24HourForecast,
//   generateDummyHealthStatus,
//   generateDummyModelDetail,
//   generateDummyPollutantMetrics,
//   PollutantMetrics,
// } from '@/services/dummyData';
// import { Button } from '@/components/ui/button';
// import Sidebar from '@/components/Sidebar';
// import OverviewMetrics from '@/components/OverviewMetrics';
// import PollutantGauge from '@/components/PollutantGauge';
// import HighestPollutants from '@/components/HighestPollutants';
// import SiteSelector from '@/components/SiteSelector';
// import ForecastChart from '@/components/ForecastChart';
// import InteractiveForecastTool from '@/components/InteractiveForecastTool';
// import HealthStatus from '@/components/HealthStatus';
// import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
// import ModelEvaluation from '@/components/ModelEvaluation';

// type TabType = 'overview' | 'forecast' | 'tool' | 'health' | 'evaluation';
// type TimeRange = 1 | 24 | 48;

// export default function Dashboard() {
//   const { theme, toggleTheme } = useTheme();
//   const [selectedSite, setSelectedSite] = useState(1);
//   const [activeTab, setActiveTab] = useState<TabType>('overview');
//   const [timeRange, setTimeRange] = useState<TimeRange>(24);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState<string | null>(null);
//   const [activeNav, setActiveNav] = useState('dashboard');
  
//   // Data states
//   const [liveData, setLiveData] = useState<LivePredictionResponse | null>(null);
//   const [forecastData, setForecastData] = useState<PredictResponse | null>(null);
//   const [healthStatus, setHealthStatus] = useState<ModelHealthResponse | null>(null);
//   const [modelDetail, setModelDetail] = useState<ModelDetailResponse | null>(null);
//   const [pollutantMetrics, setPollutantMetrics] = useState<PollutantMetrics | null>(null);

//   // Fetch all data
//   const fetchAllData = async (siteId: number, hours: TimeRange = 24) => {
//     try {
//       setLoading(true);
//       setError(null);

//       const [live, forecast, health, detail, metrics] = await Promise.all([
//         Promise.resolve(generateDummyLivePrediction(siteId)),
//         Promise.resolve(generateDummy24HourForecast(siteId, hours)),
//         Promise.resolve(generateDummyHealthStatus()),
//         Promise.resolve(generateDummyModelDetail(siteId)),
//         Promise.resolve(generateDummyPollutantMetrics()),
//       ]);

//       setLiveData(live);
//       setForecastData(forecast);
//       setHealthStatus(health);
//       setModelDetail(detail);
//       setPollutantMetrics(metrics);
//     } catch (err) {
//       const errorMessage = err instanceof Error ? err.message : 'Failed to fetch data';
//       setError(errorMessage);
//       console.error('Error:', err);
//     } finally {
//       setLoading(false);
//     }
//   };

//   const handleSiteChange = (siteId: number) => {
//     setSelectedSite(siteId);
//   };

//   const handleRefresh = () => {
//     fetchAllData(selectedSite);
//   };

//   // Update activeTab based on activeNav
//   useEffect(() => {
//     if (activeNav === 'evaluation') {
//       setActiveTab('evaluation');
//     } else if (activeNav === 'forecast') {
//       setActiveTab('forecast');
//     } else if (activeNav === 'dashboard') {
//       setActiveTab('overview');
//     }
//   }, [activeNav]);

//   // Initial fetch
//   useEffect(() => {
//     fetchAllData(selectedSite, timeRange);
//   }, [selectedSite, timeRange]);

//   // Generate trend chart data with realistic patterns
//   const generateTrendData = () => {
//     const hours = timeRange;
//     const now = new Date();
//     const data = [];

//     for (let i = 0; i < hours; i++) {
//       const hour = (now.getHours() + i) % 24;
//       const timeOfDay = hour / 24;

//       // Realistic O3: peaks in afternoon
//       const o3Base = Math.sin((timeOfDay - 0.25) * Math.PI);
//       const o3Value = 35 + (o3Base * 40 + 20) + (Math.random() - 0.5) * 10;

//       // Realistic NO2: peaks in morning/evening (traffic hours)
//       const no2Morning = Math.sin((timeOfDay - 0.2) * Math.PI * 2) * 30;
//       const no2Evening = Math.sin((timeOfDay - 0.8) * Math.PI * 2) * 35;
//       const no2Value = 50 + (no2Morning + no2Evening) / 2 + (Math.random() - 0.5) * 8;

//       // Realistic PM2.5: similar to NO2 but smoother
//       const pm25Base = Math.cos(timeOfDay * Math.PI * 2) * 0.5 + 0.5;
//       const pm25Value = 35 + pm25Base * 60 + (Math.random() - 0.5) * 12;

//       // Realistic PM10: similar pattern to PM2.5 but with higher amplitude
//       const pm10Value = 65 + pm25Base * 90 + (Math.random() - 0.5) * 15;

//       data.push({
//         time: `${String(hour).padStart(2, '0')}:00`,
//         O3: Math.max(5, Math.round(o3Value * 10) / 10),
//         NO2: Math.max(10, Math.round(no2Value * 10) / 10),
//         PM25: Math.max(5, Math.round(pm25Value * 10) / 10),
//         PM10: Math.max(10, Math.round(pm10Value * 10) / 10),
//       });
//     }

//     return data;
//   };

//   const trendData = generateTrendData();

//   return (
//     <div className={`flex h-screen ${theme === 'dark' ? 'bg-slate-950 text-slate-100' : 'bg-white text-slate-900'}`}>
//       {/* Sidebar */}
//       <Sidebar activeNav={activeNav} onNavChange={setActiveNav} />

//       {/* Main Content */}
//       <main className="flex-1 overflow-auto">
//         {/* Header */}
//         <div className={`sticky top-0 z-40 ${theme === 'dark' ? 'bg-slate-900/95' : 'bg-white/95'} backdrop-blur ${theme === 'dark' ? 'border-slate-700' : 'border-slate-200'} border-b`}>
//           <div className="px-8 py-6 flex items-center justify-between">
//             <div>
//               <h1 className={`text-3xl font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>Air Quality Overview</h1>
//               <p className={`text-sm mt-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>Site Tracking Dashboard</p>
//             </div>
//             <div className="flex gap-3">
//               <Button
//                 onClick={toggleTheme}
//                 disabled={loading}
//                 className={`gap-2 border-0 ${theme === 'dark' ? 'bg-slate-800 hover:bg-slate-700 text-white' : 'bg-slate-200 hover:bg-slate-300 text-slate-900'}`}
//               >
//                 {theme === 'dark' ? (
//                   <Sun className="w-4 h-4" />
//                 ) : (
//                   <Moon className="w-4 h-4" />
//                 )}
//               </Button>
//               <Button
//                 onClick={handleRefresh}
//                 disabled={loading}
//                 className="gap-2 bg-cyan-600 hover:bg-cyan-700 text-white border-0"
//               >
//                 <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
//                 Refresh
//               </Button>
//             </div>
//           </div>
//         </div>

//         {/* Content */}
//         <div className={`p-8 ${theme === 'dark' ? 'bg-slate-950' : 'bg-slate-50'}`}>
//           {error && (
//             <div className={`mb-6 p-4 border rounded-lg flex items-start gap-3 ${theme === 'dark' ? 'bg-red-900/20 border-red-700' : 'bg-red-50 border-red-300'}`}>
//               <AlertCircle className={`w-5 h-5 flex-shrink-0 mt-0.5 ${theme === 'dark' ? 'text-red-500' : 'text-red-600'}`} />
//               <div className={`text-sm ${theme === 'dark' ? 'text-red-200' : 'text-red-700'}`}>
//                 <p className="font-semibold">Error</p>
//                 <p className="mt-1">{error}</p>
//               </div>
//             </div>
//           )}

//           {/* Site Selector */}
//           <div className="mb-8">
//             <SiteSelector 
//               selectedSite={selectedSite} 
//               onSiteChange={handleSiteChange}
//               disabled={loading}
//             />
//           </div>

//           {loading ? (
//             <div className="flex items-center justify-center py-12">
//               <div className="text-center">
//                 <RefreshCw className="w-8 h-8 animate-spin text-cyan-500 mx-auto mb-4" />
//                 <p className={theme === 'dark' ? 'text-slate-300' : 'text-slate-600'}>Loading data...</p>
//               </div>
//             </div>
//           ) : (
//             <>
//               {/* Tab Navigation */}
//               <div className={`mb-6 flex gap-2 rounded-lg p-1 border w-fit ${theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-slate-100 border-slate-300'}`}>
//                 {[
//                   { id: 'overview', label: 'Overview' },
//                   { id: 'forecast', label: '24-Hour Forecast' },
//                   { id: 'tool', label: 'Forecast Tool' },
//                   { id: 'evaluation', label: 'Model Evaluation' },
//                   { id: 'health', label: 'Health Status' },
//                 ].map(({ id, label }) => (
//                   <button
//                     key={id}
//                     onClick={() => setActiveTab(id as TabType)}
//                     className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
//                       activeTab === id
//                         ? 'bg-cyan-600 text-white'
//                         : theme === 'dark'
//                         ? 'text-slate-300 hover:text-white'
//                         : 'text-slate-600 hover:text-slate-900'
//                     }`}
//                   >
//                     {label}
//                   </button>
//                 ))}
//               </div>

//               {/* Tab Content */}
//               {activeTab === 'overview' && (
//                 <div className="space-y-6">
//                   {/* Overview Metrics */}
//                   {liveData && pollutantMetrics && (
//                     <OverviewMetrics
//                       o3={liveData.predictions[0].O3_target}
//                       no2={liveData.predictions[0].NO2_target}
//                       hcho={liveData.predictions[0].HCHO_target}
//                       co = {liveData.predictions[0].CO_target}
//                       pm25={liveData.predictions[0].PM25_target}
//                       pm10={liveData.predictions[0].PM10_target}
//                       o3Trend={-5}
//                       no2Trend={3}
//                       hchoTrend={-2}
//                       coTrend={1}
//                       pm25Trend={4}
//                       pm10Trend={-3}
//                     />
//                   )}

//                   {/* Time Range Selector & Trend Chart */}
//                   <div className={`rounded-xl border p-6 ${theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
//                     <div className="flex items-center justify-between mb-6">
//                       <h2 className={`text-lg font-semibold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>Pollution Trends</h2>
//                       <div className={`flex gap-2 rounded-lg p-1 ${theme === 'dark' ? 'bg-slate-700' : 'bg-slate-100'}`}>
//                         {[1, 24, 48].map((hours) => (
//                           <button
//                             key={hours}
//                             onClick={() => setTimeRange(hours as TimeRange)}
//                             className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
//                               timeRange === hours
//                                 ? 'bg-cyan-600 text-white'
//                                 : theme === 'dark'
//                                 ? 'text-slate-300 hover:bg-slate-600'
//                                 : 'text-slate-600 hover:bg-slate-200'
//                             }`}
//                           >
//                             {hours}h
//                           </button>
//                         ))}
//                       </div>
//                     </div>
//                     <ResponsiveContainer width="100%" height={400}>
//                       <LineChart data={trendData}>
//                         <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
//                         <XAxis dataKey="time" stroke="#94a3b8" />
//                         <YAxis stroke="#94a3b8" />
//                         <Tooltip
//                           contentStyle={{
//                             backgroundColor: '#1e293b',
//                             border: '1px solid #475569',
//                             borderRadius: '8px',
//                           }}
//                           labelStyle={{ color: '#f1f5f9' }}
//                           formatter={(value: number) => value.toFixed(1)}
//                         />
//                         <Legend wrapperStyle={{ paddingTop: '20px' }} />
//                         <Line
//                           type="monotone"
//                           dataKey="O3"
//                           stroke="#06b6d4"
//                           strokeWidth={3}
//                           dot={false}
//                           isAnimationActive={true}
//                           name="O₃ (ppb)"
//                         />
//                         <Line
//                           type="monotone"
//                           dataKey="NO2"
//                           stroke="#2563eb"
//                           strokeWidth={3}
//                           dot={false}
//                           isAnimationActive={true}
//                           name="NO₂ (ppb)"
//                         />
//                         <Line
//                           type="monotone"
//                           dataKey="PM25"
//                           stroke="#f97316"
//                           strokeWidth={2}
//                           dot={false}
//                           isAnimationActive={true}
//                           strokeDasharray="5 5"
//                           name="PM2.5 (µg/m³)"
//                         />
//                         <Line
//                           type="monotone"
//                           dataKey="PM10"
//                           stroke="#ef4444"
//                           strokeWidth={2}
//                           dot={false}
//                           isAnimationActive={true}
//                           strokeDasharray="8 4"
//                           name="PM10 (µg/m³)"
//                         />
//                       </LineChart>
//                     </ResponsiveContainer>
//                   </div>

//                   {/* Pollutant Gauges Grid */}
//                   <div>
//                     <h2 className={`text-lg font-semibold mb-4 ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>Today's AQI</h2>
//                     <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
//                       {pollutantMetrics && (
//                         <>
//                           <PollutantGauge 
//                             name="PM2.5"
//                             value={pollutantMetrics.PM25}
//                             maxValue={150}
//                             unit="µg/m³"
//                             color="#f59e0b"
//                           />
//                           <PollutantGauge 
//                             name="PM10"
//                             value={pollutantMetrics.PM10}
//                             maxValue={250}
//                             unit="µg/m³"
//                             color="#f97316"
//                           />
//                           <PollutantGauge 
//                             name="NO₂"
//                             value={liveData?.predictions[0].NO2_target || 0}
//                             maxValue={200}
//                             unit="ppb"
//                             color="#3b82f6"
//                           />
//                           <PollutantGauge 
//                             name="SO₂"
//                             value={pollutantMetrics.SO2}
//                             maxValue={150}
//                             unit="ppb"
//                             color="#fbbf24"
//                           />
//                           <PollutantGauge 
//                             name="CO"
//                             value={pollutantMetrics.CO}
//                             maxValue={5000}
//                             unit="ppm"
//                             color="#ef4444"
//                           />
//                           <PollutantGauge 
//                             name="O₃"
//                             value={liveData?.predictions[0].O3_target || 0}
//                             maxValue={150}
//                             unit="ppb"
//                             color="#06b6d4"
//                           />
//                           <PollutantGauge 
//                             name="PM1"
//                             value={pollutantMetrics.PM1}
//                             maxValue={50}
//                             unit="µg/m³"
//                             color="#ec4899"
//                           />
//                         </>
//                       )}
//                     </div>
//                   </div>

//                   {/* Highest Pollutants */}
//                   <HighestPollutants
//                     o3Events={[
//                       { name: 'Central', location: 'Delhi/National Capital Region', value: '85 µg/m³' },
//                       { name: 'East', location: 'Delhi/National Capital Region', value: '82 µg/m³' },
//                       { name: 'South', location: 'Delhi/National Capital Region', value: '78 µg/m³' },
//                     ]}
//                     no2Events={[
//                       { name: 'Central', location: 'Delhi/National Capital Region', value: '95 µg/m³' },
//                       { name: 'North', location: 'Delhi/National Capital Region', value: '92 µg/m³' },
//                       { name: 'West', location: 'Delhi/National Capital Region', value: '88 µg/m³' },
//                     ]}
//                   />

//                   {/* Station Info */}
//                   {liveData?.live_source && (
//                     <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
//                       <div className={`rounded-xl border p-4 ${theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
//                         <div className="flex items-center gap-2 mb-4">
//                           <MapPin className="w-5 h-5 text-cyan-500" />
//                           <h3 className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>Station Information</h3>
//                         </div>
//                         <div className={`space-y-2 text-sm ${theme === 'dark' ? 'text-slate-300' : 'text-slate-600'}`}>
//                           <p><span className={theme === 'dark' ? 'text-slate-400' : 'text-slate-500'}>Station:</span> {liveData.live_source.station_name}</p>
//                           <p><span className={theme === 'dark' ? 'text-slate-400' : 'text-slate-500'}>Location:</span> {liveData.live_source.station_location[0].toFixed(4)}°N, {liveData.live_source.station_location[1].toFixed(4)}°E</p>
//                           <p><span className={theme === 'dark' ? 'text-slate-400' : 'text-slate-500'}>AQI:</span> <span className="text-amber-400 font-semibold">{liveData.live_source.overall_aqi}</span></p>
//                         </div>
//                       </div>

//                       <div className={`rounded-xl border p-4 ${theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
//                         <div className="flex items-center gap-2 mb-4">
//                           <Clock className="w-5 h-5 text-cyan-500" />
//                           <h3 className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>Measurement Time</h3>
//                         </div>
//                         <div className={`space-y-2 text-sm ${theme === 'dark' ? 'text-slate-300' : 'text-slate-600'}`}>
//                           <p><span className={theme === 'dark' ? 'text-slate-400' : 'text-slate-500'}>Time:</span> {liveData.live_source.measurement_time}</p>
//                           <p><span className={theme === 'dark' ? 'text-slate-400' : 'text-slate-500'}>Timezone:</span> {liveData.live_source.timezone}</p>
//                         </div>
//                       </div>
//                     </div>
//                   )}
//                 </div>
//               )}

//               {activeTab === 'forecast' && forecastData && (
//                 <ForecastChart data={forecastData} siteId={selectedSite} />
//               )}

//               {activeTab === 'tool' && (
//                 <InteractiveForecastTool
//                   siteId={selectedSite}
//                   onForecastGenerated={setForecastData}
//                 />
//               )}

//               {activeTab === 'evaluation' && (
//                 <ModelEvaluation />
//               )}

//               {activeTab === 'health' && healthStatus && modelDetail && (
//                 <HealthStatus 
//                   healthStatus={healthStatus}
//                   modelDetail={modelDetail}
//                 />
//               )}
//             </>
//           )}
//         </div>
//       </main>
//     </div>
//   );
// }


'use client';

import React, { useState, useEffect } from 'react';
import Header from '@/components/Header';
import { RefreshCw, AlertCircle, Clock, MapPin, Car, CheckCircle2 } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';
import { 
  LivePredictionResponse, 
  ModelHealthResponse, 
  PredictResponse,
  ModelDetailResponse 
} from '@shared/api';
import { 
  generateDummyLivePrediction,
  generateDummy24HourForecast,
  generateDummyHealthStatus,
  generateDummyModelDetail,
  generateDummyPollutantMetrics,
  PollutantMetrics,
  
} from '@/services/dummyData';
import { Button } from '@/components/ui/button';
import PollutantGauge from '@/components/PollutantGauge';
import HighestPollutants from '@/components/HighestPollutants';
import SiteSelector from '@/components/SiteSelector';
import ForecastChart from '@/components/ForecastChart';
import InteractiveForecastTool from '@/components/InteractiveForecastTool';
import HealthStatus from '@/components/HealthStatus';
import { ResponsiveLine } from '@nivo/line';
import { parseCSV, convertToChartData, CSVRow } from '@/utils/csvReader';
import DelhiAirMap from '@/components/DelhiAirMap';
type TabType = 'overview' | 'health' ;
type TimeRange = 1 | 24 | 48;

export default function Dashboard() {
  const { theme, toggleTheme } = useTheme();
  const [selectedSite, setSelectedSite] = useState(1);
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [timeRange, setTimeRange] = useState<TimeRange>(24);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeNav, setActiveNav] = useState('dashboard');
  
  // Data states
  const [liveData, setLiveData] = useState<LivePredictionResponse | null>(null);
  const [forecastData, setForecastData] = useState<PredictResponse | null>(null);
  const [healthStatus, setHealthStatus] = useState<ModelHealthResponse | null>(null);
  const [modelDetail, setModelDetail] = useState<ModelDetailResponse | null>(null);
  const [pollutantMetrics, setPollutantMetrics] = useState<PollutantMetrics | null>(null);
  const [csvData, setCsvData] = useState<CSVRow[]>([]);
  const [csvLoading, setCsvLoading] = useState(true);
  
  // Gas selection states
  const [selectedGas, setSelectedGas] = useState<string | null>(null);
  const [visibleGases, setVisibleGases] = useState<Set<string>>(new Set([
    'O₃ (ppb)',
    'NO₂ (ppb)',
    'HCHO (ppb)',
    'CO (ppm)',
    'PM2.5 (µg/m³)',
    'PM10 (µg/m³)'
  ]));

  // Fetch all data
  const fetchAllData = async (siteId: number, hours: TimeRange = 24) => {
    try {
      setLoading(true);
      setError(null);

      const [live, forecast, health, detail, metrics] = await Promise.all([
        Promise.resolve(generateDummyLivePrediction(siteId)),
        Promise.resolve(generateDummy24HourForecast(siteId, hours)),
        Promise.resolve(generateDummyHealthStatus()),
        Promise.resolve(generateDummyModelDetail(siteId)),
        Promise.resolve(generateDummyPollutantMetrics()),
      ]);

      setLiveData(live);
      setForecastData(forecast);
      setHealthStatus(health);
      setModelDetail(detail);
      setPollutantMetrics(metrics);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch data';
      setError(errorMessage);
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSiteChange = (siteId: number) => {
    setSelectedSite(siteId);
  };

  const handleRefresh = () => {
    fetchAllData(selectedSite);
  };

  // Update activeTab based on activeNav
  useEffect(() => {
    if (activeNav === 'dashboard') {
      setActiveTab('overview');
    }
  }, [activeNav]);

  // Load CSV data
  useEffect(() => {
    const loadCSVData = async () => {
      try {
        setCsvLoading(true);
        const data = await parseCSV('/site_1_train_data.csv');
        setCsvData(data);
      } catch (err) {
        console.error('Error loading CSV:', err);
        setError('Failed to load CSV data');
      } finally {
        setCsvLoading(false);
      }
    };
    loadCSVData();
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchAllData(selectedSite, timeRange);
  }, [selectedSite, timeRange]);

  // Generate trend chart data from CSV
  const generateTrendData = () => {
    if (csvData.length === 0) {
      // Fallback to dummy data if CSV not loaded
      const hours = timeRange;
      const now = new Date();
      const data = [];

      for (let i = 0; i < hours; i++) {
        const hour = (now.getHours() + i) % 24;
        const timeOfDay = hour / 24;
        const o3Base = Math.sin((timeOfDay - 0.25) * Math.PI);
        const o3Value = 35 + (o3Base * 40 + 20) + (Math.random() - 0.5) * 10;
        const no2Morning = Math.sin((timeOfDay - 0.2) * Math.PI * 2) * 30;
        const no2Evening = Math.sin((timeOfDay - 0.8) * Math.PI * 2) * 35;
        const no2Value = 50 + (no2Morning + no2Evening) / 2 + (Math.random() - 0.5) * 8;
        const pm25Base = Math.cos(timeOfDay * Math.PI * 2) * 0.5 + 0.5;
        const pm25Value = 35 + pm25Base * 60 + (Math.random() - 0.5) * 12;
        const pm10Value = 65 + pm25Base * 90 + (Math.random() - 0.5) * 15;
        const hchoValue = 10 + (Math.random() - 0.5) * 2;
        const coValue = 50 + (Math.random() - 0.5) * 20;

        // For 48-hour data, add "Day 1" or "Day 2" prefix to ensure unique labels
        let timeLabel = `${String(hour).padStart(2, '0')}:00`;
        if (hours === 48) {
          const day = Math.floor(i / 24) + 1;
          timeLabel = `Day ${day} ${timeLabel}`;
        }

        data.push({
          time: timeLabel,
          O3: Math.max(5, Math.round(o3Value* 10) / 10),
          NO2: Math.max(10, Math.round(no2Value* 10) / 10),
          HCHO: Math.max(1, Math.round(hchoValue* 10) / 10),
          CO: Math.max(100, Math.round(coValue * 10) / 10),
          PM25: Math.max(5, Math.round(pm25Value * 10) / 10),
          PM10: Math.max(10, Math.round(pm10Value * 10) / 10),
        });
      }
      return data;
    }

    // Use CSV data - take the last N hours based on timeRange
    const limit = timeRange === 1 ? 1 : timeRange === 24 ? 24 : 48;
    const chartData = convertToChartData(csvData, limit);
    
    // If CSV doesn't have all gases, fill missing ones with generated values
    // Note: CSV has O3_target, NO2_target, and sometimes HCHO_satellite
    // CO, PM25, PM10 are not in the CSV, so we generate them
    // Note: convertToChartData already handles Day 1/Day 2 prefixes for 48-hour data
    return chartData.map((d, idx) => {
      const baseHour = idx % 24;
      const timeOfDay = baseHour / 24;
      const pm25Base = Math.cos(timeOfDay * Math.PI * 2) * 0.5 + 0.5;
      
      // Generate values for gases not in CSV (CO, PM25, PM10)
      // For HCHO, use CSV value if available, otherwise generate
      return {
        ...d,
        CO: d.CO > 0 ? d.CO : Math.max(100, Math.round((50 + (Math.random() - 0.5) * 20) * 10) / 10),
        PM25: d.PM25 > 0 ? d.PM25 : Math.max(5, Math.round((35 + pm25Base * 60 + (Math.random() - 0.5) * 12) * 10) / 10),
        PM10: d.PM10 > 0 ? d.PM10 : Math.max(10, Math.round((65 + pm25Base * 90 + (Math.random() - 0.5) * 15) * 10) / 10),
        HCHO: d.HCHO > 0 ? d.HCHO : Math.max(1, Math.round((10 + (Math.random() - 0.5) * 2) * 10) / 10),
      };
    });
  };

  const trendData = generateTrendData();

  // Gas definitions
  const gasDefinitions = [
    { id: 'O₃ (ppb)', key: 'O3', color: '#06b6d4', label: 'O₃' },
    { id: 'NO₂ (ppb)', key: 'NO2', color: '#2563eb', label: 'NO₂' },
    { id: 'HCHO (ppb)', key: 'HCHO', color: '#10b981', label: 'HCHO' },
    { id: 'CO (ppm)', key: 'CO', color: '#f59e0b', label: 'CO' },
    { id: 'PM2.5 (µg/m³)', key: 'PM25', color: '#f97316', label: 'PM2.5' },
    { id: 'PM10 (µg/m³)', key: 'PM10', color: '#ef4444', label: 'PM10' },
  ];

  // Helper function to convert hex to rgba
  const hexToRgba = (hex: string, alpha: number) => {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  };

  // Transform data for Nivo ResponsiveLine with click-to-highlight functionality
  const nivoData = gasDefinitions
    .filter(gas => visibleGases.has(gas.id))
    .map(gas => {
      const isSelected = selectedGas === gas.id;
      const isGrayedOut = selectedGas !== null && selectedGas !== gas.id;
      
      return {
        id: gas.id,
        data: trendData.map(d => ({ x: d.time, y: d[gas.key as keyof typeof d] as number })),
        color: isGrayedOut ? hexToRgba('#9ca3af', 0.3) : gas.color,
        originalColor: gas.color, // Keep original color for points
        lineWidth: isSelected ? 3 : 2,
        pointSize: isSelected ? 18 : 16, // Circular dots at every hour - consistent size for all gases
      };
    });

  // Handle gas selection toggle
  const handleGasToggle = (gasId: string) => {
    setVisibleGases(prev => {
      const newSet = new Set(prev);
      if (newSet.has(gasId)) {
        newSet.delete(gasId);
      } else {
        newSet.add(gasId);
      }
      return newSet;
    });
  };

  // Handle gas click for highlighting
  const handleGasClick = (gasId: string) => {
    if (selectedGas === gasId) {
      setSelectedGas(null); // Deselect if clicking the same gas
    } else {
      setSelectedGas(gasId);
    }
  };


  return (
    <div className={`flex h-screen ${theme === 'dark' ? 'bg-slate-950 text-slate-100' : 'bg-white text-slate-900'}`}>
      
      {/* Main Content */}
      <main className="flex-1 ">
        <Header 
          title="Air Quality Overview"
          subtitle="Current gases levels across all 7 regions of Delhi."
          onRefresh={handleRefresh}
          loading={loading}
          showRefresh={true}
          theme={theme}
          onThemeToggle={toggleTheme}
        />
        
        {/* Content */}
        <div className={`p-8 ${theme === 'dark' ? 'bg-slate-950' : 'bg-slate-50'}`}>
          {error && (
            <div className={`mb-6 p-4 border rounded-lg flex items-start gap-3 ${theme === 'dark' ? 'bg-red-900/20 border-red-700' : 'bg-red-50 border-red-300'}`}>
              <AlertCircle className={`w-5 h-5 flex-shrink-0 mt-0.5 ${theme === 'dark' ? 'text-red-500' : 'text-red-600'}`} />
              <div className={`text-sm ${theme === 'dark' ? 'text-red-200' : 'text-red-700'}`}>
                <p className="font-semibold">Error</p>
                <p className="mt-1">{error}</p>
              </div>
            </div>
          )}

          {loading ? (
            <div className="flex items-center justify-center py-12">
              <div className="text-center">
                <RefreshCw className="w-8 h-8 animate-spin text-cyan-500 mx-auto mb-4" />
                <p className={theme === 'dark' ? 'text-slate-300' : 'text-slate-600'}>Loading data...</p>
              </div>
            </div>
          ) : (
            <>
              {/* Tab Content */}
              {activeTab === 'overview' && (
                <div className="space-y-6">
                  {/* Current Air Pollution Data */}
                  {liveData && (
                    <div className="space-y-4">
                      <h2 className={`text-lg font-semibold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>
                        Current Air Pollution Data
                      </h2>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {/* NO2 Cards - Sentinel-5P OFFL */}
                        <div className={`relative rounded-lg border p-5 shadow-sm transition-all hover:shadow-md ${theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-slate-50 border-slate-200'}`}>
                          <div className="absolute top-4 right-4">
                            <CheckCircle2 className="w-6 h-6 text-green-500" />
                          </div>
                          <div className="flex items-center gap-3 mb-3">
                            <div className={`p-2 rounded-lg ${theme === 'dark' ? 'bg-red-500/20' : 'bg-red-50'}`}>
                              <Car className="w-6 h-6 text-red-500" />
                            </div>
                            <div>
                              <h3 className={`text-2xl font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>NO₂</h3>
                              <p className={`text-xs mt-0.5 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>Sentinel-5P OFFL</p>
                            </div>
                          </div>
                          <div className="mt-4">
                            <div className="flex items-baseline gap-2 mb-2">
                              <span className={`text-4xl font-bold ${theme === 'dark' ? 'text-green-400' : 'text-green-600'}`}>
                                {liveData.predictions[0].NO2_target.toFixed(1)}
                              </span>
                              <span className={`text-sm font-medium ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
                                µg/m³
                              </span>
                            </div>
                            <p className={`text-base font-semibold ${theme === 'dark' ? 'text-green-400' : 'text-green-600'}`}>Good</p>
                          </div>
                        </div>

                        {/* NO2 Cards - Sentinel-5P NRTI */}
                        <div className={`relative rounded-lg border p-5 shadow-sm transition-all hover:shadow-md ${theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-slate-50 border-slate-200'}`}>
                          <div className="absolute top-4 right-4">
                            <CheckCircle2 className="w-6 h-6 text-green-500" />
                          </div>
                          <div className="flex items-center gap-3 mb-3">
                            <div className={`p-2 rounded-lg ${theme === 'dark' ? 'bg-red-500/20' : 'bg-red-50'}`}>
                              <Car className="w-6 h-6 text-red-500" />
                            </div>
                            <div>
                              <h3 className={`text-2xl font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>NO₂</h3>
                              <p className={`text-xs mt-0.5 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>Sentinel-5P NRTI</p>
                            </div>
                          </div>
                          <div className="mt-4">
                            <div className="flex items-baseline gap-2 mb-2">
                              <span className={`text-4xl font-bold ${theme === 'dark' ? 'text-green-400' : 'text-green-600'}`}>
                                {liveData.predictions[0].NO2_target.toFixed(1)}
                              </span>
                              <span className={`text-sm font-medium ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
                                µg/m³
                              </span>
                            </div>
                            <p className={`text-base font-semibold ${theme === 'dark' ? 'text-green-400' : 'text-green-600'}`}>Good</p>
                          </div>
                        </div>

                        {/* O3 Card - Sentinel-5P OFFL */}
                        <div className={`relative rounded-lg border p-5 shadow-sm transition-all hover:shadow-md ${theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-slate-50 border-slate-200'}`}>
                          <div className="absolute top-4 right-4">
                            <CheckCircle2 className="w-6 h-6 text-green-500" />
                          </div>
                          <div className="flex items-center gap-3 mb-3">
                            <div className={`p-2 rounded-lg ${theme === 'dark' ? 'bg-red-500/20' : 'bg-red-50'}`}>
                              <Car className="w-6 h-6 text-red-500" />
                            </div>
                            <div>
                              <h3 className={`text-2xl font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>O₃</h3>
                              <p className={`text-xs mt-0.5 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>Sentinel-5P OFFL</p>
                            </div>
                          </div>
                          <div className="mt-4">
                            <div className="flex items-baseline gap-2 mb-2">
                              <span className={`text-4xl font-bold ${theme === 'dark' ? 'text-green-400' : 'text-green-600'}`}>
                                {liveData.predictions[0].O3_target.toFixed(1)}
                              </span>
                              <span className={`text-sm font-medium ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
                                µg/m³
                              </span>
                            </div>
                            <p className={`text-base font-semibold ${theme === 'dark' ? 'text-green-400' : 'text-green-600'}`}>Good</p>
                          </div>
                        </div>

                        {/* O3 Card - Sentinel-5P NRTI */}
                        <div className={`relative rounded-lg border p-5 shadow-sm transition-all hover:shadow-md ${theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-slate-50 border-slate-200'}`}>
                          <div className="absolute top-4 right-4">
                            <CheckCircle2 className="w-6 h-6 text-green-500" />
                          </div>
                          <div className="flex items-center gap-3 mb-3">
                            <div className={`p-2 rounded-lg ${theme === 'dark' ? 'bg-red-500/20' : 'bg-red-50'}`}>
                              <Car className="w-6 h-6 text-red-500" />
                            </div>
                            <div>
                              <h3 className={`text-2xl font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>O₃</h3>
                              <p className={`text-xs mt-0.5 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>Sentinel-5P NRTI</p>
                            </div>
                          </div>
                          <div className="mt-4">
                            <div className="flex items-baseline gap-2 mb-2">
                              <span className={`text-4xl font-bold ${theme === 'dark' ? 'text-green-400' : 'text-green-600'}`}>
                                {liveData.predictions[0].O3_target.toFixed(1)}
                              </span>
                              <span className={`text-sm font-medium ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
                                µg/m³
                              </span>
                            </div>
                            <p className={`text-base font-semibold ${theme === 'dark' ? 'text-green-400' : 'text-green-600'}`}>Good</p>
                          </div>
                        </div>

                        {/* PM2.5 Card - Sentinel-5P OFFL */}
                        <div className={`relative rounded-lg border p-5 shadow-sm transition-all hover:shadow-md ${theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-slate-50 border-slate-200'}`}>
                          <div className="absolute top-4 right-4">
                            <CheckCircle2 className="w-6 h-6 text-green-500" />
                          </div>
                          <div className="flex items-center gap-3 mb-3">
                            <div className={`p-2 rounded-lg ${theme === 'dark' ? 'bg-red-500/20' : 'bg-red-50'}`}>
                              <Car className="w-6 h-6 text-red-500" />
                            </div>
                            <div>
                              <h3 className={`text-2xl font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>PM₂.₅</h3>
                              <p className={`text-xs mt-0.5 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>Sentinel-5P OFFL</p>
                            </div>
                          </div>
                          <div className="mt-4">
                            <div className="flex items-baseline gap-2 mb-2">
                              <span className={`text-4xl font-bold ${theme === 'dark' ? 'text-green-400' : 'text-green-600'}`}>
                                {liveData.predictions[0].PM25_target.toFixed(1)}
                              </span>
                              <span className={`text-sm font-medium ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
                                µg/m³
                              </span>
                            </div>
                            <p className={`text-base font-semibold ${theme === 'dark' ? 'text-green-400' : 'text-green-600'}`}>Good</p>
                          </div>
                        </div>

                        {/* PM2.5 Card - Sentinel-5P NRTI */}
                        <div className={`relative rounded-lg border p-5 shadow-sm transition-all hover:shadow-md ${theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-slate-50 border-slate-200'}`}>
                          <div className="absolute top-4 right-4">
                            <CheckCircle2 className="w-6 h-6 text-green-500" />
                          </div>
                          <div className="flex items-center gap-3 mb-3">
                            <div className={`p-2 rounded-lg ${theme === 'dark' ? 'bg-red-500/20' : 'bg-red-50'}`}>
                              <Car className="w-6 h-6 text-red-500" />
                            </div>
                            <div>
                              <h3 className={`text-2xl font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>PM₂.₅</h3>
                              <p className={`text-xs mt-0.5 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>Sentinel-5P NRTI</p>
                            </div>
                          </div>
                          <div className="mt-4">
                            <div className="flex items-baseline gap-2 mb-2">
                              <span className={`text-4xl font-bold ${theme === 'dark' ? 'text-green-400' : 'text-green-600'}`}>
                                {liveData.predictions[0].PM25_target.toFixed(1)}
                              </span>
                              <span className={`text-sm font-medium ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
                                µg/m³
                              </span>
                            </div>
                            <p className={`text-base font-semibold ${theme === 'dark' ? 'text-green-400' : 'text-green-600'}`}>Good</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Delhi Heatmaps */}
                  <DelhiAirMap />

                  {/* Time Range Selector & Trend Chart */}
                  <div className={`rounded-xl border p-6 ${theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
                    <div className="flex items-center justify-between mb-6 flex-wrap gap-4">
                      <h2 className={`text-lg font-semibold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>Pollution Trends</h2>
                      <div className="flex items-center gap-4 flex-wrap">
                        {/* Gas Selector */}
                        <div className={`flex items-center gap-2 flex-wrap ${theme === 'dark' ? 'bg-slate-700' : 'bg-slate-100'} rounded-lg p-2`}>
                          <span className={`text-xs font-medium px-2 ${theme === 'dark' ? 'text-slate-300' : 'text-slate-600'}`}>Gases:</span>
                          {gasDefinitions.map(gas => (
                            <label
                              key={gas.id}
                              className={`flex items-center gap-1.5 px-2 py-1 rounded cursor-pointer transition-all ${
                                visibleGases.has(gas.id)
                                  ? theme === 'dark' ? 'bg-slate-600' : 'bg-slate-200'
                                  : theme === 'dark' ? 'hover:bg-slate-600/50' : 'hover:bg-slate-200/50'
                              }`}
                            >
                              <input
                                type="checkbox"
                                checked={visibleGases.has(gas.id)}
                                onChange={() => handleGasToggle(gas.id)}
                                className="cursor-pointer"
                              />
                              <span
                                className="text-xs font-medium"
                                style={{ color: visibleGases.has(gas.id) ? gas.color : theme === 'dark' ? '#94a3b8' : '#64748b' }}
                              >
                                {gas.label}
                              </span>
                            </label>
                          ))}
                        </div>
                        {/* Time Range Selector */}
                        <div className={`flex gap-2 rounded-lg p-1 ${theme === 'dark' ? 'bg-slate-700' : 'bg-slate-100'}`}>
                          {[1, 24, 48].map((hours) => (
                            <button
                              key={hours}
                              onClick={() => setTimeRange(hours as TimeRange)}
                              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                                timeRange === hours
                                  ? 'bg-cyan-600 text-white'
                                  : theme === 'dark'
                                  ? 'text-slate-300 hover:bg-slate-600'
                                  : 'text-slate-600 hover:bg-slate-200'
                              }`}
                            >
                              {hours}h
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                    <div style={{ height: '400px' }}>
                      <ResponsiveLine
                        data={nivoData}
                        margin={{ top: 50, right: 110, bottom: 50, left: 60 }}
                        xScale={{ type: 'point' }}
                        yScale={{
                          type: 'linear',
                          min: 'auto',
                          max: 'auto',
                          stacked: false,
                          reverse: false
                        }}
                        curve="monotoneX"
                        colors={(d: any) => d.color}
                        lineWidth={((d: any) => d.lineWidth || 2) as any}
                        // Use original color for points (not grayed out)
                        axisTop={null}
                        axisRight={null}
                        axisBottom={{
                          tickSize: 5,
                          tickPadding: 5,
                          tickRotation: 0,
                          legend: 'Time',
                          legendOffset: 36,
                          legendPosition: 'middle',
                          tickValues: trendData.length > 12 
                            ? trendData.filter((_, i) => i % Math.ceil(trendData.length / 12) === 0).map(d => d.time)
                            : trendData.map(d => d.time)
                        }}
                        axisLeft={{
                          tickSize: 5,
                          tickPadding: 5,
                          tickRotation: 0,
                          legend: 'Concentration',
                          legendOffset: -50,
                          legendPosition: 'middle'
                        }}
                        enablePoints={true}
                        pointSize={6}
                        pointColor={(point: any) => {
                          // Find the series by matching the data structure - ensure each curve has its own color
                          const seriesId = point.serieId || point.seriesId || point.id;
                          const series = nivoData.find(s => s.id === seriesId);
                          
                          // First try to get originalColor from series (preserves color even when grayed out)
                          if (series && (series as any)?.originalColor) {
                            return (series as any).originalColor;
                          }
                          
                          // If no originalColor, try to get from gasDefinitions directly
                          const gasDef = gasDefinitions.find(g => g.id === seriesId);
                          if (gasDef) {
                            return gasDef.color;
                          }
                          
                          // Fallback to series color or point color
                          if (series) {
                            return series.color || '#94a3b8';
                          }
                          
                          return point.serieColor || point.seriesColor || point.color || '#94a3b8';
                        }}
                        pointBorderWidth={1.5}
                        pointBorderColor={(point: any) => {
                          // Use white border for dark theme, dark border for light theme - more prominent like reference
                          return theme === 'dark' ? '#ffffff' : '#1e293b';
                        }}
                        pointLabelYOffset={-12}
                        enablePointLabel={false}
                        isInteractive={true}
                        animate={true}
                        enableArea={false}
                        useMesh={false}
                        legends={[
                          {
                            anchor: 'bottom-right',
                            direction: 'column',
                            justify: false,
                            translateX: 100,
                            translateY: 0,
                            itemsSpacing: 4,
                            itemDirection: 'left-to-right',
                            itemWidth: 90,
                            itemHeight: 24,
                            itemOpacity: 0.75,
                            symbolSize: 14,
                            symbolShape: 'circle',
                            onClick: (datum) => handleGasClick(datum.id as string),
                            effects: [
                              {
                                on: 'hover',
                                style: {
                                  itemBackground: 'rgba(0, 0, 0, .03)',
                                  itemOpacity: 1,
                                  itemTextColor: theme === 'dark' ? '#f1f5f9' : '#1e293b'
                                }
                              }
                            ]
                          }
                        ]}
                        theme={{
                          background: 'transparent',
                          text: {
                            fontSize: 12,
                            fill: theme === 'dark' ? '#94a3b8' : '#64748b',
                            fontFamily: 'inherit'
                          },
                          axis: {
                            domain: {
                              line: {
                                stroke: theme === 'dark' ? '#334155' : '#e2e8f0',
                                strokeWidth: 1
                              }
                            },
                            legend: {
                              text: {
                                fontSize: 12,
                                fill: theme === 'dark' ? '#94a3b8' : '#64748b',
                                fontFamily: 'inherit'
                              }
                            },
                            ticks: {
                              line: {
                                stroke: theme === 'dark' ? '#334155' : '#e2e8f0',
                                strokeWidth: 1
                              },
                              text: {
                                fontSize: 11,
                                fill: theme === 'dark' ? '#94a3b8' : '#64748b',
                                fontFamily: 'inherit'
                              }
                            }
                          },
                          grid: {
                            line: {
                              stroke: theme === 'dark' ? '#334155' : '#e2e8f0',
                              strokeWidth: 1,
                              strokeDasharray: '3 3'
                            }
                          },
                          tooltip: {
                            container: {
                              background: theme === 'dark' ? '#1e293b' : '#ffffff',
                              color: theme === 'dark' ? '#f1f5f9' : '#1e293b',
                              fontSize: '14px',
                              border: `1px solid ${theme === 'dark' ? '#475569' : '#e2e8f0'}`,
                              borderRadius: '12px',
                              padding: '12px 16px',
                              boxShadow: theme === 'dark' 
                                ? '0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)'
                                : '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                              minWidth: '200px'
                            }
                          }
                        }}
                        layers={['grid', 'markers', 'areas', 'lines', 'axes', 'points', 'legends', 'slices', 'crosshair']}
                        enableSlices="x"
                        sliceTooltip={({ slice }) => {
                          const sortedPoints = [...slice.points].sort((a, b) => {
                            const aId = (a as any).seriesId || (a as any).serieId || '';
                            const bId = (b as any).seriesId || (b as any).serieId || '';
                            return aId.localeCompare(bId);
                          });
                          
                          return (
                            <div
                              style={{
                                background: theme === 'dark' ? '#1e293b' : '#ffffff',
                                padding: '14px 18px',
                                border: `1px solid ${theme === 'dark' ? '#475569' : '#e2e8f0'}`,
                                borderRadius: '12px',
                                fontSize: '14px',
                                color: theme === 'dark' ? '#f1f5f9' : '#1e293b',
                                boxShadow: theme === 'dark' 
                                  ? '0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)'
                                  : '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                                minWidth: '220px'
                              }}
                            >
                              <div style={{ 
                                marginBottom: '10px', 
                                fontWeight: 'bold', 
                                fontSize: '15px',
                                paddingBottom: '8px',
                                borderBottom: `1px solid ${theme === 'dark' ? '#475569' : '#e2e8f0'}`
                              }}>
                                Time: {slice.points[0].data.x}
                              </div>
                              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                {sortedPoints.map(point => {
                                  const pointId = (point as any).seriesId || (point as any).serieId || '';
                                  const pointColor = (point as any).seriesColor || (point as any).serieColor || '#94a3b8';
                                  const isGrayedOut = selectedGas !== null && selectedGas !== pointId;
                                  
                                  return (
                                    <div 
                                      key={point.id} 
                                      style={{ 
                                        display: 'flex', 
                                        alignItems: 'center', 
                                        justifyContent: 'space-between',
                                        gap: '12px',
                                        padding: '6px 8px',
                                        borderRadius: '6px',
                                        backgroundColor: isGrayedOut 
                                          ? (theme === 'dark' ? 'rgba(148, 163, 184, 0.1)' : 'rgba(148, 163, 184, 0.05)')
                                          : 'transparent',
                                        opacity: isGrayedOut ? 0.6 : 1
                                      }}
                                    >
                                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flex: 1 }}>
                                        <div
                                          style={{
                                            width: '14px',
                                            height: '14px',
                                            borderRadius: '50%',
                                            background: pointColor,
                                            border: `2px solid ${theme === 'dark' ? '#1e293b' : '#ffffff'}`,
                                            boxShadow: `0 0 0 1px ${pointColor}40`
                                          }}
                                        />
                                        <span style={{ fontWeight: '500' }}>{pointId}</span>
                                      </div>
                                      <span style={{ 
                                        fontWeight: '600',
                                        color: pointColor,
                                        fontSize: '15px'
                                      }}>
                                        {point.data.y.toFixed(1)}
                                      </span>
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          );
                        }}
                      />
                    </div>
                  </div>

                  {/* Pollutant Gauges Grid - FIXED HERE */}
                  <div>
                    <h2 className={`text-lg font-semibold mb-4 ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>Today's AQI</h2>
                    {/* UPDATED GRID CLASS below for better spacing */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-3 gap-6">
                      {pollutantMetrics && (
                        <>
                          <PollutantGauge name="O₃" value={liveData?.predictions[0].O3_target || 0} maxValue={150} unit="ppb" color="#06b6d4" />  
                          <PollutantGauge name="NO₂" value={liveData?.predictions[0].NO2_target || 0} maxValue={200} unit="ppb" color="#3b82f6" />
                          <PollutantGauge name="HCHO" value={liveData?.predictions[0].HCHO_target || 0} maxValue={150} unit="ppb" color="#fbbf24" />
                          <PollutantGauge name="CO" value={liveData?.predictions[0].CO_target || 0} maxValue={5000} unit="ppm" color="#ef4444" />
                          <PollutantGauge name="PM2.5" value={liveData?.predictions[0].PM25_target || 0} maxValue={150} unit="µg/m³" color="#f59e0b" />
                          <PollutantGauge name="PM10" value={liveData?.predictions[0].PM10_target || 0} maxValue={250} unit="µg/m³" color="#f97316" />
                          </>
                      )}
                    </div>
                  </div>

                  {/* Highest Pollutants */}
                  <HighestPollutants
                    o3Events={[
                      { name: 'Central', location: 'Delhi/National Capital Region', value: '85 µg/m³' },
                      { name: 'East', location: 'Delhi/National Capital Region', value: '82 µg/m³' },
                      { name: 'South', location: 'Delhi/National Capital Region', value: '78 µg/m³' },
                    ]}
                    no2Events={[
                      { name: 'Central', location: 'Delhi/National Capital Region', value: '95 µg/m³' },
                      { name: 'North', location: 'Delhi/National Capital Region', value: '92 µg/m³' },
                      { name: 'West', location: 'Delhi/National Capital Region', value: '88 µg/m³' },
                    ]}
                  />

                  {/* Station Info */}
                  {liveData?.live_source && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className={`rounded-xl border p-4 ${theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
                        <div className="flex items-center gap-2 mb-4">
                          <MapPin className="w-5 h-5 text-cyan-500" />
                          <h3 className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>Station Information</h3>
                        </div>
                        <div className={`space-y-2 text-sm ${theme === 'dark' ? 'text-slate-300' : 'text-slate-600'}`}>
                          <p><span className={theme === 'dark' ? 'text-slate-400' : 'text-slate-500'}>Station:</span> {liveData.live_source.station_name}</p>
                          <p><span className={theme === 'dark' ? 'text-slate-400' : 'text-slate-500'}>Location:</span> {liveData.live_source.station_location[0].toFixed(4)}°N, {liveData.live_source.station_location[1].toFixed(4)}°E</p>
                          <p><span className={theme === 'dark' ? 'text-slate-400' : 'text-slate-500'}>AQI:</span> <span className="text-amber-400 font-semibold">{liveData.live_source.overall_aqi}</span></p>
                        </div>
                      </div>

                      <div className={`rounded-xl border p-4 ${theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
                        <div className="flex items-center gap-2 mb-4">
                          <Clock className="w-5 h-5 text-cyan-500" />
                          <h3 className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>Measurement Time</h3>
                        </div>
                        <div className={`space-y-2 text-sm ${theme === 'dark' ? 'text-slate-300' : 'text-slate-600'}`}>
                          <p><span className={theme === 'dark' ? 'text-slate-400' : 'text-slate-500'}>Time:</span> {liveData.live_source.measurement_time}</p>
                          <p><span className={theme === 'dark' ? 'text-slate-400' : 'text-slate-500'}>Timezone:</span> {liveData.live_source.timezone}</p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Other Tab Content (Health, etc.) */}
              {activeTab === 'health' && healthStatus && modelDetail && (
                <HealthStatus 
                  healthStatus={healthStatus}
                  modelDetail={modelDetail}
                />
              )}
            </>
          )}
        </div>
      </main>
    </div>
  );
}