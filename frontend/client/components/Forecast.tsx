'use client';

import React, { useEffect, useState } from 'react';
import Header from '@/components/Header';
import { RefreshCw, AlertCircle } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';
import { ModelHealthResponse, PredictResponse, ModelDetailResponse } from '@shared/api';
import { generateDummy24HourForecast, generateDummyHealthStatus, generateDummyModelDetail } from '@/services/dummyData';
import SiteSelector from '@/components/SiteSelector';
import HealthStatus from '@/components/HealthStatus';
import ForecastChart from './ForecastChart';
import InteractiveForecastTool from './InteractiveForecastTool';

type TabType = 'forecast' | 'tools' | 'health';
type TimeRange = 24 | 48;

export default function Forecast() {
  const { theme, toggleTheme } = useTheme();
  const [selectedSite, setSelectedSite] = useState(1);
  const [activeTab, setActiveTab] = useState<TabType>('forecast');
  const [timeRange, setTimeRange] = useState<TimeRange>(24);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [forecastData, setForecastData] = useState<PredictResponse | null>(null);
  const [healthStatus, setHealthStatus] = useState<ModelHealthResponse | null>(null);
  const [modelDetail, setModelDetail] = useState<ModelDetailResponse | null>(null);

  const fetchAllData = async (siteId: number, hours: TimeRange = 24) => {
    try {
      setLoading(true);
      setError(null);

      const [forecast, health, detail] = await Promise.all([
        Promise.resolve(generateDummy24HourForecast(siteId, hours)),
        Promise.resolve(generateDummyHealthStatus()),
        Promise.resolve(generateDummyModelDetail(siteId)),
      ]);

      setForecastData(forecast);
      setHealthStatus(health);
      setModelDetail(detail);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch data';
      setError(errorMessage);
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = () => fetchAllData(selectedSite, timeRange);

  useEffect(() => {
    fetchAllData(selectedSite, timeRange);
  }, [selectedSite, timeRange]);

  return (
    <>
      <Header
        title="SkySense Forecast"
        subtitle="Forecast, tools, and health status for selected site."
        onRefresh={handleRefresh}
        loading={loading}
        showRefresh={true}
        theme={theme}
        onThemeToggle={toggleTheme}
      />

      <div className={`p-8 min-h-full ${theme === 'dark' ? 'bg-slate-950' : 'bg-slate-50'}`}>
        {error && (
          <div
            className={`mb-6 p-4 border rounded-lg flex items-start gap-3 ${
              theme === 'dark' ? 'bg-red-900/20 border-red-700' : 'bg-red-50 border-red-300'
            }`}
          >
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
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
              <div
                className={`flex gap-2 rounded-lg p-1 border w-fit ${
                  theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-slate-100 border-slate-300'
                }`}
              >
                {[
                  { id: 'forecast', label: 'Forecast' },
                  { id: 'tools', label: 'Interactive Tool' },
                  { id: 'health', label: 'Health Status' },
                ].map(({ id, label }) => (
                  <button
                    key={id}
                    onClick={() => setActiveTab(id as TabType)}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                      activeTab === id
                        ? 'bg-cyan-600 text-white shadow-sm'
                        : theme === 'dark'
                        ? 'text-slate-300 hover:text-white'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>

              <div className="w-full md:w-auto md:min-w-[280px]">
                <SiteSelector selectedSite={selectedSite} onSiteChange={setSelectedSite} disabled={loading} />
              </div>
            </div>

            {activeTab === 'forecast' && forecastData && (
              <div className="space-y-6">
                <div
                  className={`flex items-center gap-4 p-4 rounded-lg border ${
                    theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'
                  }`}
                >
                  <span className={`text-sm font-medium ${theme === 'dark' ? 'text-slate-300' : 'text-slate-700'}`}>
                    Forecast Period:
                  </span>
                  <div className={`flex gap-2 rounded-lg p-1 ${theme === 'dark' ? 'bg-slate-700' : 'bg-slate-100'}`}>
                    {[24, 48].map((hours) => (
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

                <ForecastChart data={forecastData} siteId={selectedSite} timeRange={timeRange} />

              </div>
            )}

            {activeTab === 'tools' && (
              <InteractiveForecastTool siteId={selectedSite} onForecastGenerated={setForecastData} />
            )}

            {activeTab === 'health' && healthStatus && modelDetail && (
              <HealthStatus healthStatus={healthStatus} modelDetail={modelDetail} />
            )}
          </>
        )}
      </div>
    </>
  );
}

