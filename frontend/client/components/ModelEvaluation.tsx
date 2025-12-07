'use client';

import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { useTheme } from '@/contexts/ThemeContext';
import { Filter, ChevronDown, Loader2 } from 'lucide-react';
import apiClient from '@/services/apiClient';
import { MetricsResponse, SiteMetrics } from '@shared/api';

type PollutantType = 'O3' | 'NO2';
type MetricType = 'RMSE' | 'MAE' | 'R2' | 'RIA';

const ModelEvaluation = () => {
  const { theme } = useTheme();
  const [metricsData, setMetricsData] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedSiteId, setSelectedSiteId] = useState<number>(1);
  const [selectedPollutant, setSelectedPollutant] = useState<PollutantType>('O3');
  const [selectedMetric, setSelectedMetric] = useState<MetricType>('RMSE');
  const [isSiteDropdownOpen, setIsSiteDropdownOpen] = useState(false);
  const [isPollutantDropdownOpen, setIsPollutantDropdownOpen] = useState(false);
  const [isMetricDropdownOpen, setIsMetricDropdownOpen] = useState(false);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await apiClient.getMetrics();
        setMetricsData(data);
        if (data.sites.length > 0) {
          setSelectedSiteId(data.sites[0].site_id);
        }
      } catch (err: any) {
        const errorMessage = err?.message || 'Failed to fetch metrics. Make sure the backend server is running on http://localhost:8000';
        setError(errorMessage);
        console.error('Error fetching metrics:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
  }, []);

  const pollutantOptions: { value: PollutantType; label: string }[] = [
    { value: 'O3', label: 'O₃' },
    { value: 'NO2', label: 'NO₂' },
  ];

  const metricOptions: { value: MetricType; label: string }[] = [
    { value: 'RMSE', label: 'RMSE' },
    { value: 'MAE', label: 'MAE' },
    { value: 'R2', label: 'R²' },
    { value: 'RIA', label: 'RIA' },
  ];

  const currentSite = metricsData?.sites.find(s => s.site_id === selectedSiteId);
  const currentMetrics = currentSite?.[selectedPollutant.toLowerCase() as 'o3' | 'no2'];

  const pollutantColors: Record<PollutantType, string> = {
    O3: '#06b6d4',
    NO2: '#2563eb',
  };

  const pollutantLabels: Record<PollutantType, string> = {
    O3: 'O₃',
    NO2: 'NO₂',
  };

  const pollutantUnits: Record<PollutantType, string> = {
    O3: 'ppb',
    NO2: 'ppb',
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="w-8 h-8 animate-spin text-cyan-500" />
          <p className={`text-sm ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
            Loading model metrics...
          </p>
        </div>
      </div>
    );
  }

  if (error || !metricsData || !currentSite || !currentMetrics) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Card className={`border p-6 rounded-xl ${theme === 'dark' ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-white'}`}>
          <div className="space-y-4 text-center">
            <p className={`text-lg font-semibold ${theme === 'dark' ? 'text-red-400' : 'text-red-600'}`}>
              {error || 'Failed to load metrics data'}
            </p>
            {error && (
              <div className={`text-sm space-y-2 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
                <p>Please ensure:</p>
                <ul className="list-disc list-inside space-y-1">
                  <li>The backend server is running on http://localhost:8000</li>
                  <li>The endpoint /api/v1/predict/metrics is accessible</li>
                  <li>There are no CORS issues</li>
                </ul>
                <button
                  onClick={() => {
                    setError(null);
                    setLoading(true);
                    apiClient.getMetrics()
                      .then(data => {
                        setMetricsData(data);
                        if (data.sites.length > 0) {
                          setSelectedSiteId(data.sites[0].site_id);
                        }
                        setLoading(false);
                      })
                      .catch(err => {
                        setError(err?.message || 'Failed to fetch metrics');
                        setLoading(false);
                      });
                  }}
                  className={`mt-4 px-4 py-2 rounded-lg font-medium transition-colors ${theme === 'dark'
                    ? 'bg-cyan-600 hover:bg-cyan-700 text-white'
                    : 'bg-cyan-500 hover:bg-cyan-600 text-white'
                    }`}
                >
                  Retry
                </button>
              </div>
            )}
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Site and Pollutant Selector */}
      <Card className={`border p-4 rounded-xl ${theme === 'dark' ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-white'}`}>
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-4">
            <div className="relative">
              <button
                onClick={() => {
                  setIsSiteDropdownOpen(!isSiteDropdownOpen);
                  setIsPollutantDropdownOpen(false);
                  setIsMetricDropdownOpen(false);
                }}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all border ${theme === 'dark'
                  ? 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
                  : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-50'
                  }`}
              >
                <Filter className="w-4 h-4" />
                <span>Site {selectedSiteId}</span>
                <ChevronDown className={`w-4 h-4 transition-transform ${isSiteDropdownOpen ? 'rotate-180' : ''}`} />
              </button>

              {isSiteDropdownOpen && (
                <>
                  <div
                    className="fixed inset-0 z-10"
                    onClick={() => setIsSiteDropdownOpen(false)}
                  />
                  <div
                    className={`absolute top-full left-0 mt-2 z-20 min-w-[150px] rounded-lg border shadow-lg ${theme === 'dark'
                      ? 'bg-slate-800 border-slate-700'
                      : 'bg-white border-slate-200'
                      }`}
                  >
                    <div className="py-1">
                      {metricsData.sites.map((site) => {
                        const isSelected = selectedSiteId === site.site_id;
                        return (
                          <button
                            key={site.site_id}
                            onClick={() => {
                              setSelectedSiteId(site.site_id);
                              setIsSiteDropdownOpen(false);
                            }}
                            className={`w-full text-left px-4 py-2 text-sm transition-colors ${isSelected
                              ? theme === 'dark'
                                ? 'bg-cyan-600/20 text-cyan-400'
                                : 'bg-cyan-50 text-cyan-600'
                              : theme === 'dark'
                                ? 'text-slate-300 hover:bg-slate-700'
                                : 'text-slate-700 hover:bg-slate-50'
                              }`}
                          >
                            Site {site.site_id}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </>
              )}
            </div>

            <div className="relative">
              <button
                onClick={() => {
                  setIsPollutantDropdownOpen(!isPollutantDropdownOpen);
                  setIsSiteDropdownOpen(false);
                  setIsMetricDropdownOpen(false);
                }}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all border ${theme === 'dark'
                  ? 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
                  : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-50'
                  }`}
              >
                <Filter className="w-4 h-4" />
                <span>{pollutantLabels[selectedPollutant]}</span>
                <ChevronDown className={`w-4 h-4 transition-transform ${isPollutantDropdownOpen ? 'rotate-180' : ''}`} />
              </button>

              {isPollutantDropdownOpen && (
                <>
                  <div
                    className="fixed inset-0 z-10"
                    onClick={() => setIsPollutantDropdownOpen(false)}
                  />
                  <div
                    className={`absolute top-full left-0 mt-2 z-20 min-w-[150px] rounded-lg border shadow-lg ${theme === 'dark'
                      ? 'bg-slate-800 border-slate-700'
                      : 'bg-white border-slate-200'
                      }`}
                  >
                    <div className="py-1">
                      {pollutantOptions.map((option) => {
                        const isSelected = selectedPollutant === option.value;
                        return (
                          <button
                            key={option.value}
                            onClick={() => {
                              setSelectedPollutant(option.value);
                              setIsPollutantDropdownOpen(false);
                            }}
                            className={`w-full text-left px-4 py-2 text-sm transition-colors ${isSelected
                              ? theme === 'dark'
                                ? 'bg-cyan-600/20 text-cyan-400'
                                : 'bg-cyan-50 text-cyan-600'
                              : theme === 'dark'
                                ? 'text-slate-300 hover:bg-slate-700'
                                : 'text-slate-700 hover:bg-slate-50'
                              }`}
                          >
                            {option.label}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>

          {metricsData.generated_at && (
            <p className={`text-xs ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
              Generated: {new Date(metricsData.generated_at).toLocaleString()}
            </p>
          )}
        </div>
      </Card>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* RMSE Card */}
        <Card className={`border p-6 rounded-xl ${theme === 'dark' ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-white'}`}>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className={`text-sm font-semibold uppercase tracking-wide ${theme === 'dark' ? 'text-slate-300' : 'text-slate-700'}`}>
                  RMSE
                </h3>
                <p className={`text-xs mt-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
                  {pollutantLabels[selectedPollutant]}
                </p>
              </div>
              <Badge className="bg-cyan-600/20 text-cyan-400 border-0">
                {pollutantUnits[selectedPollutant]}
              </Badge>
            </div>
            <div className="flex items-end gap-2">
              <span className={`text-5xl font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>
                {currentMetrics.rmse.toFixed(4)}
              </span>
              <span className={`text-lg mb-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
                {pollutantUnits[selectedPollutant]}
              </span>
            </div>
            <p className={`text-xs ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
              Root Mean Square Error - measures prediction error magnitude
            </p>
          </div>
        </Card>

        {/* MAE Card */}
        <Card className={`border p-6 rounded-xl ${theme === 'dark' ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-white'}`}>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className={`text-sm font-semibold uppercase tracking-wide ${theme === 'dark' ? 'text-slate-300' : 'text-slate-700'}`}>
                  MAE
                </h3>
                <p className={`text-xs mt-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
                  {pollutantLabels[selectedPollutant]}
                </p>
              </div>
              <Badge className="bg-blue-600/20 text-blue-400 border-0">
                {pollutantUnits[selectedPollutant]}
              </Badge>
            </div>
            <div className="flex items-end gap-2">
              <span className={`text-5xl font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>
                {currentMetrics.mae.toFixed(4)}
              </span>
              <span className={`text-lg mb-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
                {pollutantUnits[selectedPollutant]}
              </span>
            </div>
            <p className={`text-xs ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
              Mean Absolute Error - average absolute deviation
            </p>
          </div>
        </Card>

        {/* R² Card */}
        <Card className={`border p-6 rounded-xl ${theme === 'dark' ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-white'}`}>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className={`text-sm font-semibold uppercase tracking-wide ${theme === 'dark' ? 'text-slate-300' : 'text-slate-700'}`}>
                  R²
                </h3>
                <p className={`text-xs mt-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
                  {pollutantLabels[selectedPollutant]}
                </p>
              </div>
              <Badge className="bg-emerald-600/20 text-emerald-400 border-0">
                Score
              </Badge>
            </div>
            <div className="flex items-end gap-2">
              <span className={`text-5xl font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>
                {currentMetrics.r2.toFixed(4)}
              </span>
              <span className={`text-lg mb-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>out of 1.0</span>
            </div>
            <p className={`text-xs ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
              Coefficient of Determination - explained variance ratio
            </p>
          </div>
        </Card>
      </div>

      {/* Additional Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {/* RIA Card */}
        <Card className={`border p-6 rounded-xl ${theme === 'dark' ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-white'}`}>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className={`text-sm font-semibold uppercase tracking-wide ${theme === 'dark' ? 'text-slate-300' : 'text-slate-700'}`}>
                  RIA
                </h3>
                <p className={`text-xs mt-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
                  {pollutantLabels[selectedPollutant]}
                </p>
              </div>
              <Badge className="bg-purple-600/20 text-purple-400 border-0">
                Score
              </Badge>
            </div>
            <div className="flex items-end gap-2">
              <span className={`text-4xl font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>
                {currentMetrics.ria.toFixed(4)}
              </span>
            </div>
            <p className={`text-xs ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
              Relative Index of Agreement
            </p>
          </div>
        </Card>

        {/* Bias Card */}
        <Card className={`border p-6 rounded-xl ${theme === 'dark' ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-white'}`}>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className={`text-sm font-semibold uppercase tracking-wide ${theme === 'dark' ? 'text-slate-300' : 'text-slate-700'}`}>
                  Bias
                </h3>
                <p className={`text-xs mt-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
                  {pollutantLabels[selectedPollutant]}
                </p>
              </div>
              <Badge className="bg-orange-600/20 text-orange-400 border-0">
                {pollutantUnits[selectedPollutant]}
              </Badge>
            </div>
            <div className="flex items-end gap-2">
              <span className={`text-4xl font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>
                {currentMetrics.bias.toFixed(4)}
              </span>
            </div>
            <p className={`text-xs ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
              Mean bias error
            </p>
          </div>
        </Card>

        {/* Baseline RMSE Card */}
        <Card className={`border p-6 rounded-xl ${theme === 'dark' ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-white'}`}>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className={`text-sm font-semibold uppercase tracking-wide ${theme === 'dark' ? 'text-slate-300' : 'text-slate-700'}`}>
                  Baseline RMSE
                </h3>
                <p className={`text-xs mt-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
                  {pollutantLabels[selectedPollutant]}
                </p>
              </div>
              <Badge className="bg-red-600/20 text-red-400 border-0">
                {pollutantUnits[selectedPollutant]}
              </Badge>
            </div>
            <div className="flex items-end gap-2">
              <span className={`text-4xl font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>
                {currentMetrics.baseline_rmse.toFixed(4)}
              </span>
            </div>
            <p className={`text-xs ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
              Baseline model RMSE
            </p>
          </div>
        </Card>

        {/* RMSE Reduction Card */}
        <Card className={`border p-6 rounded-xl ${theme === 'dark' ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-white'}`}>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className={`text-sm font-semibold uppercase tracking-wide ${theme === 'dark' ? 'text-slate-300' : 'text-slate-700'}`}>
                  RMSE Reduction
                </h3>
                <p className={`text-xs mt-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
                  {pollutantLabels[selectedPollutant]}
                </p>
              </div>
              <Badge className="bg-green-600/20 text-green-400 border-0">
                %
              </Badge>
            </div>
            <div className="flex items-end gap-2">
              <span className={`text-4xl font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>
                {currentMetrics.rmse_reduction_pct.toFixed(2)}
              </span>
              <span className={`text-lg mb-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>%</span>
            </div>
            <p className={`text-xs ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
              Improvement over baseline
            </p>
          </div>
        </Card>
      </div>

      {/* Metrics Comparison Across Sites Chart */}
      <Card className={`border p-6 rounded-xl ${theme === 'dark' ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-white'}`}>
        <div className="flex items-center justify-between mb-6">
          <h3 className={`text-lg font-semibold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>
            {selectedMetric === 'R2' ? 'R²' : selectedMetric === 'RIA' ? 'RIA' : selectedMetric} Comparison - {pollutantLabels[selectedPollutant]}
          </h3>
          <div className="flex items-center gap-2">
            <div className="relative">
              <button
                onClick={() => {
                  setIsMetricDropdownOpen(!isMetricDropdownOpen);
                  setIsPollutantDropdownOpen(false);
                }}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all border ${theme === 'dark'
                  ? 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
                  : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-50'
                  }`}
              >
                <Filter className="w-4 h-4" />
                <span>{selectedMetric === 'R2' ? 'R²' : selectedMetric === 'RIA' ? 'RIA' : selectedMetric}</span>
                <ChevronDown className={`w-4 h-4 transition-transform ${isMetricDropdownOpen ? 'rotate-180' : ''}`} />
              </button>

              {isMetricDropdownOpen && (
                <>
                  <div
                    className="fixed inset-0 z-10"
                    onClick={() => setIsMetricDropdownOpen(false)}
                  />
                  <div
                    className={`absolute top-full right-0 mt-2 z-20 min-w-[150px] rounded-lg border shadow-lg ${theme === 'dark'
                      ? 'bg-slate-800 border-slate-700'
                      : 'bg-white border-slate-200'
                      }`}
                  >
                    <div className="py-1">
                      {metricOptions.map((option) => {
                        const isSelected = selectedMetric === option.value;
                        return (
                          <button
                            key={option.value}
                            onClick={() => {
                              setSelectedMetric(option.value);
                              setIsMetricDropdownOpen(false);
                            }}
                            className={`w-full text-left px-4 py-2 text-sm transition-colors ${isSelected
                              ? theme === 'dark'
                                ? 'bg-cyan-600/20 text-cyan-400'
                                : 'bg-cyan-50 text-cyan-600'
                              : theme === 'dark'
                                ? 'text-slate-300 hover:bg-slate-700'
                                : 'text-slate-700 hover:bg-slate-50'
                              }`}
                          >
                            {option.label}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart
            data={metricsData.sites.map(site => ({
              site: `Site ${site.site_id}`,
              value: selectedMetric === 'RMSE' ? site[selectedPollutant.toLowerCase() as 'o3' | 'no2'].rmse :
                selectedMetric === 'MAE' ? site[selectedPollutant.toLowerCase() as 'o3' | 'no2'].mae :
                  selectedMetric === 'R2' ? site[selectedPollutant.toLowerCase() as 'o3' | 'no2'].r2 :
                    site[selectedPollutant.toLowerCase() as 'o3' | 'no2'].ria,
            }))}
          >
            <CartesianGrid strokeDasharray="3 3" stroke={theme === 'dark' ? '#334155' : '#e2e8f0'} />
            <XAxis dataKey="site" stroke={theme === 'dark' ? '#94a3b8' : '#64748b'} />
            <YAxis
              stroke={theme === 'dark' ? '#94a3b8' : '#64748b'}
              domain={selectedMetric === 'R2' || selectedMetric === 'RIA' ? [0, 1] : undefined}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: theme === 'dark' ? '#1e293b' : '#ffffff',
                border: `1px solid ${theme === 'dark' ? '#475569' : '#e2e8f0'}`,
                borderRadius: '8px',
              }}
              labelStyle={{ color: theme === 'dark' ? '#f1f5f9' : '#1e293b' }}
              formatter={(value: number) => (selectedMetric === 'R2' || selectedMetric === 'RIA') ? value.toFixed(4) : value.toFixed(2)}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="value"
              stroke={pollutantColors[selectedPollutant]}
              strokeWidth={3}
              dot={{ fill: pollutantColors[selectedPollutant], r: 5 }}
              activeDot={{ r: 7 }}
              isAnimationActive={true}
              name={`${selectedMetric === 'R2' ? 'R²' : selectedMetric === 'RIA' ? 'RIA' : selectedMetric} - ${pollutantLabels[selectedPollutant]}`}
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Model Performance Info */}
      <Card className={`border p-6 rounded-xl ${theme === 'dark' ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-white'}`}>
        <h3 className={`text-lg font-semibold mb-4 ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>
          Model Performance Analysis - Site {selectedSiteId} - {pollutantLabels[selectedPollutant]}
        </h3>
        <div className={`grid grid-cols-1 md:grid-cols-2 gap-6 text-sm ${theme === 'dark' ? 'text-slate-300' : 'text-slate-700'}`}>
          <div className="space-y-3">
            <div>
              <p className={`font-medium mb-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>RMSE (Root Mean Square Error)</p>
              <p className={theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}>
                The RMSE of {currentMetrics.rmse.toFixed(4)} {pollutantUnits[selectedPollutant]} indicates the average magnitude of prediction
                errors. Lower values indicate better model performance.
              </p>
            </div>
            <div>
              <p className={`font-medium mb-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>MAE (Mean Absolute Error)</p>
              <p className={theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}>
                The MAE of {currentMetrics.mae.toFixed(4)} {pollutantUnits[selectedPollutant]} represents the average absolute difference
                between predicted and actual values.
              </p>
            </div>
            <div>
              <p className={`font-medium mb-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>Bias</p>
              <p className={theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}>
                The bias of {currentMetrics.bias.toFixed(4)} {pollutantUnits[selectedPollutant]} indicates the systematic error in predictions.
                A value close to zero indicates minimal bias.
              </p>
            </div>
          </div>
          <div className="space-y-3">
            <div>
              <p className={`font-medium mb-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>R² (Coefficient of Determination)</p>
              <p className={theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}>
                An R² value of {currentMetrics.r2.toFixed(4)} (on a scale of 0 to 1) indicates that the model
                explains {(currentMetrics.r2 * 100).toFixed(2)}% of the variance in the data.
              </p>
            </div>
            <div>
              <p className={`font-medium mb-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>RIA (Relative Index of Agreement)</p>
              <p className={theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}>
                The RIA value of {currentMetrics.ria.toFixed(4)} measures the model's agreement with observations.
                Values closer to 1.0 indicate better agreement.
              </p>
            </div>
            <div>
              <p className={`font-medium mb-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>Model Improvement</p>
              <p className="text-emerald-400">
                ✓ The model shows {currentMetrics.rmse_reduction_pct.toFixed(2)}% improvement over baseline (RMSE reduced from {currentMetrics.baseline_rmse.toFixed(2)} to {currentMetrics.rmse.toFixed(2)} {pollutantUnits[selectedPollutant]})
              </p>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default ModelEvaluation;
