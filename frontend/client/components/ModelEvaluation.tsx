'use client';

import React, { useState } from 'react';
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
import { Filter, ChevronDown } from 'lucide-react';

interface PollutantEvalMetrics {
  pollutant: string;
  rmse: number;
  mae: number;
  r_mean: number;
  unit: string;
  color: string;
}

interface EvaluationDataPoint {
  time: string;
  predicted: number;
  actual: number;
}

interface RunningMeanDataPoint {
  date: string;
  rmse: number;
  mae: number;
  r2: number;
}

type PollutantType = 'NO2' | 'O3' | 'HCHO' | 'CO' | 'PM25' | 'PM10';
type MetricType = 'RMSE' | 'MAE' | 'R2';

const ModelEvaluation = () => {
  const { theme } = useTheme();
  const [selectedPollutant, setSelectedPollutant] = useState<PollutantType>('O3');
  const [selectedMetric, setSelectedMetric] = useState<MetricType>('RMSE');
  const [isPollutantDropdownOpen, setIsPollutantDropdownOpen] = useState(false);
  const [isMetricDropdownOpen, setIsMetricDropdownOpen] = useState(false);

  // Dummy metrics data for demonstration
  const metricsData: Record<PollutantType, PollutantEvalMetrics> = {
    O3: {
      pollutant: 'O₃',
      rmse: 2.0,
      mae: 10,
      r_mean: 0.92,
      unit: 'ppb',
      color: '#06b6d4',
    },
    NO2: {
      pollutant: 'NO₂',
      rmse: 3.5,
      mae: 15,
      r_mean: 0.88,
      unit: 'ppb',
      color: '#2563eb',
    },
    HCHO: {
      pollutant: 'HCHO',
      rmse: 1.8,
      mae: 8,
      r_mean: 0.90,
      unit: 'ppb',
      color: '#10b981',
    },
    CO: {
      pollutant: 'CO',
      rmse: 25.0,
      mae: 120,
      r_mean: 0.85,
      unit: 'ppm',
      color: '#f59e0b',
    },
    PM25: {
      pollutant: 'PM₂.₅',
      rmse: 4.2,
      mae: 18,
      r_mean: 0.87,
      unit: 'µg/m³',
      color: '#f97316',
    },
    PM10: {
      pollutant: 'PM₁₀',
      rmse: 6.5,
      mae: 25,
      r_mean: 0.84,
      unit: 'µg/m³',
      color: '#ef4444',
    },
  };

  const pollutantOptions: { value: PollutantType; label: string }[] = [
    { value: 'NO2', label: 'NO₂' },
    { value: 'O3', label: 'O₃' },
    { value: 'HCHO', label: 'HCHO' },
    { value: 'CO', label: 'CO' },
    { value: 'PM25', label: 'PM₂.₅' },
    { value: 'PM10', label: 'PM₁₀' },
  ];

  const metricOptions: { value: MetricType; label: string }[] = [
    { value: 'RMSE', label: 'RMSE' },
    { value: 'MAE', label: 'MAE' },
    { value: 'R2', label: 'R²' },
  ];

  // Generate realistic evaluation chart data
  const generateEvaluationData = (pollutant: PollutantType): EvaluationDataPoint[] => {
    const baseValues: Record<PollutantType, number> = {
      O3: 50,
      NO2: 70,
      HCHO: 12,
      CO: 500,
      PM25: 45,
      PM10: 80,
    };
    const baseValue = baseValues[pollutant];
    const data: EvaluationDataPoint[] = [];

    for (let i = 0; i < 24; i++) {
      const variance = Math.sin(i / 24 * Math.PI * 2) * (baseValue * 0.3);
      const predicted = baseValue + variance + (Math.random() - 0.5) * (baseValue * 0.15);
      const actual = baseValue + variance + (Math.random() - 0.5) * (baseValue * 0.2);

      data.push({
        time: `${String(i).padStart(2, '0')}:00`,
        predicted: Math.max(0, Math.round(predicted * 10) / 10),
        actual: Math.max(0, Math.round(actual * 10) / 10),
      });
    }

    return data;
  };

  // Generate running mean data for RMSE, MAE, and R²
  const generateRunningMeanData = (pollutant: PollutantType): RunningMeanDataPoint[] => {
    const startDate = new Date('2024-11-29');
    const data: RunningMeanDataPoint[] = [];

    const baseMetrics = metricsData[pollutant];

    for (let i = 0; i < 10; i++) {
      const currentDate = new Date(startDate);
      currentDate.setDate(currentDate.getDate() + i);

      const dateStr = `${currentDate.toLocaleString('en-US', { month: 'short' })} ${currentDate.getDate()}`;

      // Create a smooth curve for running means
      const progress = i / 9;
      const rmseValue = baseMetrics.rmse * 1.5 + Math.sin(progress * Math.PI * 2) * (baseMetrics.rmse * 0.5) + progress * (baseMetrics.rmse * 0.3);
      const maeValue = baseMetrics.mae * 1.3 + Math.sin(progress * Math.PI * 1.5) * (baseMetrics.mae * 0.4) + progress * (baseMetrics.mae * 0.3);
      const r2Value = baseMetrics.r_mean * 0.8 + Math.sin(progress * Math.PI * 1.2) * 0.15 + progress * 0.1;

      data.push({
        date: dateStr,
        rmse: Math.max(0, Math.round(rmseValue * 10) / 10),
        mae: Math.max(0, Math.round(maeValue * 10) / 10),
        r2: Math.max(0, Math.min(1, Math.round(r2Value * 100) / 100)),
      });
    }

    return data;
  };

  const evaluationData = generateEvaluationData(selectedPollutant);
  const runningMeanData = generateRunningMeanData(selectedPollutant);
  const metrics = metricsData[selectedPollutant];

  return (
    <div className="space-y-6">

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
                  {metrics.pollutant}
                </p>
              </div>
              <Badge className="bg-cyan-600/20 text-cyan-400 border-0">
                {metrics.unit}
              </Badge>
            </div>
            <div className="flex items-end gap-2">
              <span className={`text-5xl font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>{metrics.rmse}</span>
              <span className={`text-lg mb-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>{metrics.unit}</span>
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
                  {metrics.pollutant}
                </p>
              </div>
              <Badge className="bg-blue-600/20 text-blue-400 border-0">
                {metrics.unit}
              </Badge>
            </div>
            <div className="flex items-end gap-2">
              <span className={`text-5xl font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>{metrics.mae}</span>
              <span className={`text-lg mb-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>{metrics.unit}</span>
            </div>
            <p className={`text-xs ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
              Mean Absolute Error - average absolute deviation
            </p>
          </div>
        </Card>

        {/* R² (R-mean) Card */}
        <Card className={`border p-6 rounded-xl ${theme === 'dark' ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-white'}`}>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className={`text-sm font-semibold uppercase tracking-wide ${theme === 'dark' ? 'text-slate-300' : 'text-slate-700'}`}>
                  R²
                </h3>
                <p className={`text-xs mt-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
                  {metrics.pollutant}
                </p>
              </div>
              <Badge className="bg-emerald-600/20 text-emerald-400 border-0">
                Score
              </Badge>
            </div>
            <div className="flex items-end gap-2">
              <span className={`text-5xl font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>{metrics.r_mean}</span>
              <span className={`text-lg mb-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>out of 1.0</span>
            </div>
            <p className={`text-xs ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
              Coefficient of Determination - explained variance ratio
            </p>
          </div>
        </Card>
      </div>

      {/* Prediction vs Actual Chart */}
      <Card className={`border p-6 rounded-xl ${theme === 'dark' ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-white'}`}>
        <div className="flex items-center justify-between mb-6">
          <h3 className={`text-lg font-semibold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>
            {metrics.pollutant} - Predicted vs Actual Values
          </h3>
          {/* Pollutant Selector Dropdown - Right Upper Corner */}
          <div className="relative">
            <button
              onClick={() => {
                setIsPollutantDropdownOpen(!isPollutantDropdownOpen);
                setIsMetricDropdownOpen(false);
              }}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all border ${
                theme === 'dark'
                  ? 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
                  : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-50'
              }`}
            >
              <Filter className="w-4 h-4" />
              <span>{metricsData[selectedPollutant].pollutant}</span>
              <ChevronDown className={`w-4 h-4 transition-transform ${isPollutantDropdownOpen ? 'rotate-180' : ''}`} />
            </button>
            
            {isPollutantDropdownOpen && (
              <>
                <div
                  className="fixed inset-0 z-10"
                  onClick={() => setIsPollutantDropdownOpen(false)}
                />
                <div
                  className={`absolute top-full right-0 mt-2 z-20 min-w-[200px] rounded-lg border shadow-lg ${
                    theme === 'dark'
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
                          className={`w-full text-left px-4 py-2 text-sm transition-colors ${
                            isSelected
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
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={evaluationData}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke={theme === 'dark' ? '#334155' : '#e2e8f0'}
            />
            <XAxis dataKey="time" stroke={theme === 'dark' ? '#94a3b8' : '#64748b'} />
            <YAxis stroke={theme === 'dark' ? '#94a3b8' : '#64748b'} />
            <Tooltip
              contentStyle={{
                backgroundColor: theme === 'dark' ? '#1e293b' : '#ffffff',
                border: `1px solid ${theme === 'dark' ? '#475569' : '#e2e8f0'}`,
                borderRadius: '8px',
              }}
              labelStyle={{ color: theme === 'dark' ? '#f1f5f9' : '#1e293b' }}
              formatter={(value: number) => value.toFixed(2)}
            />
            <Legend wrapperStyle={{ paddingTop: '20px' }} />
            <Line
              type="monotone"
              dataKey="predicted"
              stroke={metrics.color}
              strokeWidth={3}
              dot={{ fill: metrics.color, r: 4 }}
              activeDot={{ r: 6 }}
              isAnimationActive={true}
              name={`Predicted ${metrics.pollutant}`}
            />
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#ef4444"
              strokeWidth={3}
              dot={{ fill: '#ef4444', r: 4 }}
              activeDot={{ r: 6 }}
              isAnimationActive={true}
              name={`Actual ${metrics.pollutant}`}
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Running Mean Chart - Filtered by Selected Metric */}
      <Card className={`border p-6 rounded-xl ${theme === 'dark' ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-white'}`}>
        <div className="flex items-center justify-between mb-6">
          <h3 className={`text-lg font-semibold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>
            Running Mean ({selectedMetric === 'R2' ? 'R²' : selectedMetric}) - {metrics.pollutant}
          </h3>
          {/* Dropdowns in Right Upper Corner */}
          <div className="flex items-center gap-2">
            {/* Pollutant Selector Dropdown */}
            <div className="relative">
              <button
                onClick={() => {
                  setIsPollutantDropdownOpen(!isPollutantDropdownOpen);
                  setIsMetricDropdownOpen(false);
                }}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all border ${
                  theme === 'dark'
                    ? 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
                    : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-50'
                }`}
              >
                <Filter className="w-4 h-4" />
                <span>{metricsData[selectedPollutant].pollutant}</span>
                <ChevronDown className={`w-4 h-4 transition-transform ${isPollutantDropdownOpen ? 'rotate-180' : ''}`} />
              </button>
              
              {isPollutantDropdownOpen && (
                <>
                  <div
                    className="fixed inset-0 z-10"
                    onClick={() => setIsPollutantDropdownOpen(false)}
                  />
                  <div
                    className={`absolute top-full right-0 mt-2 z-20 min-w-[200px] rounded-lg border shadow-lg ${
                      theme === 'dark'
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
                            className={`w-full text-left px-4 py-2 text-sm transition-colors ${
                              isSelected
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

            {/* Metric Selector Dropdown */}
            <div className="relative">
              <button
                onClick={() => {
                  setIsMetricDropdownOpen(!isMetricDropdownOpen);
                  setIsPollutantDropdownOpen(false);
                }}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all border ${
                  theme === 'dark'
                    ? 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
                    : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-50'
                }`}
              >
                <Filter className="w-4 h-4" />
                <span>{selectedMetric === 'R2' ? 'R²' : selectedMetric}</span>
                <ChevronDown className={`w-4 h-4 transition-transform ${isMetricDropdownOpen ? 'rotate-180' : ''}`} />
              </button>
              
              {isMetricDropdownOpen && (
                <>
                  <div
                    className="fixed inset-0 z-10"
                    onClick={() => setIsMetricDropdownOpen(false)}
                  />
                  <div
                    className={`absolute top-full right-0 mt-2 z-20 min-w-[150px] rounded-lg border shadow-lg ${
                      theme === 'dark'
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
                            className={`w-full text-left px-4 py-2 text-sm transition-colors ${
                              isSelected
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
          <LineChart data={runningMeanData}>
            <CartesianGrid strokeDasharray="3 3" stroke={theme === 'dark' ? '#334155' : '#e2e8f0'} />
            <XAxis dataKey="date" stroke={theme === 'dark' ? '#94a3b8' : '#64748b'} />
            <YAxis 
              stroke={theme === 'dark' ? '#94a3b8' : '#64748b'} 
              domain={selectedMetric === 'R2' ? [0, 1] : 'auto'}
              label={{ 
                value: selectedMetric === 'R2' ? 'Score' : `${metrics.unit}`, 
                angle: -90, 
                position: 'insideLeft' 
              }} 
            />
            <Tooltip
              contentStyle={{
                backgroundColor: theme === 'dark' ? '#1e293b' : '#ffffff',
                border: `1px solid ${theme === 'dark' ? '#475569' : '#e2e8f0'}`,
                borderRadius: '8px',
              }}
              labelStyle={{ color: theme === 'dark' ? '#f1f5f9' : '#1e293b' }}
              formatter={(value: number) => selectedMetric === 'R2' ? value.toFixed(3) : value.toFixed(2)}
            />
            <Line
              type="monotone"
              dataKey={selectedMetric === 'R2' ? 'r2' : selectedMetric.toLowerCase()}
              stroke={
                selectedMetric === 'RMSE' ? '#06b6d4' :
                selectedMetric === 'MAE' ? '#2563eb' :
                '#10b981'
              }
              strokeWidth={3}
              dot={{
                fill: selectedMetric === 'RMSE' ? '#06b6d4' :
                      selectedMetric === 'MAE' ? '#2563eb' :
                      '#10b981',
                r: 4
              }}
              activeDot={{ r: 6 }}
              isAnimationActive={true}
              name={
                selectedMetric === 'RMSE' ? `RMSE (${metrics.unit})` :
                selectedMetric === 'MAE' ? `MAE (${metrics.unit})` :
                'R² Score'
              }
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Model Performance Info */}
      <Card className={`border p-6 rounded-xl ${theme === 'dark' ? 'border-slate-700 bg-slate-800' : 'border-slate-200 bg-white'}`}>
        <h3 className={`text-lg font-semibold mb-4 ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>Model Performance Analysis</h3>
        <div className={`grid grid-cols-1 md:grid-cols-2 gap-6 text-sm ${theme === 'dark' ? 'text-slate-300' : 'text-slate-700'}`}>
          <div className="space-y-3">
            <div>
              <p className={`font-medium mb-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>RMSE (Root Mean Square Error)</p>
              <p className={theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}>
                The RMSE of {metrics.rmse} {metrics.unit} indicates the average magnitude of prediction
                errors. Lower values indicate better model performance.
              </p>
            </div>
            <div>
              <p className={`font-medium mb-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>MAE (Mean Absolute Error)</p>
              <p className={theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}>
                The MAE of {metrics.mae} {metrics.unit} represents the average absolute difference
                between predicted and actual values.
              </p>
            </div>
          </div>
          <div className="space-y-3">
            <div>
              <p className={`font-medium mb-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>R² (Coefficient of Determination)</p>
              <p className={theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}>
                An R² value of {metrics.r_mean} (on a scale of 0 to 1) indicates that the model
                explains {(metrics.r_mean * 100).toFixed(0)}% of the variance in the data.
              </p>
            </div>
            <div>
              <p className={`font-medium mb-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>Model Status</p>
              <p className="text-emerald-400">
                ✓ Model performance is excellent with high R² value and low error metrics
              </p>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default ModelEvaluation;
