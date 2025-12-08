import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { useTheme } from '@/contexts/ThemeContext';

interface PollutantGaugeProps {
  name: string;
  value: number;
  maxValue: number;
  unit: string;
  color: string;
}

const getStatusColor = (percentage: number) => {
  if (percentage <= 25) return '#10b981'; // Green
  if (percentage <= 50) return '#f59e0b'; // Amber
  if (percentage <= 75) return '#f97316'; // Orange
  return '#ef4444'; // Red
};

const getStatusLabel = (percentage: number) => {
  if (percentage <= 25) return 'Good';
  if (percentage <= 50) return 'Fair';
  if (percentage <= 75) return 'Poor';
  return 'Very Poor';
};

export default function PollutantGauge({
  name,
  value,
  maxValue,
  unit,
  color,
}: PollutantGaugeProps) {
  const { theme } = useTheme();
  const percentage = (value / maxValue) * 100;
  const statusColor = getStatusColor(percentage);
  const statusLabel = getStatusLabel(percentage);
  const remainingColor = theme === 'dark' ? '#1e293b' : '#e2e8f0';

  const data = [
    { name: 'Used', value: Math.min(percentage, 100) },
    { name: 'Remaining', value: Math.max(100 - percentage, 0) },
  ];

  return (
    <div className={`rounded-xl p-4 flex flex-col items-center border ${theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200 shadow-sm'}`}>
      <h3 className={`text-sm font-semibold mb-4 ${theme === 'dark' ? 'text-slate-100' : 'text-slate-900'}`}>{name}</h3>

      <div className="relative w-20 h-20 md:w-24 md:h-24">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              startAngle={180}
              endAngle={0}
              innerRadius={32}
              outerRadius={40}
              dataKey="value"
            >
              <Cell fill={statusColor} />
              <Cell fill={remainingColor} />
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-lg font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>{Math.round(value)}</span>
          <span className={`text-xs ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>{unit}</span>
        </div>
      </div>

      <div className="mt-4 text-center">
        <p className="text-xl font-bold" style={{ color: statusColor }}>
          {percentage.toFixed(0)}%
        </p>
        <p className={`text-xs mt-1 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>{statusLabel}</p>
      </div>
    </div>
  );
}
