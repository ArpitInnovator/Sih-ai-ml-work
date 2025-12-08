'use client';

import React, { useEffect, useMemo, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L, { LatLngBoundsExpression, LatLngTuple } from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet.heat';
import { useTheme } from '@/contexts/ThemeContext';
import { RefreshCw, MapPin, Flame, Droplets } from 'lucide-react';

import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

type Pollutant = 'O3' | 'NO2';
type Dataset = 'actual' | 'predicted';

type SiteReading = {
  id: number;
  name: string;
  lat: number;
  lon: number;
  O3_actual: number;
  O3_predicted: number;
  NO2_actual: number;
  NO2_predicted: number;
  updatedAt: string;
};

type DelhiAirMapProps = {
  sites?: SiteReading[];
  onRefresh?: () => void;
};

const defaultSites: SiteReading[] = [
  { id: 1, name: 'Site 1', lat: 28.69536, lon: 77.18168, O3_actual: 48, O3_predicted: 52, NO2_actual: 60, NO2_predicted: 58, updatedAt: new Date().toISOString() },
  { id: 2, name: 'Site 2', lat: 28.5718, lon: 77.07125, O3_actual: 42, O3_predicted: 46, NO2_actual: 55, NO2_predicted: 53, updatedAt: new Date().toISOString() },
  { id: 3, name: 'Site 3', lat: 28.58278, lon: 77.23441, O3_actual: 50, O3_predicted: 54, NO2_actual: 63, NO2_predicted: 61, updatedAt: new Date().toISOString() },
  { id: 4, name: 'Site 4', lat: 28.82286, lon: 77.10197, O3_actual: 44, O3_predicted: 48, NO2_actual: 52, NO2_predicted: 50, updatedAt: new Date().toISOString() },
  { id: 5, name: 'Site 5', lat: 28.53077, lon: 77.27123, O3_actual: 46, O3_predicted: 51, NO2_actual: 58, NO2_predicted: 57, updatedAt: new Date().toISOString() },
  { id: 6, name: 'Site 6', lat: 28.72954, lon: 77.09601, O3_actual: 49, O3_predicted: 53, NO2_actual: 57, NO2_predicted: 55, updatedAt: new Date().toISOString() },
  { id: 7, name: 'Site 7', lat: 28.71052, lon: 77.24951, O3_actual: 47, O3_predicted: 50, NO2_actual: 59, NO2_predicted: 58, updatedAt: new Date().toISOString() },
];

const pollutantRanges: Record<Pollutant, { min: number; max: number; color: string; icon: React.ReactNode }> = {
  O3: { min: 0, max: 150, color: '#06b6d4', icon: <Droplets className="w-4 h-4" /> },
  NO2: { min: 0, max: 200, color: '#f97316', icon: <Flame className="w-4 h-4" /> },
};

// Fix marker icons for Vite builds
const DefaultIcon = L.icon({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});
L.Marker.prototype.options.icon = DefaultIcon;

const HeatLayer: React.FC<{
  points: Array<[number, number, number]>;
  radius?: number;
  blur?: number;
  gradient?: Record<string, string>;
}> = ({ points, radius = 28, blur = 18, gradient }) => {
  const map = useMap();

  useEffect(() => {
    if (!map) return;
    const layer = (L as any).heatLayer(points, {
      radius,
      blur,
      maxZoom: 17,
      minOpacity: 0.15,
      gradient,
    });
    layer.addTo(map);
    return () => {
      map.removeLayer(layer);
    };
  }, [map, points, radius, blur, gradient]);

  return null;
};

export default function DelhiAirMap({ sites, onRefresh }: DelhiAirMapProps) {
  const { theme } = useTheme();
  const [dataset, setDataset] = useState<Dataset>('actual');

  const data = sites ?? defaultSites;
  const center: LatLngTuple = [28.61, 77.21];
  const bounds: LatLngBoundsExpression = [
    [28.35, 76.8],
    [28.95, 77.45],
  ];

  const gradient = useMemo(
    () => ({
      0.0: '#0ea5e9',
      0.25: '#22d3ee',
      0.5: '#a3e635',
      0.75: '#f59e0b',
      1.0: '#ef4444',
    }),
    []
  );

  const tileUrl =
    theme === 'dark'
      ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
      : 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png';

  return (
    <div className={`rounded-xl border ${theme === 'dark' ? 'bg-slate-900/60 border-slate-800' : 'bg-white border-slate-200'} overflow-hidden`}>
      <div className="flex flex-col gap-4 p-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h3 className={`text-lg font-semibold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>Delhi O₃ & NO₂ Heatmaps</h3>
          <p className={`text-sm ${theme === 'dark' ? 'text-slate-300' : 'text-slate-600'}`}>Side-by-side intensity for Delhi sites only</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div className={`flex rounded-lg p-1 ${theme === 'dark' ? 'bg-slate-800' : 'bg-slate-100'}`}>
            {(['actual', 'predicted'] as Dataset[]).map((d) => (
              <button
                key={d}
                onClick={() => setDataset(d)}
                className={`px-3 py-1.5 text-sm rounded-md font-medium transition ${
                  dataset === d
                    ? 'bg-emerald-600 text-white shadow-sm'
                    : theme === 'dark'
                    ? 'text-slate-200 hover:bg-slate-700'
                    : 'text-slate-700 hover:bg-white'
                }`}
              >
                {d === 'actual' ? 'Actual' : 'Predicted'}
              </button>
            ))}
          </div>
          {onRefresh && (
            <button
              onClick={onRefresh}
              className={`inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition ${
                theme === 'dark'
                  ? 'bg-slate-800 text-slate-200 hover:bg-slate-700 border border-slate-700'
                  : 'bg-white text-slate-700 hover:bg-slate-100 border border-slate-200'
              }`}
            >
              <RefreshCw className="w-4 h-4" /> Refresh
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 p-4 lg:grid-cols-2">
        {(['NO2', 'O3'] as Pollutant[]).map((pollutant) => {
          const range = pollutantRanges[pollutant];
          const heatPoints = data.map((site) => {
            const key = `${pollutant}_${dataset}` as keyof SiteReading;
            const raw = site[key] as number;
            const clamp = (v: number) => Math.max(range.min, Math.min(range.max, v));
            const normalized = (clamp(raw) - range.min) / (range.max - range.min || 1);
            return [site.lat, site.lon, normalized] as [number, number, number];
          });

          return (
            <div key={pollutant} className="flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {range.icon}
                  <div>
                    <p className={`text-sm font-semibold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>{pollutant === 'O3' ? 'O₃' : 'NO₂'} Heatmap</p>
                    <p className={`text-xs ${theme === 'dark' ? 'text-slate-300' : 'text-slate-600'}`}>Delhi focus • {dataset === 'actual' ? 'Actual readings' : 'Model predictions'}</p>
                  </div>
                </div>
              </div>

              <div className="relative h-[420px] w-full">
                <MapContainer
                  center={center}
                  zoom={11}
                  minZoom={10}
                  maxZoom={16}
                  maxBounds={bounds}
                  maxBoundsViscosity={1.0}
                  scrollWheelZoom
                  className="h-full w-full"
                >
                  <TileLayer attribution='&copy; OpenStreetMap' url={tileUrl} />
                  <HeatLayer points={heatPoints} gradient={gradient} />

                  {data.map((site) => {
                    const key = `${pollutant}_${dataset}` as keyof SiteReading;
                    const value = site[key] as number;
                    return (
                      <Marker key={`${pollutant}-${site.id}`} position={[site.lat, site.lon]}>
                        <Popup>
                          <div className="space-y-1">
                            <div className="flex items-center gap-2 font-semibold text-slate-800">
                              <MapPin className="w-4 h-4 text-cyan-600" />
                              <span>{site.name}</span>
                            </div>
                            <div className="text-sm text-slate-600">
                              <div className="flex items-center gap-2">
                                {range.icon}
                                <span>
                                  {pollutant === 'O3' ? 'O₃' : 'NO₂'} ({dataset}):{' '}
                                  <span className="font-semibold text-slate-900">{value.toFixed(1)} ppb</span>
                                </span>
                              </div>
                              <div className="text-xs text-slate-500 mt-1">Updated: {new Date(site.updatedAt).toLocaleString()}</div>
                            </div>
                          </div>
                        </Popup>
                      </Marker>
                    );
                  })}
                </MapContainer>

                <div
                  className={`absolute right-3 bottom-3 rounded-lg border px-3 py-2 shadow-md ${
                    theme === 'dark' ? 'bg-slate-900/80 border-slate-700 text-slate-100' : 'bg-white/90 border-slate-200 text-slate-700'
                  }`}
                >
                  <div className="text-xs font-semibold mb-1 flex items-center gap-1" style={{ color: range.color }}>
                    {range.icon}
                    <span>{pollutant === 'O3' ? 'O₃' : 'NO₂'} ({dataset})</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-2 w-32 rounded-full" style={{ background: 'linear-gradient(90deg, #0ea5e9, #22d3ee, #a3e635, #f59e0b, #ef4444)' }} />
                    <div className="text-[10px] text-slate-500">{range.min}–{range.max} ppb</div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}


