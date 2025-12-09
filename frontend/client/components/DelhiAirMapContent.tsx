'use client';

import React, { useEffect, useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L, { LatLngBoundsExpression, LatLngTuple } from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet.heat';

import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

// Fix marker icons for Vite builds
if (typeof window !== 'undefined') {
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
}

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

// Component to fit map bounds to show all markers
const FitBounds: React.FC<{ sites: SiteReading[] }> = ({ sites }) => {
  const map = useMap();

  useEffect(() => {
    if (!map || sites.length === 0) return;
    
    // Calculate bounds from all site locations
    const bounds = L.latLngBounds(
      sites.map(site => [site.lat, site.lon] as LatLngTuple)
    );
    
    // Fit bounds with padding to ensure all markers are visible
    map.fitBounds(bounds, {
      padding: [50, 50], // Add padding so markers aren't at the edge
      maxZoom: 12, // Limit max zoom to keep overview
    });
  }, [map, sites]);

  return null;
};

type DelhiAirMapContentProps = {
  data: SiteReading[];
  pollutant: Pollutant;
  dataset: Dataset;
  center: LatLngTuple;
  bounds: LatLngBoundsExpression;
  tileUrl: string;
  gradient: Record<string, string>;
  range: { min: number; max: number; color: string; icon: React.ReactNode };
};

export default function DelhiAirMapContent({
  data,
  pollutant,
  dataset,
  center,
  bounds,
  tileUrl,
  gradient,
  range,
}: DelhiAirMapContentProps) {
  const heatPoints = useMemo(() => {
    return data.map((site) => {
      const key = `${pollutant}_${dataset}` as keyof SiteReading;
      const raw = site[key] as number;
      const clamp = (v: number) => Math.max(range.min, Math.min(range.max, v));
      const normalized = (clamp(raw) - range.min) / (range.max - range.min || 1);
      return [site.lat, site.lon, normalized] as [number, number, number];
    });
  }, [data, pollutant, dataset, range]);

  return (
    <div className="relative h-[420px] w-full">
      <MapContainer
        center={center}
        zoom={11}
        minZoom={9}
        maxZoom={16}
        maxBounds={bounds}
        maxBoundsViscosity={1.0}
        scrollWheelZoom
        className="h-full w-full"
      >
        <TileLayer attribution='&copy; OpenStreetMap' url={tileUrl} />
        <FitBounds sites={data} />
        <HeatLayer points={heatPoints} gradient={gradient} />

        {data.map((site) => {
          const key = `${pollutant}_${dataset}` as keyof SiteReading;
          const value = site[key] as number;
          return (
            <Marker key={`${pollutant}-${site.id}`} position={[site.lat, site.lon]}>
              <Popup>
                <div className="space-y-1">
                  <div className="flex items-center gap-2 font-semibold text-slate-800">
                    <span>{site.name}</span>
                  </div>
                  <div className="text-sm text-slate-600">
                    <div className="flex items-center gap-2">
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
        className={`absolute right-3 bottom-3 rounded-lg border px-3 py-2 shadow-md bg-white/90 border-slate-200 text-slate-700`}
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
  );
}






