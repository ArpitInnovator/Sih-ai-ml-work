declare module 'leaflet.heat' {
  import * as L from 'leaflet';
  namespace heatLayer {
    function heatLayer(
      latlngs: Array<L.LatLngExpression | [number, number, number]>,
      options?: {
        minOpacity?: number;
        maxZoom?: number;
        radius?: number;
        blur?: number;
        max?: number;
        gradient?: Record<string, string>;
      }
    ): L.Layer;
  }
  export = heatLayer;
}

declare module 'leaflet' {
  function heatLayer(
    latlngs: Array<L.LatLngExpression | [number, number, number]>,
    options?: {
      minOpacity?: number;
      maxZoom?: number;
      radius?: number;
      blur?: number;
      max?: number;
      gradient?: Record<string, string>;
    }
  ): L.Layer;
}


