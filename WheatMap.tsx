import React, { useEffect, useRef, useState } from "react";
import { MapContainer, TileLayer, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";

interface Props {
  layerUrl: string;
  activeIndex: "NDVI" | "NDRE" | "HEALTH" | "RISK";
}

/**
 * RasterLayer — dynamically loads a GeoTIFF via georaster-layer-for-leaflet.
 * Handles the case where the TIF is missing (404 from /api/raster/) gracefully.
 */
const RasterLayer: React.FC<Props> = ({ layerUrl, activeIndex }) => {
  const map = useMap();
  const layerRef = useRef<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!layerUrl) return;

    let cancelled = false;

    const loadRaster = async () => {
      try {
        // Dynamic imports avoid bundling georaster in environments where it's absent
        const parseGeoRaster      = (await import("georaster")).default;
        const GeoRasterLayer      = (await import("georaster-layer-for-leaflet")).default;

        const response = await fetch(layerUrl);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status} — TIF not found at ${layerUrl}`);
        }
        const arrayBuffer = await response.arrayBuffer();
        const georaster   = await parseGeoRaster(arrayBuffer);

        if (cancelled) return;

        // Remove previous layer
        if (layerRef.current) {
          map.removeLayer(layerRef.current);
        }

        const colourFn = (values: number[]) => {
          const v = values[0];
          if (isNaN(v) || v === 0) return null;

          switch (activeIndex) {
            case "NDVI":
            case "NDRE":
              if (v > 0.55) return "rgba(0,100,0,0.75)";
              if (v > 0.45) return "rgba(34,197,94,0.75)";
              if (v > 0.40) return "rgba(134,239,172,0.75)";
              if (v > 0.38) return "rgba(253,230,138,0.75)";
              return "rgba(248,113,113,0.75)";

            case "HEALTH":
              if (v === 1) return "rgba(34,197,94,0.8)";
              if (v === 2) return "rgba(234,179,8,0.8)";
              if (v === 3) return "rgba(239,68,68,0.8)";
              return null;

            case "RISK":
              if (v === 1) return "rgba(34,197,94,0.8)";
              if (v === 2) return "rgba(234,179,8,0.8)";
              if (v === 3) return "rgba(239,68,68,0.8)";
              return null;

            default:
              return "rgba(56,189,248,0.75)";
          }
        };

        const layer = new GeoRasterLayer({
          georaster,
          opacity:              0.75,
          pixelValuesToColorFn: colourFn,
          resolution:           256,
        });

        layer.addTo(map);
        layerRef.current = layer;

        try {
          map.fitBounds(layer.getBounds());
        } catch {
          // fitBounds can fail on very large rasters — silently continue
        }

        setError(null);
      } catch (err: any) {
        if (!cancelled) {
          console.error("[WheatMap] Raster load failed:", err.message);
          setError(err.message);
        }
      }
    };

    loadRaster();
    return () => { cancelled = true; };
  }, [layerUrl, activeIndex, map]);

  if (error) {
    console.warn("[WheatMap]", error);
  }
  return null;
};


/**
 * WheatMap — Leaflet container with optional GeoTIFF overlay.
 * Shows base OpenStreetMap tiles even when the TIF endpoint is unavailable.
 */
const WheatMap: React.FC<Props> = ({ layerUrl, activeIndex }) => {
  return (
    <MapContainer
      center={[30.901, 75.857]}
      zoom={11}
      style={{ height: "100%", width: "100%", borderRadius: "1rem", zIndex: 0 }}
      scrollWheelZoom={true}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        maxZoom={18}
      />
      {layerUrl && (
        <RasterLayer layerUrl={layerUrl} activeIndex={activeIndex} />
      )}
    </MapContainer>
  );
};

export default WheatMap;
