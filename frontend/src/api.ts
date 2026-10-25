const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export type Exif = {
  camera: string | null;
  software: string | null;
  taken_at: string | null;
  has_gps: boolean;
  editor_detected: string | null;
};

export type AnalyzeResult = {
  label: "tampered" | "authentic";
  probability: number;
  threshold: number;
  reliable: boolean;
  warnings: string[];
  ela_png_base64: string;
  exif: Exif;
  width: number;
  height: number;
  format: string;
  inference_ms: number;
};

export type Health = {
  status: string;
  model_loaded: boolean;
  model: { model: string; size: number; input: string; quality: number; threshold: number } | null;
};

export class ApiError extends Error {}

export async function fetchHealth(): Promise<Health | null> {
  try {
    const res = await fetch(`${BASE}/health`);
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

export async function analyze(file: File): Promise<AnalyzeResult> {
  const body = new FormData();
  body.append("file", file);

  let res: Response;
  try {
    res = await fetch(`${BASE}/analyze`, { method: "POST", body });
  } catch {
    throw new ApiError(`Can't reach the analysis service at ${BASE}. Start the API and try again.`);
  }

  if (!res.ok) {
    const payload = await res.json().catch(() => null);
    const detail = payload?.detail;
    throw new ApiError(typeof detail === "string" ? detail : "The service couldn't analyze this file.");
  }
  return res.json();
}
