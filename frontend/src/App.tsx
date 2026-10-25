import { useEffect, useRef, useState } from "react";
import { analyze, ApiError, fetchHealth, type AnalyzeResult, type Health } from "./api";
import { Compare } from "./components/Compare";
import { DropStage } from "./components/DropStage";
import { Evidence } from "./components/Evidence";
import { Pending } from "./components/Pending";

const MAX_BYTES = 10 * 1024 * 1024;
const ACCEPTED = ["image/jpeg", "image/png", "image/webp"];

type State =
  | { phase: "idle" }
  | { phase: "analyzing"; name: string; preview: string }
  | { phase: "done"; name: string; preview: string; result: AnalyzeResult };

type Service = "checking" | "offline" | Health;

function serviceLabel(service: Service) {
  if (service === "checking") return { tone: "checking", text: "Checking the analysis service" };
  if (service === "offline") return { tone: "offline", text: "Analysis service offline" };
  if (!service.model_loaded || !service.model) return { tone: "offline", text: "No model loaded" };
  return { tone: "ready", text: `Model ready (${service.model.model})` };
}

export default function App() {
  const [state, setState] = useState<State>({ phase: "idle" });
  const [problem, setProblem] = useState<string | null>(null);
  const [service, setService] = useState<Service>("checking");
  const previewUrl = useRef<string | null>(null);

  const refreshService = () => {
    fetchHealth().then((health) => setService(health ?? "offline"));
  };

  const release = () => {
    if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
    previewUrl.current = null;
  };

  useEffect(() => {
    refreshService();
    return release;
  }, []);

  const handleFile = async (file: File) => {
    setProblem(null);
    if (!ACCEPTED.includes(file.type)) {
      setProblem("Only JPEG, PNG and WebP photos can be analyzed.");
      return;
    }
    if (file.size > MAX_BYTES) {
      setProblem("That file is larger than 10 MB. Choose a smaller photo.");
      return;
    }

    release();
    const preview = URL.createObjectURL(file);
    previewUrl.current = preview;
    setState({ phase: "analyzing", name: file.name, preview });

    try {
      const result = await analyze(file);
      setState({ phase: "done", name: file.name, preview, result });
    } catch (err) {
      release();
      setState({ phase: "idle" });
      setProblem(err instanceof ApiError ? err.message : "Something went wrong while analyzing the photo.");
      refreshService();
    }
  };

  const reset = () => {
    release();
    setProblem(null);
    setState({ phase: "idle" });
  };

  const status = serviceLabel(service);

  return (
    <div className="page">
      <header className="masthead">
        <p className="brand">Fabricated Image Analyzer</p>
        <p className="service" data-tone={status.tone}>
          <span className="service-dot" aria-hidden="true" />
          {status.text}
        </p>
      </header>

      <div className="intro" data-compact={state.phase !== "idle"}>
        <h1>Check whether a photo has been edited</h1>
        <p>
          Upload an image and the model looks for JPEG compression patterns that don't match across the picture, which
          often happens where something was pasted in or retouched.
        </p>
      </div>

      {problem && (
        <div className="problem" role="alert">
          <h2>Couldn't analyze this photo</h2>
          <p>{problem}</p>
        </div>
      )}

      <main className="workspace" data-phase={state.phase}>
        {state.phase === "idle" && <DropStage onFile={handleFile} />}
        {state.phase === "analyzing" && <Pending preview={state.preview} name={state.name} />}
        {state.phase === "done" && (
          <>
            <Compare
              original={state.preview}
              ela={`data:image/png;base64,${state.result.ela_png_base64}`}
              width={state.result.width}
              height={state.result.height}
              name={state.name}
            />
            <Evidence result={state.result} name={state.name} onReset={reset} />
          </>
        )}
      </main>

      <footer className="colophon">Images are analyzed in memory and are not saved.</footer>
    </div>
  );
}
