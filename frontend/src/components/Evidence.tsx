import type { AnalyzeResult } from "../api";

type Props = {
  result: AnalyzeResult;
  name: string;
  onReset: () => void;
};

type Tone = "edited" | "clean" | "unsure";

function Meter({ value, threshold, tone }: { value: number; threshold: number; tone: Tone }) {
  return (
    <div>
      <div
        className="meter"
        role="img"
        aria-label={`Score ${value.toFixed(2)} on a scale from 0 to 1. Flag threshold ${threshold.toFixed(2)}.`}
      >
        <div className="meter-fill" data-tone={tone} style={{ width: `${value * 100}%` }} />
        <div className="meter-tick" style={{ left: `${threshold * 100}%` }} />
      </div>
      <div className="meter-scale" aria-hidden="true">
        <span>0</span>
        <span>1</span>
      </div>
    </div>
  );
}

function takenAt(value: string) {
  return value.replace(/^(\d{4}):(\d{2}):(\d{2})/, "$1-$2-$3");
}

export function Evidence({ result, name, onReset }: Props) {
  const flagged = result.label === "tampered";
  const tone: Tone = !result.reliable ? "unsure" : flagged ? "edited" : "clean";
  const heading = !result.reliable
    ? "This result may be unreliable"
    : flagged
      ? "Signs of editing found"
      : "No signs of editing found";
  const { exif } = result;

  return (
    <section className="evidence" aria-live="polite">
      <div className="evidence-block">
        <h2 className="verdict" data-tone={tone}>
          {heading}
        </h2>
        <p>
          The model scored this photo {result.probability.toFixed(2)}. Scores of {result.threshold.toFixed(2)} or higher
          are flagged.
        </p>
        <Meter value={result.probability} threshold={result.threshold} tone={tone} />
      </div>

      {result.warnings.length > 0 && (
        <ul className="notices">
          {result.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}

      <p className="note">
        Brighter areas in the map recompress differently from their surroundings. That can point to edits, but it isn't
        proof either way.
      </p>

      <dl className="details">
        <dt>File</dt>
        <dd>{name}</dd>
        <dt>Format</dt>
        <dd>{result.format}</dd>
        <dt>Size</dt>
        <dd>
          {result.width} × {result.height} px
        </dd>
        {exif.camera && (
          <>
            <dt>Camera</dt>
            <dd>{exif.camera}</dd>
          </>
        )}
        {exif.software && (
          <>
            <dt>Software</dt>
            <dd>
              {exif.software}
              {exif.editor_detected && <span className="flag">Editing software</span>}
            </dd>
          </>
        )}
        {exif.taken_at && (
          <>
            <dt>Taken</dt>
            <dd>{takenAt(exif.taken_at)}</dd>
          </>
        )}
        <dt>Location data</dt>
        <dd>{exif.has_gps ? "Present in the file" : "None found"}</dd>
        <dt>Analysis time</dt>
        <dd>{result.inference_ms} ms</dd>
      </dl>

      <div>
        <button className="button button-outline" type="button" onClick={onReset}>
          Check another photo
        </button>
      </div>
    </section>
  );
}
