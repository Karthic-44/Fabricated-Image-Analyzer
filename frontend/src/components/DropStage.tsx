import { useRef, useState } from "react";

type Props = {
  onFile: (file: File) => void;
};

export function DropStage({ onFile }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const pick = (files: FileList | null) => {
    const file = files?.[0];
    if (file) onFile(file);
  };

  return (
    <div
      className="stage drop"
      data-over={over}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        pick(e.dataTransfer.files);
      }}
    >
      <span className="drop-half">Your photo</span>
      <span className="drop-half">Error level map</span>
      <div className="drop-prompt">
        <h2>Drop a photo here</h2>
        <button className="button button-amber" type="button" onClick={() => input.current?.click()}>
          Choose photo
        </button>
        <p>JPEG works best. PNG and WebP are accepted but less reliable. Up to 10 MB.</p>
      </div>
      <input
        ref={input}
        className="visually-hidden"
        type="file"
        accept="image/jpeg,image/png,image/webp"
        tabIndex={-1}
        aria-hidden="true"
        onChange={(e) => {
          pick(e.target.files);
          e.target.value = "";
        }}
      />
    </div>
  );
}
