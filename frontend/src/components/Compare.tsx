import { useEffect, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent } from "react";

type Props = {
  original: string;
  ela: string;
  width: number;
  height: number;
  name: string;
};

const STEP = 4;

export function Compare({ original, ela, width, height, name }: Props) {
  const [split, setSplit] = useState(100);
  const [animate, setAnimate] = useState(true);
  const box = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  useEffect(() => {
    const timer = setTimeout(() => setSplit(50), 80);
    return () => clearTimeout(timer);
  }, []);

  const clamp = (value: number) => Math.min(100, Math.max(0, value));

  const moveTo = (clientX: number) => {
    const rect = box.current?.getBoundingClientRect();
    if (rect) setSplit(clamp(((clientX - rect.left) / rect.width) * 100));
  };

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    dragging.current = true;
    setAnimate(false);
    e.currentTarget.setPointerCapture(e.pointerId);
    moveTo(e.clientX);
  };

  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (dragging.current) moveTo(e.clientX);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const targets: Partial<Record<string, number>> = {
      ArrowLeft: split - STEP,
      ArrowRight: split + STEP,
      Home: 0,
      End: 100,
    };
    const next = targets[e.key];
    if (next === undefined) return;
    e.preventDefault();
    setAnimate(false);
    setSplit(clamp(next));
  };

  const style = { "--split": `${split}%`, aspectRatio: `${width} / ${height}` } as CSSProperties;

  return (
    <div
      ref={box}
      className="compare"
      data-animate={animate}
      style={style}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={() => (dragging.current = false)}
      onPointerCancel={() => (dragging.current = false)}
    >
      <img className="compare-layer" src={ela} alt="Error level map of the photo" draggable={false} />
      <img className="compare-layer compare-original" src={original} alt={`Uploaded photo ${name}`} draggable={false} />
      <span className="compare-tag compare-tag-left">Your photo</span>
      <span className="compare-tag compare-tag-right">Error level map</span>
      <div
        className="seam"
        role="slider"
        tabIndex={0}
        aria-label="Compare the photo with its error level map"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(split)}
        aria-valuetext={`${Math.round(split)} percent photo`}
        onKeyDown={onKeyDown}
      >
        <svg className="seam-grip" viewBox="0 0 36 36" aria-hidden="true">
          <path d="M14 11 7 18l7 7M22 11l7 7-7 7" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
    </div>
  );
}
