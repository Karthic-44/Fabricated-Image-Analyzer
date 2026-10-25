type Props = {
  preview: string;
  name: string;
};

export function Pending({ preview, name }: Props) {
  return (
    <div className="stage pending" role="status">
      <img className="pending-image" src={preview} alt={`Uploaded photo ${name}`} />
      <div className="pending-bar">
        <span>Analyzing the compression pattern</span>
        <span className="pending-track" aria-hidden="true" />
      </div>
    </div>
  );
}
