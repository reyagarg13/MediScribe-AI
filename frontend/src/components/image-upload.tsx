import { useRef, useState } from "react";

export function ImageUpload({ onResult }: { onResult: (result: any) => void }) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImageUrl(URL.createObjectURL(file));
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("http://localhost:8000/analyze-image", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    setResult(data);
    setLoading(false);
    onResult(data);
  };

  return (
    <div className="space-y-4">
      <input
        type="file"
        accept="image/*"
        ref={fileInput}
        onChange={handleFileChange}
        className="block"
      />
      {imageUrl && <img src={imageUrl} alt="Uploaded" className="max-w-xs rounded" />}
      {loading && <p>Analyzing image...</p>}
      {result && (
        <div className="bg-zinc-800 p-4 rounded">
          <h4 className="font-semibold">Analysis Result</h4>
          <pre className="text-xs text-zinc-200">{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
