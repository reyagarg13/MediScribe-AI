import { useRef, useState } from "react";

export function ImageUpload({ onResult }: { onResult: (result: any) => void }) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImageUrl(URL.createObjectURL(file));
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`${API_BASE}/analyze-image`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Server error: ${res.status} ${text}`);
      }
      const data = await res.json();
      setResult(data);
      onResult(data);
    } catch (err: any) {
      console.error("Image analysis failed:", err);
      setError(err?.message || String(err));
    } finally {
      setLoading(false);
    }
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
  {error && <p className="text-red-400">Error: {error}</p>}
      {result && (
        <div className="bg-zinc-800 p-4 rounded">
          <h4 className="font-semibold">Analysis Result</h4>
          <pre className="text-xs text-zinc-200">{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
