import { useRef, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Upload, FileImage, Pill, Eye, CheckCircle, XCircle, AlertTriangle } from "lucide-react";

export function ImageUpload({ onResult }: { onResult: (result: any) => void }) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

  // New: allow switching between endpoints (image analysis vs prescription OCR)
  const [ocrMode, setOcrMode] = useState<"image"|"prescription">("prescription");

  const handleFileChange = async (file: File) => {
    if (!file) return;
    
    setImageUrl(URL.createObjectURL(file));
    setLoading(true);
    setError(null);
    setUploadProgress(0);
    
    const formData = new FormData();
    formData.append("file", file);
    
    try {
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      const endpoint = ocrMode === "prescription" ? "/prescription-ocr" : "/analyze-image";
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        body: formData,
      });
      
      clearInterval(progressInterval);
      setUploadProgress(100);
      
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
      setTimeout(() => setUploadProgress(0), 1000);
    }
  };

  const handleInputChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      await handleFileChange(file);
    }
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await handleFileChange(e.dataTransfer.files[0]);
    }
  }, []);

  const renderPrescriptionResult = (data: any) => {
    if (!data.medications || data.medications.length === 0) {
      return (
        <div className="flex items-center gap-2 text-zinc-400">
          <AlertTriangle className="w-4 h-4" />
          <span>No medications found in the image</span>
        </div>
      );
    }

    return (
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-green-400">
          <CheckCircle className="w-4 h-4" />
          <span className="font-medium">Found {data.medications.length} medication(s)</span>
        </div>
        {data.medications.map((med: any, idx: number) => (
          <div key={idx} className="bg-zinc-900 p-3 rounded-lg border border-zinc-700">
            <div className="flex items-start justify-between mb-2">
              <div className="flex items-center gap-2">
                <Pill className="w-4 h-4 text-blue-400" />
                <span className="font-medium text-white">
                  {med.matched_name || med.name_candidate || "Unknown medication"}
                </span>
              </div>
              {med.match_score && (
                <span className={`text-xs px-2 py-1 rounded ${
                  med.match_score >= 80 ? "bg-green-900 text-green-300" :
                  med.match_score >= 60 ? "bg-yellow-900 text-yellow-300" :
                  "bg-red-900 text-red-300"
                }`}>
                  {med.match_score}% match
                </span>
              )}
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-sm">
              {med.dosage && (
                <div>
                  <span className="text-zinc-400">Dosage:</span>
                  <span className="ml-1 text-zinc-200">{med.dosage}</span>
                </div>
              )}
              {med.frequency && (
                <div>
                  <span className="text-zinc-400">Frequency:</span>
                  <span className="ml-1 text-zinc-200">{med.frequency}</span>
                </div>
              )}
              {med.duration && (
                <div>
                  <span className="text-zinc-400">Duration:</span>
                  <span className="ml-1 text-zinc-200">{med.duration}</span>
                </div>
              )}
            </div>
            {med.notes && (
              <div className="mt-2 text-xs text-amber-400 bg-amber-900/20 p-2 rounded">
                {med.notes}
              </div>
            )}
          </div>
        ))}
        {data.ocr_text && (
          <details className="mt-4">
            <summary className="cursor-pointer text-sm text-zinc-400 hover:text-zinc-200">
              View raw OCR text
            </summary>
            <div className="mt-2 p-3 bg-zinc-900 rounded border border-zinc-700">
              <pre className="text-xs text-zinc-300 whitespace-pre-wrap">{data.ocr_text}</pre>
            </div>
          </details>
        )}
      </div>
    );
  };

  return (
    <Card className="w-full bg-zinc-900 border-zinc-800">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileImage className="w-5 h-5" />
          Medical Image Analysis
        </CardTitle>
        <div className="flex gap-2">
          <Button
            variant={ocrMode === "image" ? "default" : "outline"}
            size="sm"
            onClick={() => setOcrMode("image")}
            className="flex items-center gap-2"
          >
            <Eye className="w-4 h-4" />
            Image Analysis
          </Button>
          <Button
            variant={ocrMode === "prescription" ? "default" : "outline"}
            size="sm"
            onClick={() => setOcrMode("prescription")}
            className="flex items-center gap-2"
          >
            <Pill className="w-4 h-4" />
            Prescription OCR
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Drag and drop area */}
        <div
          className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            dragActive
              ? "border-blue-400 bg-blue-400/10"
              : "border-zinc-600 hover:border-zinc-500"
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept="image/*"
            ref={fileInput}
            onChange={handleInputChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
          <div className="flex flex-col items-center gap-3">
            <Upload className="w-12 h-12 text-zinc-400" />
            <div>
              <p className="text-lg font-medium text-zinc-200">
                {dragActive ? "Drop your image here" : "Upload an image"}
              </p>
              <p className="text-sm text-zinc-400">
                Drag and drop or click to select • PNG, JPG, GIF up to 10MB
              </p>
            </div>
            <Button variant="outline" size="sm" onClick={() => fileInput.current?.click()}>
              Choose File
            </Button>
          </div>
        </div>

        {/* Upload Progress */}
        {loading && uploadProgress > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-zinc-400">
                {ocrMode === "prescription" ? "Extracting prescription..." : "Analyzing image..."}
              </span>
              <span className="text-zinc-300">{uploadProgress}%</span>
            </div>
            <Progress value={uploadProgress} className="h-2" />
          </div>
        )}

        {/* Image Preview */}
        {imageUrl && (
          <div className="space-y-3">
            <h4 className="font-medium text-zinc-200">Uploaded Image</h4>
            <div className="relative">
              <img 
                src={imageUrl} 
                alt="Uploaded" 
                className="max-w-full h-64 object-contain rounded-lg border border-zinc-700 bg-zinc-800" 
              />
              {loading && (
                <div className="absolute inset-0 bg-zinc-900/80 flex items-center justify-center rounded-lg">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400"></div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="flex items-center gap-2 p-3 bg-red-900/20 border border-red-800 rounded-lg">
            <XCircle className="w-5 h-5 text-red-400" />
            <div>
              <p className="font-medium text-red-300">Analysis Failed</p>
              <p className="text-sm text-red-400">{error}</p>
            </div>
          </div>
        )}

        {/* Results Display */}
        {result && !loading && (
          <div className="space-y-3">
            <h4 className="font-medium text-zinc-200 flex items-center gap-2">
              {ocrMode === "prescription" ? (
                <>
                  <Pill className="w-4 h-4" />
                  Prescription Analysis
                </>
              ) : (
                <>
                  <Eye className="w-4 h-4" />
                  Image Analysis
                </>
              )}
            </h4>
            {ocrMode === "prescription" ? (
              renderPrescriptionResult(result)
            ) : (
              <div className="bg-zinc-800 p-4 rounded-lg border border-zinc-700">
                <pre className="text-xs text-zinc-200 whitespace-pre-wrap overflow-auto max-h-64">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
