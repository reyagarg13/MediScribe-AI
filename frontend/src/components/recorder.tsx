"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Mic, Square } from "lucide-react";
import { toast } from "sonner";

interface RecorderProps {
  onTranscriptionComplete?: (text: string) => void;
}

export function Recorder({ onTranscriptionComplete }: RecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [permission, setPermission] = useState<string>("unknown");
  const [statusMessage, setStatusMessage] = useState<string>("");
  const [logs, setLogs] = useState<string[]>([]);
  const [liveText, setLiveText] = useState<string>("");
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);
  const durationInterval = useRef<NodeJS.Timeout | null>(null);
  const recognitionRef = useRef<any>(null);

  const startRecording = async () => {
    try {
  log("Requesting microphone access...");
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  log("Microphone access granted");
  setPermission("granted");
  setStatusMessage("Recording started");
      mediaRecorder.current = new MediaRecorder(stream);
      audioChunks.current = [];

      mediaRecorder.current.ondataavailable = (event) => {
        audioChunks.current.push(event.data);
      };

      mediaRecorder.current.onstop = () => {
        const audioBlob = new Blob(audioChunks.current, { type: "audio/wav" });
        const audioUrl = URL.createObjectURL(audioBlob);
        // Here you would typically send the audioBlob to your transcription service
        toast.success("Recording saved!");
        log("Recording stopped, uploading audio to backend for transcription...");

        // Upload to backend /transcribe (FastAPI) - fallback/verification
        (async () => {
          try {
            const API_BASE = (window as any).NEXT_PUBLIC_API_BASE || process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
            const proxyUrl = "/api/proxy/transcribe";
            const fd = new FormData();
            fd.append("file", audioBlob, `recording-${Date.now()}.wav`);
            // Try uploading via Next.js proxy first to avoid CORS issues from the browser
            log(`Uploading to proxy ${proxyUrl}`);
            let res = null;
            try {
              res = await fetch(proxyUrl, { method: "POST", body: fd });
            } catch (proxyErr) {
              log(`Proxy upload failed, falling back to direct backend: ${String(proxyErr)}`);
            }

            if (!res) {
              log(`Uploading directly to ${API_BASE}/transcribe`);
              res = await fetch(`${API_BASE}/transcribe`, { method: "POST", body: fd });
            }
            if (!res.ok) {
              const text = await res.text();
              log(`Transcription upload failed: ${res.status} ${text}`);
              toast.error("Transcription failed on server");
              return;
            }
            const data = await res.json();
            const finalText = data.transcription || data.summary || "";
            log(`Server transcription received: ${finalText.slice(0, 120)}`);
            // If we got a server transcription, prefer it
            if (finalText) {
              onTranscriptionComplete?.(finalText);
            } else if (liveText) {
              onTranscriptionComplete?.(liveText);
            }
          } catch (err) {
            console.error(err);
            log(`Upload error: ${String(err)}`);
            toast.error("Failed to upload recording: check backend / CORS / network");
            // Fallback to liveText or a friendly message
            if (liveText) onTranscriptionComplete?.(liveText);
          }
        })();
      };

      mediaRecorder.current.start();
      setIsRecording(true);
      setDuration(0);

      // Try start SpeechRecognition for live/transient transcription (browser feature)
      try {
        const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
        if (SpeechRecognition) {
          recognitionRef.current = new SpeechRecognition();
          recognitionRef.current.continuous = true;
          recognitionRef.current.interimResults = true;
          recognitionRef.current.lang = "en-US"; // make configurable later
          recognitionRef.current.onresult = (ev: any) => {
            let interim = "";
            let final = "";
            for (let i = ev.resultIndex; i < ev.results.length; ++i) {
              const result = ev.results[i];
              if (result.isFinal) final += result[0].transcript;
              else interim += result[0].transcript;
            }
            if (interim) {
              setLiveText((prev) => interim);
              setStatusMessage("Listening (interim)");
            }
            if (final) {
              setLiveText((prev) => final);
              log(`Final local recognition: ${final}`);
              onTranscriptionComplete?.(final);
            }
          };
          recognitionRef.current.onerror = (e: any) => log(`SpeechRecognition error: ${e.error || e.message}`);
          recognitionRef.current.onend = () => {
            // If still recording, try to restart recognition
            if (isRecording) {
              try { recognitionRef.current.start(); } catch (e) { /* ignore */ }
            }
          };
          recognitionRef.current.start();
          log("SpeechRecognition started for live transcription (if supported)");
        } else {
          log("SpeechRecognition not supported in this browser");
        }
      } catch (err) {
        log(`SpeechRecognition init error: ${String(err)}`);
      }

      durationInterval.current = setInterval(() => {
        setDuration((prev) => prev + 1);
      }, 1000) as unknown as NodeJS.Timeout;
    } catch (error) {
  console.error("Error accessing microphone:", error);
  log(`Error accessing microphone: ${String(error)}`);
  setPermission("denied");
  toast.error("Could not access microphone");
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && isRecording) {
      mediaRecorder.current.stop();
      mediaRecorder.current.stream.getTracks().forEach((track) => track.stop());
      if (durationInterval.current) {
        clearInterval(durationInterval.current);
        durationInterval.current = null;
      }
      setIsRecording(false);
      setStatusMessage("Recording stopped");
      // stop speech recognition
      try {
        if (recognitionRef.current) {
          recognitionRef.current.onend = null;
          recognitionRef.current.stop();
          recognitionRef.current = null;
          log("SpeechRecognition stopped");
        }
      } catch (e) {
        /* ignore */
      }
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const log = (message: string) => {
    const ts = new Date().toLocaleTimeString();
    const entry = `[${ts}] ${message}`;
    console.log(entry);
    setLogs((prev) => [entry, ...prev].slice(0, 20));
  };

  useEffect(() => {
    // Try to query permissions where available
    (async () => {
      try {
        if ((navigator as any).permissions && (navigator as any).permissions.query) {
          const status = await (navigator as any).permissions.query({ name: "microphone" });
          setPermission(status.state);
          status.onchange = () => setPermission(status.state);
        }
      } catch (e) {
        // Not all browsers support the permissions API for microphone
        setPermission("unknown");
      }
    })();
  }, []);

  return (
    <Card className="w-full max-w-md p-6 space-y-4 bg-zinc-900 border-zinc-800">
      <div className="flex flex-col items-center space-y-4">
        <div className="relative w-32 h-32 flex items-center justify-center">
          <div className={`absolute inset-0 rounded-full ${isRecording ? "animate-pulse bg-red-500/20" : "bg-zinc-800"}`} />
          <Button
            size="lg"
            variant={isRecording ? "destructive" : "default"}
            className="relative w-24 h-24 rounded-full"
            onClick={isRecording ? stopRecording : startRecording}
          >
            {isRecording ? (
              <Square className="w-8 h-8" />
            ) : (
              <Mic className="w-8 h-8" />
            )}
          </Button>
        </div>

        <div className="w-full space-y-2">
          {isRecording && <Progress value={(duration % 60) * (100 / 60)} className="h-2" />}
          <p className="text-center text-sm text-zinc-400">{isRecording ? `Recording: ${formatDuration(duration)}` : "Idle"}</p>
          <p className="text-center text-xs text-zinc-500">Permission: {permission} {statusMessage ? `• ${statusMessage}` : ""}</p>
          {liveText && (
            <p className="text-center text-sm text-zinc-300">{liveText}</p>
          )}
        </div>

        {/* Simple log panel for debugging */}
        <div className="w-full mt-2 p-2 bg-zinc-950 border border-zinc-800 rounded text-xs text-zinc-500 h-28 overflow-auto">
          {logs.length === 0 ? (
            <div>No logs yet</div>
          ) : (
            logs.map((l, idx) => <div key={idx}>{l}</div>)
          )}
        </div>
      </div>
    </Card>
  );
}