"use client";

import { useState, useRef } from "react";
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
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);
  const durationInterval = useRef<NodeJS.Timeout>();

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
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
        
        // Simulate transcription (replace this with actual transcription service)
        setTimeout(() => {
          onTranscriptionComplete?.(
            "This is a simulated transcription. Replace this with actual transcription service integration."
          );
        }, 1000);
      };

      mediaRecorder.current.start();
      setIsRecording(true);
      setDuration(0);

      durationInterval.current = setInterval(() => {
        setDuration((prev) => prev + 1);
      }, 1000);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      toast.error("Could not access microphone");
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && isRecording) {
      mediaRecorder.current.stop();
      mediaRecorder.current.stream.getTracks().forEach((track) => track.stop());
      clearInterval(durationInterval.current);
      setIsRecording(false);
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

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

        {isRecording && (
          <div className="w-full space-y-2">
            <Progress value={duration % 60 * (100 / 60)} className="h-2" />
            <p className="text-center text-sm text-zinc-400">
              Recording: {formatDuration(duration)}
            </p>
          </div>
        )}
      </div>
    </Card>
  );
}