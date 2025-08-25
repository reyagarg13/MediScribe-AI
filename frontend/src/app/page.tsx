"use client";

import { Recorder } from "@/components/recorder";
import { TranscriptionList } from "@/components/transcription-list";
import { useState } from "react";
import { ImageUpload } from "@/components/image-upload";

type Transcription = {
  id: string;
  text: string;
  timestamp: string;
  duration: string;
};

const demoTranscriptions: Transcription[] = [];

 

export default function Page() {
  const [currentTranscription, setCurrentTranscription] = useState<string>("");
  const [transcriptions, setTranscriptions] = useState(demoTranscriptions);
  const [imageResult, setImageResult] = useState<any>(null);

  const handleTranscriptionComplete = (text: string) => {
    setCurrentTranscription(text);
    // Add to transcriptions list
    const newTranscription = {
      id: Date.now().toString(),
      text,
      timestamp: new Date().toLocaleString(),
      duration: "0:00" // You can pass the actual duration from the recorder
    };
    setTranscriptions((prev: Transcription[]) => [newTranscription, ...prev]);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto space-y-8">
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold tracking-tight">MediScribe</h1>
          <p className="text-zinc-400">Record and transcribe your voice, and analyze medical images with AI</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div>
            <Recorder onTranscriptionComplete={handleTranscriptionComplete} />
          </div>
          <div>
            <TranscriptionList 
              currentTranscription={currentTranscription}
              transcriptions={transcriptions}
            />
          </div>
        </div>
        <div className="mt-8">
          <h2 className="text-2xl font-semibold mb-4">Medical Image Analysis</h2>
          <ImageUpload onResult={setImageResult} />
        </div>
      </div>
    </div>
  );
}