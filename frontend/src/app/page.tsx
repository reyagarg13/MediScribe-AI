"use client";

import { Recorder } from "@/components/recorder";
import { TranscriptionList } from "@/components/transcription-list";
import { useState } from "react";

// Example transcriptions for demonstration
const demoTranscriptions = [
  {
    id: "1",
    text: "Patient presents with symptoms of seasonal allergies including runny nose, watery eyes, and occasional sneezing.",
    timestamp: "2024-03-20 14:30",
    duration: "1:45"
  },
  {
    id: "2",
    text: "Follow-up appointment scheduled for next week. Prescribed antihistamines and recommended using a humidifier at night.",
    timestamp: "2024-03-20 14:25",
    duration: "2:30"
  }
];

export default function Home() {
  const [currentTranscription, setCurrentTranscription] = useState<string>("");
  const [transcriptions, setTranscriptions] = useState(demoTranscriptions);
  
  const handleTranscriptionComplete = (text: string) => {
    setCurrentTranscription(text);
    // Add to transcriptions list
    const newTranscription = {
      id: Date.now().toString(),
      text,
      timestamp: new Date().toLocaleString(),
      duration: "0:00" // You can pass the actual duration from the recorder
    };
    setTranscriptions(prev => [newTranscription, ...prev]);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto space-y-8">
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold tracking-tight">MediScribe</h1>
          <p className="text-zinc-400">Record and transcribe your voice with ease</p>
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
      </div>
    </div>
  );
}