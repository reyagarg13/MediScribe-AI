"use client";

import { TranscriptionList } from "@/components/transcription-list";
import { useState, useRef } from "react";
import { ImageUpload } from "@/components/image-upload";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Stethoscope, Mic, MicOff, Volume2, VolumeX, Settings, Camera } from "lucide-react";
import Link from "next/link";
import { MedicalAudioService, type MedicalAudioServiceRef } from '@/lib/medical-voice-assistant/MedicalAudioService';

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
  
  // Medical Voice Assistant State
  const [isRecording, setIsRecording] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [status, setStatus] = useState('Ready for medical consultation');
  const [liveSpeech, setLiveSpeech] = useState('');
  const [conversation, setConversation] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [language, setLanguage] = useState('en-US');
  
  const audioServiceRef = useRef<MedicalAudioServiceRef>(null);

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

  // Medical Voice Assistant Handlers
  const handleStartStop = async () => {
    if (!audioServiceRef.current) return;

    try {
      if (isRecording) {
        audioServiceRef.current.stopRecording();
        setIsRecording(false);
      } else {
        await audioServiceRef.current.startRecording();
        setIsRecording(true);
      }
    } catch (error) {
      console.error('Failed to toggle recording:', error);
      setError(error instanceof Error ? error.message : 'Recording failed');
    }
  };

  const handleMuteToggle = () => {
    if (!audioServiceRef.current) return;
    
    const newMuted = !isMuted;
    audioServiceRef.current.setMuted(newMuted);
    setIsMuted(newMuted);
  };

  const handleStatusChange = (newStatus: string) => {
    setStatus(newStatus);
    setError(null);
  };

  const handleTranscriptUpdate = (transcriptEntry: any) => {
    setConversation(prev => [transcriptEntry, ...prev].slice(0, 20));
    // Also add to the legacy transcriptions for compatibility
    if (transcriptEntry.type === 'user') {
      handleTranscriptionComplete(transcriptEntry.content);
    }
  };

  const handleLiveSpeechUpdate = (speech: string) => {
    setLiveSpeech(speech);
    setCurrentTranscription(speech); // Update live transcription
  };

  const handleError = (errorMessage: string) => {
    setError(errorMessage);
    setIsRecording(false);
  };

  const sendTestMessage = () => {
    if (audioServiceRef.current) {
      audioServiceRef.current.sendTestMessage("Hello Dr. Aria, can you help me with some medical questions?");
    }
  };

  const defaultPatient = {
    id: 'main-patient',
    name: 'Current Patient',
    age: undefined,
    conditions: [],
    allergies: [],
    medications: []
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto space-y-8">
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold tracking-tight text-white">MediScribe</h1>
          <p className="text-zinc-300">Record and transcribe your voice, and analyze medical images with AI</p>
          
          {/* Dr. Aria Voice Assistant - Integrated with original design */}
          <div className="mt-6">
            <div className="bg-zinc-900/50 border border-zinc-700 rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4 flex items-center justify-center text-white">
                <Stethoscope className="w-6 h-6 mr-2 text-blue-400" />
                Dr. Aria - Medical AI Assistant
              </h3>
              
              {/* Status */}
              <div className={`mb-4 p-3 rounded-lg text-center ${error ? 'bg-red-900/30 text-red-300 border border-red-800' : 'bg-blue-900/30 text-blue-300 border border-blue-800'}`}>
                {error || status}
              </div>
              
              {/* Main Controls */}
              <div className="flex items-center justify-center gap-4 mb-4">
                {/* Recording Button */}
                <div className="relative">
                  <div className={`absolute inset-0 rounded-full ${isRecording ? 'animate-ping bg-green-400/75' : ''}`} />
                  <Button
                    onClick={handleStartStop}
                    variant={isRecording ? "destructive" : "default"}
                    className="w-16 h-16 rounded-full relative"
                    disabled={!!error}
                  >
                    {isRecording ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
                  </Button>
                </div>
                
                {/* Secondary Controls */}
                <div className="flex flex-col gap-2">
                  <Button
                    onClick={handleMuteToggle}
                    variant="ghost"
                    size="sm"
                    className="rounded-full text-zinc-300 hover:text-white"
                  >
                    {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
                  </Button>
                  <Button
                    onClick={sendTestMessage}
                    variant="outline"
                    size="sm"
                    className="text-zinc-300 border-zinc-600 hover:text-white hover:border-zinc-500"
                  >
                    Test
                  </Button>
                </div>
                
                {/* Language Selector */}
                <div className="text-left">
                  <label className="block text-xs font-medium text-zinc-400 mb-1">Language</label>
                  <select 
                    value={language} 
                    onChange={(e) => setLanguage(e.target.value)}
                    className="px-2 py-1 text-xs bg-zinc-800 border border-zinc-600 rounded-md text-zinc-300"
                  >
                    <option value="en-US">English</option>
                    <option value="hi-IN">हिंदी</option>
                  </select>
                </div>
              </div>
              
              {/* Live Speech Display */}
              {isRecording && liveSpeech && (
                <div className="bg-blue-900/20 border border-blue-800 rounded-lg p-3 mb-4">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                    <span className="text-xs font-medium text-blue-300">You're saying:</span>
                  </div>
                  <div className="text-sm text-zinc-200 leading-tight">{liveSpeech}</div>
                </div>
              )}
              
              {/* Quick Tips */}
              <div className="text-xs text-zinc-400 text-center">
                Try: "What are the side effects of Lisinopril?" or "Is ibuprofen safe with hypertension?"
                <br />
                <Link href="/test-conversation" className="text-blue-400 hover:text-blue-300 hover:underline">
                  Advanced conversation test page →
                </Link>
              </div>
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Left: Voice Transcriptions */}
          <div>
            <TranscriptionList 
              currentTranscription={currentTranscription}
              transcriptions={transcriptions}
            />
          </div>
          
          {/* Right: Conversation History */}
          <div>
            <div className="bg-zinc-900/50 border border-zinc-700 rounded-lg p-4">
              <h4 className="font-medium mb-3 flex items-center text-white">
                <Stethoscope className="w-4 h-4 mr-2 text-blue-400" />
                Medical Conversation
              </h4>
              <div className="space-y-3 max-h-80 overflow-y-auto">
                {conversation.length === 0 ? (
                  <p className="text-zinc-400 text-sm italic">Start talking to Dr. Aria...</p>
                ) : (
                  conversation.map((entry, i) => (
                    <div key={entry.id || i} className={`p-3 rounded-lg border ${
                      entry.type === 'user' ? 'bg-blue-900/20 border-blue-800' :
                      entry.type === 'assistant' ? 'bg-green-900/20 border-green-800' :
                      'bg-zinc-800/50 border-zinc-700'
                    }`}>
                      <div className="font-medium text-xs mb-1 text-zinc-300">
                        {entry.type === 'user' ? '👤 You' : 
                         entry.type === 'assistant' ? '🏥 Dr. Aria' : '⚙️ System'}
                      </div>
                      <div className="text-sm text-zinc-200">{entry.content}</div>
                      <div className="text-xs text-zinc-500 mt-1">
                        {new Date(entry.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
        
        <div className="mt-8">
          <h2 className="text-2xl font-semibold mb-4 text-white">Medical Image Analysis</h2>
          <ImageUpload onResult={setImageResult} />
        </div>
        
        {/* Hidden Medical Audio Service */}
        <MedicalAudioService
          ref={audioServiceRef}
          language={language}
          patientContext={defaultPatient}
          pageContext={{
            page: {
              url: '/',
              title: 'MediScribe Main',
              type: 'dashboard',
              section: 'main'
            },
            patient: defaultPatient,
            doctor: {
              id: 'main-doctor',
              name: 'Current Doctor',
              specialty: 'General Practice',
              experience: 'attending'
            }
          }}
          onStatusChange={handleStatusChange}
          onTranscriptUpdate={handleTranscriptUpdate}
          onMedicalUpdate={() => {}}
          onError={handleError}
          onLiveSpeechUpdate={handleLiveSpeechUpdate}
        />
      </div>
    </div>
  );
}