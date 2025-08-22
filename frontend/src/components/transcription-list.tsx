"use client";

import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Copy, Download } from "lucide-react";
import { toast } from "sonner";

interface Transcription {
  id: string;
  text: string;
  timestamp: string;
  duration: string;
}

interface TranscriptionListProps {
  currentTranscription?: string;
  transcriptions: Transcription[];
}

export function TranscriptionList({ currentTranscription, transcriptions }: TranscriptionListProps) {
  const copyToClipboard = async (text: string) => {
    await navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard!");
  };

  const downloadTranscription = (transcription: Transcription) => {
    const element = document.createElement("a");
    const file = new Blob([transcription.text], { type: "text/plain" });
    element.href = URL.createObjectURL(file);
    element.download = `transcription-${transcription.timestamp}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
    toast.success("Download started!");
  };

  return (
    <div className="space-y-4">
      {currentTranscription && (
        <Card className="p-4 bg-zinc-900 border-zinc-800">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-zinc-200">Current Transcription</h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => copyToClipboard(currentTranscription)}
              >
                <Copy className="h-4 w-4" />
              </Button>
            </div>
            <p className="text-sm text-zinc-400">{currentTranscription}</p>
          </div>
        </Card>
      )}

      <div className="space-y-2">
        <h3 className="text-lg font-semibold text-zinc-200">Previous Transcriptions</h3>
        <ScrollArea className="h-[400px] rounded-md border border-zinc-800">
          {transcriptions.length === 0 ? (
            <div className="p-4 text-center text-sm text-zinc-500">
              No transcriptions yet
            </div>
          ) : (
            <div className="space-y-2 p-4">
              {transcriptions.map((transcription) => (
                <Card key={transcription.id} className="p-4 bg-zinc-900 border-zinc-800">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="space-y-1">
                        <p className="text-xs text-zinc-500">
                          {transcription.timestamp} • {transcription.duration}
                        </p>
                      </div>
                      <div className="flex gap-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => copyToClipboard(transcription.text)}
                        >
                          <Copy className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => downloadTranscription(transcription)}
                        >
                          <Download className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                    <p className="text-sm text-zinc-400">{transcription.text}</p>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </ScrollArea>
      </div>
    </div>
  );
}
