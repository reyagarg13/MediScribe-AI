import React, { useState, useRef } from "react";

function Recorder() {
  const [recording, setRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [summary, setSummary] = useState("");
  const mediaRecorderRef = useRef(null);
  const audioChunks = useRef([]);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderRef.current = new MediaRecorder(stream);
    audioChunks.current = [];

    mediaRecorderRef.current.ondataavailable = (event) => {
      audioChunks.current.push(event.data);
    };

    mediaRecorderRef.current.onstop = async () => {
      const audioBlob = new Blob(audioChunks.current, { type: "audio/wav" });
      const formData = new FormData();
      formData.append("file", audioBlob, "recording.wav");

      // Call backend /transcribe
      const res = await fetch("http://127.0.0.1:8000/transcribe", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setTranscript(data.transcription);

      // Call backend /summarize
      const res2 = await fetch("http://127.0.0.1:8000/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcription: data.transcription }),
      });
      const summaryData = await res2.json();
      setSummary(summaryData.summary);
    };

    mediaRecorderRef.current.start();
    setRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    setRecording(false);
  };

  return (
    <div style={{ textAlign: "center", marginTop: "40px" }}>
      <h2>AI Medical Scribe 🎙️</h2>
      <button onClick={recording ? stopRecording : startRecording}>
        {recording ? "Stop Recording" : "Start Recording"}
      </button>

      {transcript && (
        <div>
          <h3>Transcript</h3>
          <p>{transcript}</p>
        </div>
      )}

      {summary && (
        <div>
          <h3>Summary</h3>
          <p>{summary}</p>
        </div>
      )}
    </div>
  );
}

export default Recorder;
