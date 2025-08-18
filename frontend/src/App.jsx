import React from "react";
import Recorder from "./components/Recorder"; 

function App() {
  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>🩺 AI Medical Scribe</h1>
      <p>Record patient conversations → Get instant transcription + summary</p>
      <Recorder />
    </div>
  );
}

export default App;
