/**
 * Audio Worklet Processor for Real-time Medical AI
 * Replaces deprecated ScriptProcessorNode with modern AudioWorklet
 */

class MedicalAudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.processCount = 0;
    this.isRecording = false;
    this.silenceThreshold = 0.003;
    this.lastLogTime = 0;
    
    // Listen for messages from main thread
    this.port.onmessage = (event) => {
      const { type, data } = event.data;
      
      switch (type) {
        case 'start':
          this.isRecording = true;
          this.port.postMessage({ type: 'log', message: '🎤 AudioWorklet: Started recording' });
          break;
        case 'stop':
          this.isRecording = false;
          this.port.postMessage({ type: 'log', message: '⏸️ AudioWorklet: Stopped recording' });
          break;
        case 'setSilenceThreshold':
          this.silenceThreshold = data.threshold;
          break;
      }
    };
  }
  
  process(inputs, outputs, parameters) {
    this.processCount++;
    
    // Remove confusing debug logs
    
    const input = inputs[0];
    
    if (!input || !input[0] || !this.isRecording) {
      return true;
    }
    
    const channelData = input[0]; // First channel
    
    if (channelData.length === 0) {
      return true;
    }
    
    // Calculate RMS volume
    let sum = 0;
    for (let i = 0; i < channelData.length; i++) {
      sum += channelData[i] * channelData[i];
    }
    const rms = Math.sqrt(sum / channelData.length);
    
    // Send audio data if above threshold (no logging to reduce noise)
    if (rms > this.silenceThreshold) {
      
      // Convert Float32Array to regular array for serialization
      const audioData = Array.from(channelData);
      
      this.port.postMessage({
        type: 'audioData',
        data: {
          audioData: audioData,
          volume: rms,
          sampleRate: sampleRate,
          timestamp: currentFrame / sampleRate * 1000
        }
      });
    }
    
    return true; // Keep the processor alive
  }
}

registerProcessor('medical-audio-processor', MedicalAudioProcessor);