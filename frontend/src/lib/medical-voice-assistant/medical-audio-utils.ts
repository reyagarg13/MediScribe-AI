/**
 * Medical Audio Utils - Audio processing utilities for medical voice AI
 * Based on Olli audio utilities adapted for medical conversations
 */

// Audio encoding utilities
export function createAudioBlob(data: Float32Array): { data: string; mimeType: string } {
  // Convert Float32Array to Int16Array for Gemini compatibility
  const int16 = new Int16Array(data.length);
  for (let i = 0; i < data.length; i++) {
    // Clamp values to prevent overflow
    const sample = Math.max(-1, Math.min(1, data[i]));
    int16[i] = sample * 32767; // Convert to 16-bit integer
  }
  
  // Encode as base64
  const uint8 = new Uint8Array(int16.buffer);
  const base64 = btoa(String.fromCharCode(...uint8));
  
  return {
    data: base64,
    mimeType: 'audio/pcm;rate=16000'
  };
}

// Audio decoding utilities
export async function decodeAudioData(
  data: string | Uint8Array,
  audioContext: AudioContext,
  sampleRate: number,
  numChannels: number
): Promise<AudioBuffer> {
  try {
    let audioData: Uint8Array;
    
    if (typeof data === 'string') {
      // Decode base64 to Uint8Array
      const binaryString = atob(data);
      audioData = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        audioData[i] = binaryString.charCodeAt(i);
      }
    } else {
      audioData = data;
    }
    
    // Convert to Float32Array for AudioBuffer
    const int16Array = new Int16Array(audioData.buffer);
    const float32Array = new Float32Array(int16Array.length);
    
    for (let i = 0; i < int16Array.length; i++) {
      float32Array[i] = int16Array[i] / 32768; // Convert to float range [-1, 1]
    }
    
    // Create AudioBuffer
    const audioBuffer = audioContext.createBuffer(
      numChannels,
      float32Array.length / numChannels,
      sampleRate
    );
    
    // Fill the buffer
    for (let channel = 0; channel < numChannels; channel++) {
      const channelData = audioBuffer.getChannelData(channel);
      for (let i = 0; i < channelData.length; i++) {
        channelData[i] = float32Array[i * numChannels + channel];
      }
    }
    
    return audioBuffer;
    
  } catch (error) {
    console.error('Audio decoding error:', error);
    // Return silent buffer as fallback
    return audioContext.createBuffer(numChannels, sampleRate * 0.1, sampleRate);
  }
}

// Audio quality assessment for medical conversations
export function assessAudioQuality(audioData: Float32Array): {
  volume: number;
  clarity: number;
  quality: 'excellent' | 'good' | 'fair' | 'poor';
  recommendations: string[];
} {
  // Calculate RMS volume
  const rms = Math.sqrt(
    audioData.reduce((sum, sample) => sum + sample * sample, 0) / audioData.length
  );
  
  // Simple noise assessment (frequency domain analysis would be more accurate)
  const highFreqEnergy = calculateHighFrequencyEnergy(audioData);
  const clarity = Math.max(0, 1 - highFreqEnergy / rms);
  
  const recommendations: string[] = [];
  let quality: 'excellent' | 'good' | 'fair' | 'poor' = 'excellent';
  
  if (rms < 0.01) {
    quality = 'poor';
    recommendations.push('Speak louder or move closer to microphone');
  } else if (rms < 0.05) {
    quality = 'fair';
    recommendations.push('Slightly increase volume for better recognition');
  } else if (rms > 0.8) {
    quality = 'fair';
    recommendations.push('Reduce volume to prevent distortion');
  }
  
  if (clarity < 0.5) {
    if (quality === 'excellent') quality = 'good';
    if (quality === 'good') quality = 'fair';
    recommendations.push('Check for background noise or echo');
  }
  
  return {
    volume: rms,
    clarity,
    quality,
    recommendations
  };
}

function calculateHighFrequencyEnergy(audioData: Float32Array): number {
  // Simplified high-frequency energy calculation
  // In a real implementation, you'd use FFT for proper frequency analysis
  let highFreqSum = 0;
  for (let i = 1; i < audioData.length; i++) {
    const diff = audioData[i] - audioData[i - 1];
    highFreqSum += diff * diff;
  }
  return Math.sqrt(highFreqSum / audioData.length);
}

// Audio preprocessing for medical terms recognition
export function preprocessMedicalAudio(audioData: Float32Array): Float32Array {
  // Apply noise gate for medical environments
  const noiseGate = 0.02; // Threshold for medical office noise
  const processed = new Float32Array(audioData.length);
  
  for (let i = 0; i < audioData.length; i++) {
    if (Math.abs(audioData[i]) < noiseGate) {
      processed[i] = 0; // Gate noise
    } else {
      processed[i] = audioData[i];
    }
  }
  
  return processed;
}

// Audio context setup optimized for medical voice recognition
export function createMedicalAudioContext(inputSampleRate: number = 16000): {
  inputContext: AudioContext;
  outputContext: AudioContext;
  inputGain: GainNode;
  outputGain: GainNode;
} {
  // Input context for speech recognition (16kHz for efficiency)
  const inputContext = new AudioContext({
    sampleRate: inputSampleRate,
    latencyHint: 'interactive' // Low latency for real-time conversation
  });
  
  // Output context for high-quality speech synthesis (24kHz)
  const outputContext = new AudioContext({
    sampleRate: 24000,
    latencyHint: 'interactive'
  });
  
  // Create gain nodes for volume control
  const inputGain = inputContext.createGain();
  const outputGain = outputContext.createGain();
  
  // Set initial volumes optimized for medical conversations
  inputGain.gain.value = 1.0; // Full input sensitivity
  outputGain.gain.value = 0.8; // Slightly reduced output to prevent feedback
  
  return {
    inputContext,
    outputContext,
    inputGain,
    outputGain
  };
}

// Medical audio session configuration
export interface MedicalAudioConfig {
  inputSampleRate: number;
  outputSampleRate: number;
  bufferSize: number;
  echoCancellation: boolean;
  noiseSuppression: boolean;
  autoGainControl: boolean;
  medicalModeOptimizations: boolean;
}

export const defaultMedicalAudioConfig: MedicalAudioConfig = {
  inputSampleRate: 16000,
  outputSampleRate: 24000,
  bufferSize: 4096,
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
  medicalModeOptimizations: true
};

// Setup microphone with medical-optimized constraints
export async function setupMedicalMicrophone(
  config: MedicalAudioConfig = defaultMedicalAudioConfig
): Promise<MediaStream> {
  const constraints: MediaStreamConstraints = {
    audio: {
      sampleRate: config.inputSampleRate,
      channelCount: 1,
      echoCancellation: config.echoCancellation,
      noiseSuppression: config.noiseSuppression,
      autoGainControl: config.autoGainControl,
      
      // Medical environment optimizations
      ...(config.medicalModeOptimizations && {
        // Optimize for speech in medical settings
        googEchoCancellation: true,
        googNoiseSuppression: true,
        googHighpassFilter: true,
        googAutoGainControl: true,
        
        // Prefer high-quality microphones if available
        deviceId: undefined // Let browser choose best available
      })
    }
  };
  
  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    
    // Validate audio track settings
    const audioTrack = stream.getAudioTracks()[0];
    const settings = audioTrack.getSettings();
    
    console.log('🎤 Medical microphone setup:', {
      sampleRate: settings.sampleRate,
      channelCount: settings.channelCount,
      echoCancellation: settings.echoCancellation,
      noiseSuppression: settings.noiseSuppression,
      autoGainControl: settings.autoGainControl
    });
    
    return stream;
    
  } catch (error) {
    console.error('Failed to setup medical microphone:', error);
    throw new Error(`Microphone access failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

// Audio level monitoring for medical conversations
export class MedicalAudioMonitor {
  private analyser: AnalyserNode;
  private dataArray: Uint8Array;
  private isMonitoring: boolean = false;
  
  constructor(audioContext: AudioContext, source: MediaStreamAudioSourceNode) {
    this.analyser = audioContext.createAnalyser();
    this.analyser.fftSize = 256;
    this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
    
    source.connect(this.analyser);
  }
  
  startMonitoring(callback: (level: number, quality: string) => void) {
    this.isMonitoring = true;
    
    const monitor = () => {
      if (!this.isMonitoring) return;
      
      this.analyser.getByteFrequencyData(this.dataArray);
      
      // Calculate average level
      const average = this.dataArray.reduce((sum, value) => sum + value, 0) / this.dataArray.length;
      const level = average / 255; // Normalize to 0-1
      
      // Assess quality based on frequency distribution
      const lowFreq = this.dataArray.slice(0, 32).reduce((sum, v) => sum + v, 0) / 32;
      const midFreq = this.dataArray.slice(32, 96).reduce((sum, v) => sum + v, 0) / 64;
      const highFreq = this.dataArray.slice(96).reduce((sum, v) => sum + v, 0) / (this.dataArray.length - 96);
      
      let quality = 'good';
      if (midFreq < 50) quality = 'poor'; // Low speech frequencies
      else if (highFreq > midFreq * 2) quality = 'noisy'; // Too much high-frequency noise
      else if (midFreq > 150) quality = 'excellent'; // Strong speech signal
      
      callback(level, quality);
      
      requestAnimationFrame(monitor);
    };
    
    monitor();
  }
  
  stopMonitoring() {
    this.isMonitoring = false;
  }
}