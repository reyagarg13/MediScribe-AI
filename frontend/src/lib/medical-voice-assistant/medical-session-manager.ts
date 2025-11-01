/**
 * Medical Session Manager - Singleton for persistent medical voice sessions
 * Based on Olli architecture but adapted for medical use cases
 */

// No need to import GoogleGenerativeAI here, the service handles the client

export interface MedicalTranscriptEntry {
  id: string;
  type: 'user' | 'assistant' | 'function_call' | 'medical_update';
  content: string;
  timestamp: string;
  metadata?: {
    functionName?: string;
    patientId?: string;
    confidence?: number;
    medicalData?: any;
  };
}

class MedicalSessionManager {
  private static instance: MedicalSessionManager;
  
  // Session state that persists across component lifecycles
  public currentStatus: string = 'idle';
  public isRecording: boolean = false;
  public isSessionActive: boolean = false;
  
  // Gemini client and session
  public client: any | null = null;
  public session: any = null; // Gemini session
  public model: any = null; // Gemini model
  public conversationId: string | null = null;
  
  // Audio contexts and streams (persist across component mounts)
  public inputAudioContext: AudioContext | null = null;
  public outputAudioContext: AudioContext | null = null;
  public inputNode: GainNode | null = null;
  public outputNode: GainNode | null = null;
  public mediaStream: MediaStream | null = null;
  public scriptProcessorNode: ScriptProcessorNode | null = null;
  public audioWorkletNode: AudioWorkletNode | null = null;
  
  // Medical conversation tracking
  public transcript: MedicalTranscriptEntry[] = [];
  public currentPatientId: string | null = null;
  public medicalContext: any = {};
  
  // Event handlers
  public onStatusChange?: (status: string) => void;
  public onTranscriptUpdate?: (transcript: MedicalTranscriptEntry) => void;
  public onMedicalUpdate?: (update: any) => void;
  
  // Status change notification
  private notifyStatusChange(status: string) {
    this.currentStatus = status;
    console.log('🏥 Medical Session Status:', status);
    this.onStatusChange?.(status);
  }
  
  // Add transcript entry
  public addTranscriptEntry(entry: Omit<MedicalTranscriptEntry, 'id' | 'timestamp'>) {
    const transcriptEntry: MedicalTranscriptEntry = {
      ...entry,
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString()
    };
    
    this.transcript.push(transcriptEntry);
    this.onTranscriptUpdate?.(transcriptEntry);
    
    // Keep transcript manageable (last 100 entries)
    if (this.transcript.length > 100) {
      this.transcript = this.transcript.slice(-100);
    }
    
    return transcriptEntry;
  }
  
  // Update medical context
  public updateMedicalContext(key: string, value: any) {
    this.medicalContext[key] = value;
    this.onMedicalUpdate?.({
      type: 'context_update',
      key,
      value,
      timestamp: new Date().toISOString()
    });
  }
  
  // Set current patient
  public setCurrentPatient(patientId: string | null) {
    if (this.currentPatientId !== patientId) {
      this.currentPatientId = patientId;
      this.updateMedicalContext('currentPatient', patientId);
      
      if (patientId) {
        this.notifyStatusChange(`Patient context: ${patientId}`);
      } else {
        this.notifyStatusChange('No patient context');
      }
    }
  }
  
  // Get current session status
  public getStatus() {
    return {
      status: this.currentStatus,
      isRecording: this.isRecording,
      isSessionActive: this.isSessionActive,
      hasClient: !!this.client,
      hasSession: !!this.session,
      hasAudioContext: !!(this.inputAudioContext && this.outputAudioContext),
      transcriptLength: this.transcript.length,
      currentPatient: this.currentPatientId,
      conversationId: this.conversationId
    };
  }
  
  // Start new medical session
  public async startSession() {
    this.isSessionActive = true;
    this.conversationId = `medical-${Date.now()}`;
    this.notifyStatusChange('Starting medical session...');
    
    this.addTranscriptEntry({
      type: 'assistant',
      content: 'Medical AI assistant activated. How can I help with patient care today?'
    });
  }
  
  // End current session (but keep it resumable)
  public pauseSession() {
    this.isRecording = false;
    this.notifyStatusChange('Medical session paused');
    
    this.addTranscriptEntry({
      type: 'assistant',
      content: 'Medical session paused. I\'ll be here when you need me.'
    });
  }
  
  // Completely end session and cleanup
  public endSession() {
    this.isSessionActive = false;
    this.isRecording = false;
    this.conversationId = null;
    
    // Cleanup audio resources
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
    
    if (this.audioWorkletNode) {
      this.audioWorkletNode.port.postMessage({ type: 'stop' });
      this.audioWorkletNode.disconnect();
      this.audioWorkletNode = null;
    }
    
    if (this.scriptProcessorNode) {
      this.scriptProcessorNode.disconnect();
      this.scriptProcessorNode = null;
    }
    
    // Keep audio contexts for quick restart
    // this.inputAudioContext?.close();
    // this.outputAudioContext?.close();
    
    this.notifyStatusChange('Medical session ended');
    
    this.addTranscriptEntry({
      type: 'assistant',
      content: 'Medical session ended. Thank you for using the AI assistant.'
    });
  }
  
  // Resume existing session
  public resumeSession() {
    if (this.conversationId) {
      this.isSessionActive = true;
      this.notifyStatusChange('Medical session resumed');
      
      this.addTranscriptEntry({
        type: 'assistant',
        content: 'Welcome back! Resuming our medical consultation.'
      });
    } else {
      this.startSession();
    }
  }
  
  // Get medical conversation summary
  public getMedicalSummary() {
    const medicalEntries = this.transcript.filter(entry => 
      entry.type === 'medical_update' || 
      (entry.type === 'function_call' && entry.metadata?.functionName?.includes('medical'))
    );
    
    return {
      totalEntries: this.transcript.length,
      medicalUpdates: medicalEntries.length,
      patientId: this.currentPatientId,
      sessionDuration: this.conversationId ? Date.now() - parseInt(this.conversationId.split('-')[1]) : 0,
      context: this.medicalContext
    };
  }
  
  // Singleton pattern
  public static getInstance(): MedicalSessionManager {
    if (!MedicalSessionManager.instance) {
      MedicalSessionManager.instance = new MedicalSessionManager();
    }
    return MedicalSessionManager.instance;
  }
  
  // For debugging
  public debug() {
    return {
      status: this.getStatus(),
      transcript: this.transcript.slice(-5), // Last 5 entries
      context: this.medicalContext,
      summary: this.getMedicalSummary()
    };
  }
}

export { MedicalSessionManager };