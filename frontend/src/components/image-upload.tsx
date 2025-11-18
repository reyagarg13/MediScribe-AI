import { useRef, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Upload, FileImage, Pill, Eye, CheckCircle, XCircle, AlertTriangle, Brain } from "lucide-react";

export function ImageUpload({ onResult }: { onResult: (result: any) => void }) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<{[key: string]: any}>({});
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [analysisStage, setAnalysisStage] = useState<string>("");
  const [analysisStartTime, setAnalysisStartTime] = useState<number | null>(null);
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

  // New: allow switching between endpoints (image analysis vs prescription OCR)
  const [ocrMode, setOcrMode] = useState<"image"|"prescription"|"advanced"|"custom">("prescription");
  const [advancedSettings, setAdvancedSettings] = useState({
    validationLevel: "standard",
    includeSafetyReport: true,
    patientAge: "",
    patientConditions: "",
    isPregnant: false
  });

  const handleFileChange = async (file: File) => {
    if (!file) return;
    
    // Reset any previous state
    setResults(prev => ({...prev, [ocrMode]: null}));
    setError(null);
    setImageUrl(URL.createObjectURL(file));
    setLoading(true);
    setAnalysisStartTime(Date.now());
    
    // Set analysis stages with realistic timing
    const stages = [
      { text: "Processing image", duration: 1000 },
      { text: "Extracting text", duration: 1200 },
      { text: "Analyzing prescription", duration: 1500 },
      { text: "Identifying medications", duration: 1000 }
    ];
    
    let currentStageIndex = 0;
    setAnalysisStage(stages[0].text);
    
    const stageTimeouts: NodeJS.Timeout[] = [];
    let cumulativeTime = 0;
    
    stages.forEach((stage, index) => {
      if (index > 0) {
        cumulativeTime += stages[index - 1].duration;
        const timeout = setTimeout(() => {
          setAnalysisStage(stage.text);
        }, cumulativeTime);
        stageTimeouts.push(timeout);
      }
    });
    
    const formData = new FormData();
    formData.append("file", file);
    
    try {
      let endpoint: string;
      let requestBody: FormData | string = formData;
      
      if (ocrMode === "advanced") {
        endpoint = "/prescription-ocr-advanced";
        // Add advanced parameters to form data
        if (advancedSettings.patientAge) {
          formData.append("patient_age", advancedSettings.patientAge);
        }
        if (advancedSettings.patientConditions) {
          formData.append("patient_conditions", advancedSettings.patientConditions);
        }
        formData.append("is_pregnant", String(advancedSettings.isPregnant));
        formData.append("validation_level", advancedSettings.validationLevel);
        formData.append("include_safety_report", String(advancedSettings.includeSafetyReport));
      } else if (ocrMode === "custom") {
        endpoint = "/prescription-custom-trocr";
      } else {
        endpoint = ocrMode === "prescription" ? "/prescription-ocr" : "/analyze-image";
      }
      
      const fullUrl = `${API_BASE}${endpoint}`;
      
      console.log(`🔗 Attempting to fetch: ${fullUrl}`);
      console.log(`🔗 Mode: ${ocrMode}`);
      
      const res = await fetch(fullUrl, {
        method: "POST",
        body: requestBody,
        mode: 'cors',
        credentials: 'omit',
        headers: {
          // Don't set Content-Type for FormData, let browser set it
        }
      });
      
      // Clear all stage timeouts
      stageTimeouts.forEach(timeout => clearTimeout(timeout));
      
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Server error: ${res.status} ${text}`);
      }
      const data = await res.json();
      console.log("📋 Backend response:", data);
      console.log("📋 Medications array:", data.medications);
      
      // Show completion message briefly
      const analysisTime = analysisStartTime ? ((Date.now() - analysisStartTime) / 1000).toFixed(1) : "0";
      setAnalysisStage(`Analyzed in ${analysisTime}s`);
      
      setTimeout(() => {
        setResults(prev => ({...prev, [ocrMode]: data}));
        onResult(data);
      }, 1000);
      
    } catch (err: any) {
      stageTimeouts.forEach(timeout => clearTimeout(timeout));
      console.error("Image analysis failed:", err);
      setError(err?.message || String(err));
    } finally {
      setTimeout(() => {
        setLoading(false);
        setAnalysisStage("");
        // Reset file input to allow same file upload again
        if (fileInput.current) {
          fileInput.current.value = "";
        }
      }, 1500);
    }
  };

  const handleInputChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      await handleFileChange(file);
    }
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await handleFileChange(e.dataTransfer.files[0]);
    }
  }, []);

  const renderImageProcessingShowcase = (data: any) => {
    if (data.status === "error") {
      return (
        <div className="flex items-center gap-2 text-red-400 p-4 bg-red-900/20 border border-red-800 rounded-lg">
          <AlertTriangle className="w-4 h-4" />
          <span>Error: {data.error}</span>
        </div>
      );
    }

    const techniques = data.techniques || {};

    return (
      <div className="space-y-6">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 p-4 rounded-lg border border-blue-700">
          <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
            🔬 Image Processing Techniques Showcase
          </h3>
          <div className="text-sm text-blue-300">
            Demonstrating {Object.keys(techniques).length} implemented techniques on your uploaded image
          </div>
        </div>

        {/* Original Image */}
        {data.original_image && (
          <div className="bg-zinc-900 p-4 rounded-lg border border-zinc-700">
            <h4 className="text-md font-semibold text-white mb-3 flex items-center gap-2">
              📷 Original Image
            </h4>
            <img 
              src={`data:image/jpeg;base64,${data.original_image}`}
              alt="Original"
              className="max-w-full h-64 object-contain rounded border border-zinc-600 bg-zinc-800"
            />
            <div className="mt-2 text-xs text-zinc-400">
              Original size: {data.original_size?.width} × {data.original_size?.height} pixels
            </div>
          </div>
        )}

        {/* Technique Results */}
        {Object.entries(techniques).map(([key, technique]: [string, any]) => (
          <div key={key} className="bg-zinc-900 p-4 rounded-lg border border-zinc-700">
            <h4 className="text-md font-semibold text-white mb-2 flex items-center gap-2">
              ⚡ {technique.title}
            </h4>
            
            {technique.description && (
              <p className="text-sm text-zinc-300 mb-3">{technique.description}</p>
            )}
            
            {technique.success && technique.image ? (
              <div className="space-y-3">
                <img 
                  src={`data:image/jpeg;base64,${technique.image}`}
                  alt={technique.title}
                  className="max-w-full h-64 object-contain rounded border border-zinc-600 bg-zinc-800"
                />
                <div className="flex items-center gap-2 text-green-400 text-xs">
                  <CheckCircle className="w-3 h-3" />
                  Processing successful
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-red-400 text-sm">
                <XCircle className="w-4 h-4" />
                <span>Failed: {technique.error || "Unknown error"}</span>
              </div>
            )}
          </div>
        ))}

        {/* Technical Summary */}
        <details className="mt-4">
          <summary className="cursor-pointer text-sm text-zinc-400 hover:text-zinc-200">
            View technical details
          </summary>
          <div className="mt-2 p-3 bg-zinc-900 rounded border border-zinc-700">
            <pre className="text-xs text-zinc-300 whitespace-pre-wrap overflow-auto max-h-64">
              {JSON.stringify(data, null, 2)}
            </pre>
          </div>
        </details>
      </div>
    );
  };

  const renderCustomTrOCRResult = (data: any) => {
    return (
      <div className="space-y-4">
        {/* Custom Model Information */}
        <div className="bg-gradient-to-r from-purple-900/30 to-blue-900/30 p-4 rounded-lg border border-purple-700">
          <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-400" />
            Custom TrOCR Model Results
          </h3>
          
          {data.model_info && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mb-4">
              <div className="bg-zinc-800/50 p-2 rounded">
                <div className="text-zinc-400 text-xs">Training Dataset</div>
                <div className="text-white font-medium">Kaggle Prescriptions</div>
              </div>
              <div className="bg-zinc-800/50 p-2 rounded">
                <div className="text-zinc-400 text-xs">Training Samples</div>
                <div className="text-white font-medium">{data.model_info.training_samples}</div>
              </div>
              <div className="bg-zinc-800/50 p-2 rounded">
                <div className="text-zinc-400 text-xs">Training Epochs</div>
                <div className="text-white font-medium">{data.model_info.epochs}</div>
              </div>
              <div className="bg-zinc-800/50 p-2 rounded">
                <div className="text-zinc-400 text-xs">Processing Time</div>
                <div className="text-white font-medium">{data.processing_time}s</div>
              </div>
            </div>
          )}

          {data.device && (
            <div className="text-sm text-purple-300 bg-purple-900/20 p-2 rounded">
              🚀 Processed using: {data.device.includes('cuda') ? 'NVIDIA RTX 4060 GPU' : 'CPU'}
            </div>
          )}
        </div>

        {/* Extracted Text */}
        <div className="bg-zinc-900 p-4 rounded-lg border border-zinc-700">
          <h4 className="text-md font-semibold text-white mb-3 flex items-center gap-2">
            <FileImage className="w-4 h-4 text-blue-400" />
            Extracted Prescription Text
          </h4>
          
          {data.extracted_text ? (
            <div className="bg-zinc-800 p-3 rounded border border-zinc-600">
              <pre className="text-sm text-zinc-200 whitespace-pre-wrap font-mono">
                {data.extracted_text}
              </pre>
            </div>
          ) : (
            <div className="flex items-center gap-2 text-amber-400">
              <AlertTriangle className="w-4 h-4" />
              <span>No text extracted from the image</span>
            </div>
          )}

          {/* Model Performance Info */}
          <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
            <div className="bg-zinc-800/50 p-2 rounded">
              <div className="text-zinc-400">Confidence</div>
              <div className="text-white font-medium">{Math.round((data.confidence || 0.85) * 100)}%</div>
            </div>
            <div className="bg-zinc-800/50 p-2 rounded">
              <div className="text-zinc-400">Model Status</div>
              <div className={`font-medium ${data.success ? 'text-green-400' : 'text-red-400'}`}>
                {data.success ? 'Success' : 'Failed'}
              </div>
            </div>
          </div>
        </div>

        {/* Technical Details */}
        <details className="mt-4">
          <summary className="cursor-pointer text-sm text-zinc-400 hover:text-zinc-200">
            View technical details
          </summary>
          <div className="mt-2 p-3 bg-zinc-900 rounded border border-zinc-700">
            <pre className="text-xs text-zinc-300 whitespace-pre-wrap overflow-auto max-h-64">
              {JSON.stringify(data, null, 2)}
            </pre>
          </div>
        </details>
      </div>
    );
  };

  const renderPrescriptionResult = (data: any) => {
    // Handle both basic and advanced response formats
    const medications = data.medications || [];
    const prescriptionHeader = data.prescription_header || {};
    const safetyReport = data.safety_report;
    const validationResults = data.validation_results;
    const isAdvanced = data.method === "advanced_ocr_with_validation";

    if (medications.length === 0) {
      return (
        <div className="flex items-center gap-2 text-zinc-400">
          <AlertTriangle className="w-4 h-4" />
          <span>No medications found in the image</span>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        {/* Prescription Header Information */}
        {Object.keys(prescriptionHeader).length > 0 && (
          <div className="bg-gradient-to-r from-indigo-900/20 to-blue-900/20 p-4 rounded-lg border border-indigo-800">
            <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
              <FileImage className="w-5 h-5 text-indigo-400" />
              Prescription Information
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
              {prescriptionHeader.patient_name && (
                <div className="bg-zinc-800/50 p-3 rounded">
                  <div className="text-zinc-400 text-xs">Patient Name</div>
                  <div className="text-white font-medium">{prescriptionHeader.patient_name}</div>
                </div>
              )}
              {prescriptionHeader.patient_age && (
                <div className="bg-zinc-800/50 p-3 rounded">
                  <div className="text-zinc-400 text-xs">Age</div>
                  <div className="text-white font-medium">{prescriptionHeader.patient_age}</div>
                </div>
              )}
              {prescriptionHeader.patient_sex && (
                <div className="bg-zinc-800/50 p-3 rounded">
                  <div className="text-zinc-400 text-xs">Sex</div>
                  <div className="text-white font-medium">{prescriptionHeader.patient_sex}</div>
                </div>
              )}
              {prescriptionHeader.prescription_date && (
                <div className="bg-zinc-800/50 p-3 rounded">
                  <div className="text-zinc-400 text-xs">Prescription Date</div>
                  <div className="text-white font-medium">{prescriptionHeader.prescription_date}</div>
                </div>
              )}
              {prescriptionHeader.doctor_name && (
                <div className="bg-zinc-800/50 p-3 rounded">
                  <div className="text-zinc-400 text-xs">Doctor</div>
                  <div className="text-white font-medium">{prescriptionHeader.doctor_name}</div>
                </div>
              )}
              {prescriptionHeader.clinic_name && (
                <div className="bg-zinc-800/50 p-3 rounded">
                  <div className="text-zinc-400 text-xs">Clinic</div>
                  <div className="text-white font-medium">{prescriptionHeader.clinic_name}</div>
                </div>
              )}
              {prescriptionHeader.patient_dob && (
                <div className="bg-zinc-800/50 p-3 rounded">
                  <div className="text-zinc-400 text-xs">Date of Birth</div>
                  <div className="text-white font-medium">{prescriptionHeader.patient_dob}</div>
                </div>
              )}
              {prescriptionHeader.clinic_address && (
                <div className="bg-zinc-800/50 p-3 rounded">
                  <div className="text-zinc-400 text-xs">Clinic Address</div>
                  <div className="text-white font-medium">{prescriptionHeader.clinic_address}</div>
                </div>
              )}
              {prescriptionHeader.phone && (
                <div className="bg-zinc-800/50 p-3 rounded">
                  <div className="text-zinc-400 text-xs">Phone</div>
                  <div className="text-white font-medium">{prescriptionHeader.phone}</div>
                </div>
              )}
              {prescriptionHeader.prescription_number && (
                <div className="bg-zinc-800/50 p-3 rounded">
                  <div className="text-zinc-400 text-xs">Prescription #</div>
                  <div className="text-white font-medium">{prescriptionHeader.prescription_number}</div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Safety Score (Advanced mode only) */}
        {isAdvanced && safetyReport && (
          <div className="bg-gradient-to-r from-blue-900/20 to-purple-900/20 p-4 rounded-lg border border-blue-800">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-semibold text-white">Safety Analysis</h3>
              <div className="flex items-center gap-2">
                <span className={`text-2xl font-bold ${
                  safetyReport.overall_safety_score >= 80 ? "text-green-400" :
                  safetyReport.overall_safety_score >= 60 ? "text-yellow-400" :
                  "text-red-400"
                }`}>
                  {safetyReport.overall_safety_score}/100
                </span>
                <span className="text-sm text-zinc-400">Safety Score</span>
              </div>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="bg-zinc-800/50 p-2 rounded">
                <div className="text-zinc-400">Total Medications</div>
                <div className="text-xl font-semibold text-white">{safetyReport.total_medications}</div>
              </div>
              <div className="bg-zinc-800/50 p-2 rounded">
                <div className="text-zinc-400">Critical Alerts</div>
                <div className="text-xl font-semibold text-red-400">{safetyReport.critical_alerts_count}</div>
              </div>
              <div className="bg-zinc-800/50 p-2 rounded">
                <div className="text-zinc-400">Warnings</div>
                <div className="text-xl font-semibold text-yellow-400">{safetyReport.warning_alerts_count}</div>
              </div>
              <div className="bg-zinc-800/50 p-2 rounded">
                <div className="text-zinc-400">Issues Found</div>
                <div className="text-xl font-semibold text-orange-400">{safetyReport.medications_with_issues}</div>
              </div>
            </div>

            {safetyReport.summary_recommendations && safetyReport.summary_recommendations.length > 0 && (
              <div className="mt-3">
                <h4 className="text-sm font-medium text-zinc-200 mb-2">Recommendations:</h4>
                <ul className="text-sm text-zinc-300 space-y-1">
                  {safetyReport.summary_recommendations.map((rec: string, idx: number) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="text-blue-400 mt-1">•</span>
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Medications List */}
        <div className="flex items-center gap-2 text-green-400">
          <CheckCircle className="w-4 h-4" />
          <span className="font-medium">Found {medications.length} medication(s)</span>
        </div>

        {medications.map((med: any, idx: number) => {
          const validation = validationResults && validationResults[idx];
          
          return (
            <div key={idx} className="bg-zinc-900 p-4 rounded-lg border border-zinc-700">
              {/* Medication Header */}
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Pill className="w-4 h-4 text-blue-400" />
                  <div className="flex flex-col">
                    {/* Show mapping if AI normalization occurred */}
                    {med.normalization ? (
                      <div className="font-medium text-white">
                        <span className="text-zinc-300 font-mono text-sm">{med.name_candidate}</span>
                        <span className="text-blue-400 mx-2">→</span>
                        <span className="text-green-400">{med.matched_name}</span>
                      </div>
                    ) : (
                      <span className="font-medium text-white">
                        {med.matched_name || med.name_candidate || med.raw_line || "Unknown item"}
                      </span>
                    )}
                  </div>
                </div>
                
                <div className="flex items-center gap-2">
                  {med.match_score && (
                    <span className={`text-xs px-2 py-1 rounded ${
                      med.match_score >= 80 ? "bg-green-900 text-green-300" :
                      med.match_score >= 60 ? "bg-yellow-900 text-yellow-300" :
                      "bg-red-900 text-red-300"
                    }`}>
                      {med.match_score}% match
                    </span>
                  )}
                  
                  {validation && (
                    <span className={`text-xs px-2 py-1 rounded ${
                      validation.safety_score >= 80 ? "bg-green-900 text-green-300" :
                      validation.safety_score >= 60 ? "bg-yellow-900 text-yellow-300" :
                      "bg-red-900 text-red-300"
                    }`}>
                      {validation.safety_score}/100 safety
                    </span>
                  )}
                </div>
              </div>

              {/* AI Recognition Note (if LLM normalization was used) */}
              {med.normalization && (
                <div className="mb-3 p-2 bg-blue-900/20 border border-blue-700/50 rounded">
                  <div className="text-xs text-blue-300 flex items-center gap-2">
                    <span>🤖</span>
                    <span>{med.normalization.reasoning}</span>
                  </div>
                </div>
              )}

              {/* Medication Details */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 text-sm mb-3">
                {/* Show complete dosage if available, otherwise fall back to simple dosage */}
                {(med.complete_dosage || med.dosage) && (
                  <div>
                    <span className="text-zinc-400">Complete Dosage:</span>
                    <span className="ml-1 text-zinc-200 font-medium">
                      {med.complete_dosage || med.dosage}
                    </span>
                  </div>
                )}
                {med.frequency && (
                  <div>
                    <span className="text-zinc-400">Frequency:</span>
                    <span className="ml-1 text-zinc-200">{med.frequency}</span>
                  </div>
                )}
                {med.duration && (
                  <div>
                    <span className="text-zinc-400">Duration:</span>
                    <span className="ml-1 text-zinc-200">{med.duration}</span>
                  </div>
                )}
                {med.meal_instructions && (
                  <div>
                    <span className="text-zinc-400">Meal Timing:</span>
                    <span className="ml-1 text-zinc-200 font-medium">{med.meal_instructions}</span>
                  </div>
                )}
                {med.route && (
                  <div>
                    <span className="text-zinc-400">Route:</span>
                    <span className="ml-1 text-zinc-200">{med.route}</span>
                  </div>
                )}
                {/* Show simple dosage separately if we also have complete dosage */}
                {med.complete_dosage && med.dosage && med.complete_dosage !== med.dosage && (
                  <div>
                    <span className="text-zinc-400">Extracted Dosage:</span>
                    <span className="ml-1 text-zinc-200 text-xs opacity-75">{med.dosage}</span>
                  </div>
                )}
              </div>

              {/* Validation Alerts (Advanced mode only) */}
              {validation && validation.alerts && validation.alerts.length > 0 && (
                <div className="mt-3 space-y-2">
                  <h5 className="text-sm font-medium text-zinc-200">Safety Alerts:</h5>
                  {validation.alerts.map((alert: any, alertIdx: number) => (
                    <div key={alertIdx} className={`p-2 rounded text-xs ${
                      alert.level === 'critical' ? 'bg-red-900/20 border border-red-800 text-red-300' :
                      alert.level === 'warning' ? 'bg-yellow-900/20 border border-yellow-800 text-yellow-300' :
                      'bg-blue-900/20 border border-blue-800 text-blue-300'
                    }`}>
                      <div className="font-medium">{alert.level.toUpperCase()}: {alert.message}</div>
                      <div className="mt-1 text-zinc-400">{alert.recommendation}</div>
                    </div>
                  ))}
                </div>
              )}

              {/* Generic Notes */}
              {med.notes && (
                <div className="mt-2 text-xs text-amber-400 bg-amber-900/20 p-2 rounded">
                  {med.notes}
                </div>
              )}
            </div>
          );
        })}

        {/* Processing Information (Advanced mode) */}
        {isAdvanced && data.ocr_analysis && (
          <details className="mt-4">
            <summary className="cursor-pointer text-sm text-zinc-400 hover:text-zinc-200">
              View processing details
            </summary>
            <div className="mt-2 p-3 bg-zinc-900 rounded border border-zinc-700">
              <div className="grid grid-cols-2 gap-4 text-xs mb-3">
                <div>
                  <span className="text-zinc-400">Processing Time:</span>
                  <span className="ml-1 text-zinc-200">{data.processing_time?.toFixed(2)}s</span>
                </div>
                <div>
                  <span className="text-zinc-400">Overall Confidence:</span>
                  <span className="ml-1 text-zinc-200">{data.ocr_analysis.overall_confidence}%</span>
                </div>
              </div>
              
            </div>
          </details>
        )}

        {/* OCR Text (Basic mode) */}
        {!isAdvanced && data.ocr_text && (
          <details className="mt-4">
            <summary className="cursor-pointer text-sm text-zinc-400 hover:text-zinc-200">
              View raw OCR text
            </summary>
            <div className="mt-2 p-3 bg-zinc-900 rounded border border-zinc-700">
              <pre className="text-xs text-zinc-300 whitespace-pre-wrap">{data.ocr_text}</pre>
            </div>
          </details>
        )}
      </div>
    );
  };

  return (
    <Card className="w-full bg-zinc-900 border-zinc-800">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileImage className="w-5 h-5" />
          Medical Image Analysis
        </CardTitle>
        <div className="flex gap-2 flex-wrap">
          <Button
            variant={ocrMode === "image" ? "default" : "outline"}
            size="sm"
            onClick={() => {
              setOcrMode("image");
            }}
            className="flex items-center gap-2"
          >
            <Eye className="w-4 h-4" />
            Image Analysis
          </Button>
          <Button
            variant={ocrMode === "prescription" ? "default" : "outline"}
            size="sm"
            onClick={() => {
              setOcrMode("prescription");
            }}
            className="flex items-center gap-2"
          >
            <Pill className="w-4 h-4" />
            Basic OCR
          </Button>
          <Button
            variant={ocrMode === "advanced" ? "default" : "outline"}
            size="sm"
            onClick={() => {
              setOcrMode("advanced");
            }}
            className="flex items-center gap-2"
          >
            <CheckCircle className="w-4 h-4" />
            Advanced OCR + Validation
          </Button>
          <Button
            variant={ocrMode === "custom" ? "default" : "outline"}
            size="sm"
            onClick={() => {
              setOcrMode("custom");
            }}
            className="flex items-center gap-2 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
          >
            <Brain className="w-4 h-4" />
            Custom TrOCR Model
          </Button>
        </div>
        
        {/* Advanced Settings Panel */}
        {ocrMode === "advanced" && (
          <div className="mt-4 p-4 bg-gradient-to-r from-blue-900/20 to-purple-900/20 rounded-lg border border-blue-700/50">
            <h4 className="text-sm font-medium text-blue-200 mb-3 flex items-center gap-2">
              ⚙️ Advanced Settings
              <span className="text-xs bg-blue-900/50 px-2 py-1 rounded text-blue-300">Configure validation</span>
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-zinc-400 font-medium">🔍 Validation Level</label>
                <select 
                  value={advancedSettings.validationLevel}
                  onChange={(e) => setAdvancedSettings(prev => ({...prev, validationLevel: e.target.value}))}
                  className="w-full mt-1 p-3 bg-zinc-700 border border-zinc-600 rounded text-zinc-200 text-sm hover:border-blue-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                >
                  <option value="standard">📋 Standard - Basic safety checks</option>
                  <option value="comprehensive">🔬 Comprehensive - Full medical analysis</option>
                </select>
                <p className="text-xs text-zinc-500 mt-1">
                  {advancedSettings.validationLevel === "comprehensive" 
                    ? "Includes detailed drug interactions, dosage validation, and contraindications" 
                    : "Basic drug recognition and safety warnings"}
                </p>
              </div>
              
              <div>
                <label className="text-xs text-zinc-400 font-medium">👤 Patient Age</label>
                <input 
                  type="number"
                  placeholder="Age (optional for dosage validation)"
                  value={advancedSettings.patientAge}
                  onChange={(e) => setAdvancedSettings(prev => ({...prev, patientAge: e.target.value}))}
                  className="w-full mt-1 p-3 bg-zinc-700 border border-zinc-600 rounded text-zinc-200 text-sm hover:border-blue-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                />
              </div>
              
              <div className="md:col-span-2">
                <label className="text-xs text-zinc-400 font-medium">🏥 Patient Conditions</label>
                <input 
                  type="text"
                  placeholder="e.g., diabetes, hypertension, kidney disease (optional for interaction checks)"
                  value={advancedSettings.patientConditions}
                  onChange={(e) => setAdvancedSettings(prev => ({...prev, patientConditions: e.target.value}))}
                  className="w-full mt-1 p-3 bg-zinc-700 border border-zinc-600 rounded text-zinc-200 text-sm hover:border-blue-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                />
                <p className="text-xs text-zinc-500 mt-1">Comma-separated list of medical conditions for contraindication checking</p>
              </div>
              
              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2 text-sm text-zinc-200">
                  <input 
                    type="checkbox"
                    checked={advancedSettings.isPregnant}
                    onChange={(e) => setAdvancedSettings(prev => ({...prev, isPregnant: e.target.checked}))}
                    className="rounded"
                  />
                  Pregnant Patient
                </label>
                
                <label className="flex items-center gap-2 text-sm text-zinc-200">
                  <input 
                    type="checkbox"
                    checked={advancedSettings.includeSafetyReport}
                    onChange={(e) => setAdvancedSettings(prev => ({...prev, includeSafetyReport: e.target.checked}))}
                    className="rounded"
                  />
                  Include Safety Report
                </label>
              </div>
            </div>
          </div>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Drag and drop area */}
        <div
          className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            dragActive
              ? "border-blue-400 bg-blue-400/10"
              : "border-zinc-600 hover:border-zinc-500"
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept="image/*"
            ref={fileInput}
            onChange={handleInputChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
          <div className="flex flex-col items-center gap-3">
            <Upload className="w-12 h-12 text-zinc-400" />
            <div>
              <p className="text-lg font-medium text-zinc-200">
                {dragActive ? "Drop your image here" : "Upload an image"}
              </p>
              <p className="text-sm text-zinc-400">
                Drag and drop or click to select • PNG, JPG, GIF up to 10MB
              </p>
            </div>
            <Button variant="outline" size="sm" onClick={() => fileInput.current?.click()}>
              Choose File
            </Button>
          </div>
        </div>

        {/* Modern Pulsing Analysis Animation */}
        {loading && analysisStage && (
          <div className="space-y-4 py-8">
            <div className="flex flex-col items-center justify-center space-y-4">
              {/* ChatGPT-style thinking animation */}
              <div className="flex items-center space-x-4">
                {/* Animated thinking dots */}
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0s' }}></div>
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
                
                {/* Glowing analysis text with breathing effect */}
                <div className="relative">
                  <span className="text-blue-400 font-medium text-lg animate-pulse drop-shadow-lg">
                    {analysisStage}
                  </span>
                  {/* Glow effect */}
                  <div className="absolute inset-0 text-blue-400 font-medium text-lg animate-pulse opacity-30 blur-sm">
                    {analysisStage}
                  </div>
                </div>
              </div>
              
              {/* Subtle glowing line with wave animation */}
              <div className="relative w-32 h-px bg-gradient-to-r from-transparent via-blue-400 to-transparent opacity-60 animate-pulse">
                {/* Additional glow layer */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-blue-300 to-transparent opacity-40 blur-sm animate-pulse"></div>
              </div>
            </div>
          </div>
        )}

        {/* Image Preview */}
        {imageUrl && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="font-medium text-zinc-200">Uploaded Image</h4>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setImageUrl(null);
                  setResults({});
                  setError(null);
                  if (fileInput.current) {
                    fileInput.current.value = "";
                  }
                }}
                className="text-xs"
              >
                Upload New Image
              </Button>
            </div>
            <div className="relative">
              <img 
                src={imageUrl} 
                alt="Uploaded" 
                className="max-w-full h-64 object-contain rounded-lg border border-zinc-700 bg-zinc-800" 
              />
              {loading && (
                <div className="absolute inset-0 bg-zinc-900/80 flex items-center justify-center rounded-lg">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400"></div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="flex items-center gap-2 p-3 bg-red-900/20 border border-red-800 rounded-lg">
            <XCircle className="w-5 h-5 text-red-400" />
            <div>
              <p className="font-medium text-red-300">Analysis Failed</p>
              <p className="text-sm text-red-400">{error}</p>
            </div>
          </div>
        )}

        {/* Results Display - Only show result for current mode */}
        {results[ocrMode] && !loading && (
          <div className="space-y-3">
            <h4 className="font-medium text-zinc-200 flex items-center gap-2">
              {ocrMode === "custom" ? (
                <>
                  <Brain className="w-4 h-4 text-purple-400" />
                  Custom TrOCR Model Analysis
                </>
              ) : ocrMode === "prescription" || ocrMode === "advanced" ? (
                <>
                  <Pill className="w-4 h-4" />
                  {ocrMode === "advanced" ? "Advanced Prescription Analysis" : "Prescription Analysis"}
                </>
              ) : (
                <>
                  <Eye className="w-4 h-4" />
                  Image Analysis
                </>
              )}
            </h4>
            {ocrMode === "custom" ? (
              renderCustomTrOCRResult(results[ocrMode])
            ) : ocrMode === "prescription" || ocrMode === "advanced" ? (
              renderPrescriptionResult(results[ocrMode])
            ) : ocrMode === "image" ? (
              renderImageProcessingShowcase(results[ocrMode])
            ) : (
              <div className="bg-zinc-800 p-4 rounded-lg border border-zinc-700">
                <pre className="text-xs text-zinc-200 whitespace-pre-wrap overflow-auto max-h-64">
                  {JSON.stringify(results[ocrMode], null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
