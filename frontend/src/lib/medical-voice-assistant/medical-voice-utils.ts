/**
 * Medical Voice Utils - Voice response formatting for medical AI
 * Based on Olli voice utilities adapted for medical terminology and context
 */

// Format medical function responses for natural voice output
export function formatMedicalResponseForVoice(response: any, functionName: string): string {
  if (!response || !response.success) {
    return response?.message || "I encountered an issue with that medical query. Please try again.";
  }

  switch (functionName) {
    case 'lookup_medication':
      return formatMedicationLookupForVoice(response);
    
    case 'check_drug_interactions':
      return formatDrugInteractionsForVoice(response);
    
    case 'verify_allergy_safety':
      return formatAllergySafetyForVoice(response);
    
    case 'analyze_prescription_image':
      return formatPrescriptionAnalysisForVoice(response);
    
    case 'differential_diagnosis':
      return formatDifferentialDiagnosisForVoice(response);
    
    case 'vital_signs_interpretation':
      return formatVitalSignsForVoice(response);
    
    case 'patient_history_summary':
      return formatPatientHistoryForVoice(response);
    
    case 'clinical_guidelines':
      return formatClinicalGuidelinesForVoice(response);
    
    case 'medical_calculator':
      return formatMedicalCalculatorForVoice(response);
    
    case 'icd_code_lookup':
      return formatIcdLookupForVoice(response);
    
    case 'create_prescription':
      return formatPrescriptionCreationForVoice(response);
    
    default:
      return response.message || "I've completed that medical task for you.";
  }
}

function formatMedicationLookupForVoice(response: any): string {
  const { medication, dosing, contraindications, interactions } = response;
  
  let message = `I found information for ${medication}. `;
  
  if (dosing?.adult) {
    message += `The standard adult dose is ${dosing.adult}. `;
  }
  
  if (contraindications?.length > 0) {
    message += `Important contraindications include ${contraindications.slice(0, 2).join(' and ')}. `;
  }
  
  if (interactions?.length > 0) {
    message += `There are ${interactions.length} known interaction${interactions.length !== 1 ? 's' : ''} to review. `;
  }
  
  return message.trim();
}

function formatDrugInteractionsForVoice(response: any): string {
  const { checkedMedications, interactions, safetyScore } = response;
  
  if (interactions.length === 0) {
    return `Good news! I checked ${checkedMedications.length} medications and found no significant interactions. Safety score is ${safetyScore} out of 100.`;
  }
  
  let message = `I found ${interactions.length} interaction${interactions.length !== 1 ? 's' : ''} among your ${checkedMedications.length} medications. `;
  
  const criticalInteractions = interactions.filter((i: any) => i.severity === 'major');
  const moderateInteractions = interactions.filter((i: any) => i.severity === 'moderate');
  
  if (criticalInteractions.length > 0) {
    message += `${criticalInteractions.length} critical interaction${criticalInteractions.length !== 1 ? 's' : ''} require immediate attention. `;
  }
  
  if (moderateInteractions.length > 0) {
    message += `${moderateInteractions.length} moderate interaction${moderateInteractions.length !== 1 ? 's' : ''} should be monitored. `;
  }
  
  message += `Overall safety score is ${safetyScore} out of 100.`;
  
  return message;
}

function formatAllergySafetyForVoice(response: any): string {
  const { medication, safe, warnings } = response;
  
  if (safe) {
    return `✅ Good news! ${medication} appears safe based on the patient's known allergies.`;
  } else {
    return `⚠️ ALLERGY ALERT! ${medication} may cause an allergic reaction. ${warnings.join(' ')} Please verify patient allergies before proceeding.`;
  }
}

function formatPrescriptionAnalysisForVoice(response: any): string {
  const { extractedMedications, safetyScore, warnings, confidence } = response;
  
  let message = `I analyzed the prescription and found ${extractedMedications.length} medication${extractedMedications.length !== 1 ? 's' : ''}`;
  
  if (confidence) {
    message += ` with ${Math.round(confidence * 100)} percent confidence`;
  }
  
  if (safetyScore) {
    message += `. Safety score is ${Math.round(safetyScore)} out of 100`;
  }
  
  if (warnings?.length > 0) {
    message += `. ${warnings.length} warning${warnings.length !== 1 ? 's' : ''} require attention`;
  }
  
  return message + '.';
}

function formatDifferentialDiagnosisForVoice(response: any): string {
  const { symptoms, differentials } = response;
  
  let message = `Based on the ${symptoms.length} symptoms provided, I've generated a differential diagnosis. `;
  
  const topDifferentials = differentials.slice(0, 3);
  message += `The top considerations are: `;
  
  topDifferentials.forEach((diff: any, index: number) => {
    if (index > 0) message += index === topDifferentials.length - 1 ? ', and ' : ', ';
    message += `${diff.condition} with ${diff.probability.toLowerCase()} probability`;
  });
  
  message += '. I recommend monitoring symptom progression and considering diagnostic workup if symptoms persist.';
  
  return message;
}

function formatVitalSignsForVoice(response: any): string {
  const { vitals, assessments, overallStatus } = response;
  
  let message = `Vital signs assessment completed. `;
  
  if (assessments.length === 0) {
    message += 'All parameters are within normal limits.';
  } else {
    message += `${assessments.length} finding${assessments.length !== 1 ? 's' : ''} require attention: `;
    
    assessments.forEach((assessment: any, index: number) => {
      if (index > 0) message += ', ';
      message += `${assessment.parameter} is ${assessment.assessment.toLowerCase()}`;
    });
    
    message += '. Please review and consider appropriate interventions.';
  }
  
  return message;
}

function formatPatientHistoryForVoice(response: any): string {
  const { summary } = response;
  
  let message = `Patient history summary: `;
  
  if (summary.medications?.length > 0) {
    message += `Currently taking ${summary.medications.length} medication${summary.medications.length !== 1 ? 's' : ''}. `;
  }
  
  if (summary.allergies?.length > 0) {
    message += `${summary.allergies.length} known allergies on file. `;
  }
  
  if (summary.conditions?.length > 0) {
    message += `Active conditions include ${summary.conditions.join(', ')}. `;
  }
  
  if (summary.lastVisit) {
    message += `Last visit was ${formatDateForVoice(summary.lastVisit)}.`;
  }
  
  return message.trim();
}

function formatClinicalGuidelinesForVoice(response: any): string {
  const { condition, guidelines } = response;
  
  let message = `Clinical guidelines for ${condition}: `;
  
  if (guidelines.recommendations?.length > 0) {
    message += `Key recommendations include ${guidelines.recommendations.slice(0, 2).join(', and ')}. `;
  }
  
  if (guidelines.evidenceLevel) {
    message += `This is based on ${guidelines.evidenceLevel.toLowerCase()}. `;
  }
  
  if (guidelines.source) {
    message += `Guidelines from ${guidelines.source}.`;
  }
  
  return message;
}

function formatMedicalCalculatorForVoice(response: any): string {
  const { calculationType, result } = response;
  
  const calculationNames: Record<string, string> = {
    'bmi': 'Body Mass Index',
    'creatinine_clearance': 'Creatinine Clearance',
    'medication_dosage': 'Medication Dosage',
    'risk_score': 'Risk Score'
  };
  
  const name = calculationNames[calculationType] || calculationType;
  
  let message = `${name} calculation: ${result.value}`;
  
  if (result.category) {
    message += `, which is ${result.category.toLowerCase()}`;
  }
  
  if (result.interpretation) {
    message += `. ${result.interpretation}`;
  }
  
  return message;
}

function formatIcdLookupForVoice(response: any): string {
  const { condition, codes } = response;
  
  if (codes.length === 0) {
    return `I couldn't find ICD-10 codes for ${condition}. Please check the spelling or try a more specific term.`;
  }
  
  const primaryCode = codes[0];
  return `ICD-10 code for ${condition} is ${primaryCode.code}: ${primaryCode.description}.`;
}

function formatPrescriptionCreationForVoice(response: any): string {
  const { prescriptionId, medications } = response;
  
  return `Prescription created successfully with ID ${prescriptionId}. Added ${medications.length} medication${medications.length !== 1 ? 's' : ''} to the prescription.`;
}

// Utility functions for voice formatting

export function formatDateForVoice(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));
  
  if (diffDays === 0) return 'today';
  if (diffDays === 1) return 'yesterday';
  if (diffDays < 7) return `${diffDays} days ago`;
  if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
  if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;
  
  return date.toLocaleDateString();
}

export function formatMedicalTermForVoice(term: string): string {
  // Handle common medical abbreviations and make them voice-friendly
  const abbreviations: Record<string, string> = {
    'mg': 'milligrams',
    'ml': 'milliliters',
    'mcg': 'micrograms',
    'iu': 'international units',
    'bid': 'twice daily',
    'tid': 'three times daily',
    'qid': 'four times daily',
    'qd': 'once daily',
    'prn': 'as needed',
    'po': 'by mouth',
    'iv': 'intravenous',
    'im': 'intramuscular',
    'sq': 'subcutaneous',
    'gt': 'drops',
    'tab': 'tablet',
    'cap': 'capsule',
    'susp': 'suspension',
    'sol': 'solution',
    'oint': 'ointment',
    'cream': 'cream',
    'gel': 'gel'
  };
  
  let formatted = term.toLowerCase();
  
  // Replace abbreviations
  Object.entries(abbreviations).forEach(([abbr, full]) => {
    const regex = new RegExp(`\\b${abbr}\\b`, 'gi');
    formatted = formatted.replace(regex, full);
  });
  
  return formatted;
}

export function formatDosageForVoice(dosage: string): string {
  if (!dosage) return 'dosage not specified';
  
  // Convert numeric dosages to voice-friendly format
  let formatted = dosage.toLowerCase();
  
  // Handle common patterns
  formatted = formatted.replace(/(\d+)\s*mg/g, '$1 milligrams');
  formatted = formatted.replace(/(\d+)\s*ml/g, '$1 milliliters');
  formatted = formatted.replace(/(\d+)\s*mcg/g, '$1 micrograms');
  formatted = formatted.replace(/(\d+)\s*iu/g, '$1 international units');
  
  // Handle fractions
  formatted = formatted.replace('1/2', 'half');
  formatted = formatted.replace('1/4', 'quarter');
  formatted = formatted.replace('3/4', 'three quarters');
  
  return formatted;
}

export function formatFrequencyForVoice(frequency: string): string {
  if (!frequency) return 'frequency not specified';
  
  const frequencyMap: Record<string, string> = {
    'qd': 'once daily',
    'bid': 'twice daily',
    'tid': 'three times daily',
    'qid': 'four times daily',
    'q4h': 'every four hours',
    'q6h': 'every six hours',
    'q8h': 'every eight hours',
    'q12h': 'every twelve hours',
    'prn': 'as needed',
    'stat': 'immediately',
    'ac': 'before meals',
    'pc': 'after meals',
    'hs': 'at bedtime',
    'am': 'in the morning',
    'pm': 'in the evening'
  };
  
  let formatted = frequency.toLowerCase().trim();
  
  // Replace abbreviations
  Object.entries(frequencyMap).forEach(([abbr, full]) => {
    const regex = new RegExp(`\\b${abbr}\\b`, 'gi');
    formatted = formatted.replace(regex, full);
  });
  
  return formatted;
}

// Medical emergency responses
export function formatEmergencyResponse(type: 'allergy' | 'interaction' | 'contraindication'): string {
  switch (type) {
    case 'allergy':
      return 'URGENT: Potential allergic reaction risk detected. Please verify patient allergies immediately before administering this medication.';
    
    case 'interaction':
      return 'ALERT: Significant drug interaction detected. Review patient medications and consult clinical pharmacist if needed.';
    
    case 'contraindication':
      return 'WARNING: This medication is contraindicated for this patient. Consider alternative treatment options.';
    
    default:
      return 'Medical alert: Please review this finding carefully before proceeding.';
  }
}

// Professional medical communication tone
export function formatProfessionalResponse(message: string): string {
  // Ensure professional medical tone
  let formatted = message;
  
  // Add appropriate medical prefixes for serious findings
  if (formatted.toLowerCase().includes('contraindication') || 
      formatted.toLowerCase().includes('allergic reaction')) {
    formatted = 'Clinical Alert: ' + formatted;
  }
  
  // Ensure recommendations are clearly marked
  if (formatted.toLowerCase().includes('recommend') || 
      formatted.toLowerCase().includes('suggest')) {
    formatted = formatted.replace(/recommend/gi, 'I recommend');
    formatted = formatted.replace(/suggest/gi, 'I suggest');
  }
  
  return formatted;
}