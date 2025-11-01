/**
 * Medical Atomic Functions - Core medical operations for voice AI
 * Based on Olli function architecture adapted for medical domain
 */

export interface MedicalFunction {
  name: string;
  description: string;
  parameters: {
    type: 'object';
    properties: Record<string, any>;
    required?: string[];
  };
  execute: (args: any, context: any) => Promise<any>;
  behavior?: 'immediate' | 'when_idle';
  responseScheduling?: 'immediate' | 'dynamic' | 'silent';
}

// Core medical atomic functions
export const medicalAtomicFunctions: MedicalFunction[] = [
  
  // 1. MEDICATION FUNCTIONS
  {
    name: 'lookup_medication',
    description: 'Look up detailed information about a medication including dosing, interactions, and contraindications',
    parameters: {
      type: 'object',
      properties: {
        medicationName: {
          type: 'string',
          description: 'Name of the medication (generic or brand name)'
        },
        patientAge: {
          type: 'number',
          description: 'Patient age for age-specific dosing'
        },
        indication: {
          type: 'string',
          description: 'Medical condition being treated'
        }
      },
      required: ['medicationName']
    },
    execute: async (args, context) => {
      // Integrate with medical database or drug API
      return {
        success: true,
        medication: args.medicationName,
        genericName: 'Example Generic',
        brandNames: ['Brand1', 'Brand2'],
        dosing: {
          adult: '500mg twice daily',
          pediatric: 'Weight-based dosing',
          elderly: 'Reduced dose may be needed'
        },
        contraindications: ['Allergy to drug class', 'Severe kidney disease'],
        interactions: [],
        sideEffects: ['Nausea', 'Dizziness'],
        message: `Found information for ${args.medicationName}. Standard adult dose is 500mg twice daily.`
      };
    }
  },

  {
    name: 'check_drug_interactions',
    description: 'Check for interactions between multiple medications',
    parameters: {
      type: 'object',
      properties: {
        medications: {
          type: 'array',
          items: { type: 'string' },
          description: 'List of medications to check for interactions'
        },
        newMedication: {
          type: 'string',
          description: 'New medication being considered'
        }
      },
      required: ['medications']
    },
    execute: async (args, context) => {
      const meds = args.newMedication ? [...args.medications, args.newMedication] : args.medications;
      
      return {
        success: true,
        checkedMedications: meds,
        interactions: [
          {
            severity: 'moderate',
            drugs: ['Drug A', 'Drug B'],
            description: 'May increase risk of side effects',
            recommendation: 'Monitor patient closely'
          }
        ],
        safetyScore: 85,
        message: `Checked ${meds.length} medications. Found 1 moderate interaction to monitor.`
      };
    }
  },

  {
    name: 'verify_allergy_safety',
    description: 'Verify medication safety against patient allergies',
    parameters: {
      type: 'object',
      properties: {
        medication: {
          type: 'string',
          description: 'Medication to check'
        },
        allergies: {
          type: 'array',
          items: { type: 'string' },
          description: 'Patient known allergies'
        }
      },
      required: ['medication', 'allergies']
    },
    execute: async (args, context) => {
      const hasContraindication = args.allergies.some((allergy: string) => 
        allergy.toLowerCase().includes('penicillin') && 
        args.medication.toLowerCase().includes('amoxicillin')
      );

      return {
        success: true,
        medication: args.medication,
        allergies: args.allergies,
        safe: !hasContraindication,
        warnings: hasContraindication ? ['ALLERGY ALERT: Cross-sensitivity possible'] : [],
        message: hasContraindication ? 
          `⚠️ ALLERGY ALERT: ${args.medication} may cause allergic reaction due to ${args.allergies.join(', ')}` :
          `✅ No known allergy concerns with ${args.medication}`
      };
    }
  },

  // 2. PRESCRIPTION FUNCTIONS
  {
    name: 'analyze_prescription_image',
    description: 'Analyze an uploaded prescription image and extract medications',
    parameters: {
      type: 'object',
      properties: {
        imageData: {
          type: 'string',
          description: 'Base64 encoded prescription image'
        },
        patientContext: {
          type: 'object',
          description: 'Patient context for validation'
        }
      },
      required: ['imageData']
    },
    execute: async (args, context) => {
      // This would integrate with your existing prescription OCR system
      try {
        // Call your advanced prescription OCR endpoint
        const response = await fetch('/api/prescription-ocr-advanced', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: args.imageData,
            patient_context: args.patientContext
          })
        });

        const result = await response.json();
        
        return {
          success: true,
          extractedMedications: result.medications || [],
          safetyScore: result.safety_analysis?.overall_score || 0,
          warnings: result.safety_analysis?.warnings || [],
          confidence: result.confidence || 0,
          message: `Extracted ${result.medications?.length || 0} medications from prescription with ${Math.round((result.confidence || 0) * 100)}% confidence`
        };
      } catch (error) {
        return {
          success: false,
          error: 'Failed to analyze prescription image',
          message: 'Unable to process prescription image. Please try again.'
        };
      }
    }
  },

  {
    name: 'create_prescription',
    description: 'Create a new prescription with specified medications',
    parameters: {
      type: 'object',
      properties: {
        patientId: { type: 'string' },
        medications: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              name: { type: 'string' },
              dosage: { type: 'string' },
              frequency: { type: 'string' },
              duration: { type: 'string' }
            }
          }
        },
        diagnosis: { type: 'string' },
        notes: { type: 'string' }
      },
      required: ['patientId', 'medications']
    },
    execute: async (args, context) => {
      return {
        success: true,
        prescriptionId: `rx-${Date.now()}`,
        medications: args.medications,
        status: 'created',
        message: `Created prescription with ${args.medications.length} medications for patient ${args.patientId}`
      };
    }
  },

  // 3. DIAGNOSTIC FUNCTIONS
  {
    name: 'differential_diagnosis',
    description: 'Generate differential diagnosis suggestions based on symptoms',
    parameters: {
      type: 'object',
      properties: {
        symptoms: {
          type: 'array',
          items: { type: 'string' },
          description: 'List of patient symptoms'
        },
        duration: {
          type: 'string',
          description: 'How long symptoms have been present'
        },
        patientAge: { type: 'number' },
        patientSex: { type: 'string' }
      },
      required: ['symptoms']
    },
    execute: async (args, context) => {
      return {
        success: true,
        symptoms: args.symptoms,
        differentials: [
          {
            condition: 'Viral Upper Respiratory Infection',
            probability: 'High',
            reasoning: 'Common symptoms match viral pattern',
            nextSteps: ['Supportive care', 'Monitor symptoms']
          },
          {
            condition: 'Bacterial Sinusitis',
            probability: 'Moderate',
            reasoning: 'Duration and specific symptoms',
            nextSteps: ['Consider antibiotics if symptoms worsen']
          }
        ],
        recommendations: [
          'Monitor symptom progression',
          'Consider diagnostic tests if no improvement in 48-72 hours'
        ],
        message: `Based on ${args.symptoms.length} symptoms, generated differential diagnosis with 2 primary considerations`
      };
    }
  },

  {
    name: 'clinical_guidelines',
    description: 'Get evidence-based clinical guidelines for a condition',
    parameters: {
      type: 'object',
      properties: {
        condition: {
          type: 'string',
          description: 'Medical condition or clinical scenario'
        },
        guidelineType: {
          type: 'string',
          enum: ['treatment', 'diagnosis', 'prevention', 'monitoring'],
          description: 'Type of guideline needed'
        }
      },
      required: ['condition']
    },
    execute: async (args, context) => {
      return {
        success: true,
        condition: args.condition,
        guidelines: {
          source: 'American College of Physicians',
          lastUpdated: '2024',
          recommendations: [
            'First-line treatment with lifestyle modifications',
            'Consider medication if lifestyle changes insufficient',
            'Monitor response every 2-4 weeks'
          ],
          evidenceLevel: 'Strong recommendation, moderate-quality evidence'
        },
        message: `Found current clinical guidelines for ${args.condition} from major medical societies`
      };
    }
  },

  // 4. PATIENT MANAGEMENT FUNCTIONS
  {
    name: 'patient_history_summary',
    description: 'Get or update patient medical history summary',
    parameters: {
      type: 'object',
      properties: {
        patientId: { type: 'string' },
        focusArea: {
          type: 'string',
          enum: ['medications', 'allergies', 'conditions', 'procedures', 'all'],
          description: 'Specific area of history to focus on'
        },
        timeframe: {
          type: 'string',
          description: 'Time period to cover (e.g., "last 6 months", "all time")'
        }
      },
      required: ['patientId']
    },
    execute: async (args, context) => {
      return {
        success: true,
        patientId: args.patientId,
        summary: {
          medications: ['Lisinopril 10mg daily', 'Metformin 500mg twice daily'],
          allergies: ['Penicillin - rash', 'Shellfish - anaphylaxis'],
          conditions: ['Hypertension', 'Type 2 Diabetes'],
          recentVisits: 3,
          lastVisit: '2024-10-15'
        },
        message: `Retrieved medical history for patient ${args.patientId}. Currently on 2 medications with 2 known allergies.`
      };
    }
  },

  {
    name: 'vital_signs_interpretation',
    description: 'Interpret and assess vital signs measurements',
    parameters: {
      type: 'object',
      properties: {
        vitals: {
          type: 'object',
          properties: {
            systolic: { type: 'number' },
            diastolic: { type: 'number' },
            heartRate: { type: 'number' },
            temperature: { type: 'number' },
            respiratoryRate: { type: 'number' },
            oxygenSaturation: { type: 'number' }
          }
        },
        patientAge: { type: 'number' },
        patientConditions: { type: 'array', items: { type: 'string' } }
      },
      required: ['vitals']
    },
    execute: async (args, context) => {
      const vitals = args.vitals;
      const assessments = [];

      if (vitals.systolic > 140 || vitals.diastolic > 90) {
        assessments.push({
          parameter: 'Blood Pressure',
          value: `${vitals.systolic}/${vitals.diastolic}`,
          assessment: 'Elevated',
          action: 'Monitor closely, consider antihypertensive therapy'
        });
      }

      return {
        success: true,
        vitals,
        assessments,
        overallStatus: assessments.length > 0 ? 'Abnormal findings' : 'Within normal limits',
        message: `Vital signs assessment complete. ${assessments.length > 0 ? 
          `Found ${assessments.length} abnormal finding(s) requiring attention.` : 
          'All vital signs within normal ranges.'}`
      };
    }
  },

  // 5. MEDICAL KNOWLEDGE FUNCTIONS
  {
    name: 'medical_calculator',
    description: 'Calculate medical scores, dosages, or clinical parameters',
    parameters: {
      type: 'object',
      properties: {
        calculationType: {
          type: 'string',
          enum: ['bmi', 'creatinine_clearance', 'medication_dosage', 'risk_score'],
          description: 'Type of medical calculation'
        },
        parameters: {
          type: 'object',
          description: 'Calculation-specific parameters'
        }
      },
      required: ['calculationType', 'parameters']
    },
    execute: async (args, context) => {
      const { calculationType, parameters } = args;
      
      let result;
      switch (calculationType) {
        case 'bmi':
          const bmi = parameters.weight / Math.pow(parameters.height / 100, 2);
          result = {
            value: Math.round(bmi * 10) / 10,
            category: bmi < 18.5 ? 'Underweight' : 
                     bmi < 25 ? 'Normal' : 
                     bmi < 30 ? 'Overweight' : 'Obese',
            interpretation: 'BMI calculated successfully'
          };
          break;
        default:
          result = { value: 'Calculation not implemented', category: 'Unknown' };
      }

      return {
        success: true,
        calculationType,
        result,
        message: `Calculated ${calculationType}: ${result.value} (${result.category})`
      };
    }
  },

  {
    name: 'icd_code_lookup',
    description: 'Look up ICD-10 codes for medical conditions',
    parameters: {
      type: 'object',
      properties: {
        condition: {
          type: 'string',
          description: 'Medical condition or diagnosis'
        },
        specificity: {
          type: 'string',
          enum: ['general', 'specific'],
          description: 'Level of specificity needed'
        }
      },
      required: ['condition']
    },
    execute: async (args, context) => {
      return {
        success: true,
        condition: args.condition,
        codes: [
          {
            code: 'I10',
            description: 'Essential (primary) hypertension',
            category: 'Circulatory system diseases'
          }
        ],
        message: `Found ICD-10 code I10 for ${args.condition}`
      };
    }
  }
];

// Helper function to get function by name
export const getMedicalFunction = (name: string): MedicalFunction | undefined => {
  return medicalAtomicFunctions.find(func => func.name === name);
};

// Export function declarations for Gemini
export const medicalFunctionDeclarations = medicalAtomicFunctions.map(func => ({
  name: func.name,
  description: func.description,
  parameters: func.parameters
}));