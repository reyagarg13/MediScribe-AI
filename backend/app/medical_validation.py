"""
Medical Validation and Safety Checking System
"""
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ValidationLevel(Enum):
    """Different levels of medical validation"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class MedicalAlert:
    """Medical alert or warning"""
    level: AlertLevel
    category: str
    message: str
    recommendation: str
    source: str
    confidence: float


@dataclass
class DosageValidation:
    """Dosage validation result"""
    is_valid: bool
    parsed_value: Optional[float]
    parsed_unit: Optional[str]
    standard_range: Optional[Tuple[float, float]]
    alerts: List[MedicalAlert]


class MedicalValidator:
    """LLM-powered medical validation system with enhanced drug database integration"""
    
    def __init__(self):
        self.load_enhanced_database()
        self.load_basic_safety_data()
    
    def load_enhanced_database(self):
        """Load enhanced drug database for validation"""
        try:
            from .drug_database import drug_db, search_drug_enhanced
            self.enhanced_db = drug_db
            self.search_drug_enhanced = search_drug_enhanced
            self.enhanced_db_available = True
            print("✅ Enhanced drug database loaded for medical validation")
        except ImportError:
            self.enhanced_db = None
            self.search_drug_enhanced = None
            self.enhanced_db_available = False
            print("❌ Enhanced drug database not available for validation")
    
    def load_basic_safety_data(self):
        """Load basic safety data for critical medications"""
        # Keep minimal critical safety data for high-risk drugs
        self.critical_drugs = {
            "warfarin": {"monitoring": ["INR"], "black_box": True},
            "insulin": {"monitoring": ["glucose"], "black_box": False},
            "digoxin": {"monitoring": ["level", "ECG"], "black_box": True},
            "lithium": {"monitoring": ["level", "renal"], "black_box": True}
        }
        
        # Dosage unit conversions
        self.unit_conversions = {
            "mg": 1,
            "milligram": 1,
            "milligrams": 1,
            "g": 1000,
            "gram": 1000,
            "grams": 1000,
            "mcg": 0.001,
            "microgram": 0.001,
            "micrograms": 0.001,
            "μg": 0.001,
            "ml": 1,  # for liquids
            "milliliter": 1,
            "milliliters": 1,
            "l": 1000,
            "liter": 1000,
            "liters": 1000,
            "iu": 1,  # international units (varies by drug)
            "units": 1,
            "unit": 1
        }
        
        # Frequency conversions to daily
        self.frequency_conversions = {
            "once daily": 1,
            "twice daily": 2,
            "three times daily": 3,
            "four times daily": 4,
            "every 6 hours": 4,
            "every 8 hours": 3,
            "every 12 hours": 2,
            "every 24 hours": 1,
            "q6h": 4,
            "q8h": 3,
            "q12h": 2,
            "q24h": 1,
            "qid": 4,
            "tid": 3,
            "bid": 2,
            "qd": 1
        }
        
        # Drug interaction matrix - now handled by LLM analysis
        self.interaction_matrix = {}
        
        # Pregnancy categories
        self.pregnancy_categories = {
            "A": "No risk in controlled studies",
            "B": "No risk in animal studies",
            "C": "Risk cannot be ruled out",
            "D": "Positive evidence of risk",
            "X": "Contraindicated in pregnancy"
        }
    
    
    def _llm_batch_validate_medications(self, medications: List[Dict], patient_info: Optional[Dict] = None) -> Dict[str, List[MedicalAlert]]:
        """
        Batch validate multiple medications in a single LLM call for speed
        """
        if not medications:
            return {}
            
        try:
            from . import gemini
            if not hasattr(gemini, 'analyze_with_gemini') or not gemini.available():
                return {}
            
            # Prepare medication list for batch analysis
            med_list = []
            for i, med in enumerate(medications):
                matched_name = med.get("matched_name", med.get("name_candidate", ""))
                original_name = med.get("name_candidate", "")
                dosage = med.get("complete_dosage", med.get("dosage", ""))
                frequency = med.get("frequency", "")
                duration = med.get("duration", "")
                
                med_list.append(f"{i+1}. {matched_name} (original: {original_name})")
                med_list.append(f"   Dosage: {dosage}")
                med_list.append(f"   Frequency: {frequency}")
                med_list.append(f"   Duration: {duration}")
            
            medications_text = "\n".join(med_list)
            
            # Get other drug names for interaction checking
            drug_names = [med.get("matched_name", med.get("name_candidate", "")) for med in medications]
            
            patient_context = ""
            if patient_info:
                age = patient_info.get("age", "")
                conditions = patient_info.get("conditions", [])
                if age: patient_context += f"Age: {age}. "
                if conditions: patient_context += f"Conditions: {', '.join(conditions)}. "
            
            # Create comprehensive batch validation prompt
            batch_prompt = f"""
You are a clinical pharmacist reviewing a prescription with multiple medications. Analyze ALL medications together for safety concerns.

MEDICATIONS TO ANALYZE:
{medications_text}

PATIENT CONTEXT:
{patient_context if patient_context else "No patient information provided"}

ANALYZE FOR:
1. Individual medication safety (dosage appropriateness, frequency clarity)
2. Drug-drug interactions between the medications listed
3. Contraindications based on patient conditions
4. Any red flags requiring immediate attention

RULES:
- Only flag genuine safety concerns, not routine checks
- CRITICAL: Reserved for immediate dangers (overdose, severe interactions, absolute contraindications)
- WARNING: For dosage concerns, moderate interactions, relative contraindications  
- INFO: For routine allergy checks, monitoring requirements, general precautions
- DO NOT flag routine allergy history verification as CRITICAL unless patient has known allergies
- Focus on major/moderate interactions and safety issues
- Be concise and specific

RESPOND IN THIS FORMAT:
MEDICATION 1: [drug_name]
SAFE: [yes/no]
CONCERNS: [list concerns or "none"]
RECOMMENDATIONS: [recommendations or "none"]

MEDICATION 2: [drug_name]
SAFE: [yes/no]
CONCERNS: [list concerns or "none"]  
RECOMMENDATIONS: [recommendations or "none"]

OVERALL INTERACTIONS:
[Any significant drug interactions between the medications, or "none detected"]
"""
            
            result = gemini.analyze_with_gemini(batch_prompt)
            
            if result:
                return self._parse_batch_validation_response(result, medications)
                
        except Exception as e:
            print(f"Batch validation failed: {e}")
        
        return {}
    
    def _parse_batch_validation_response(self, response: str, medications: List[Dict]) -> Dict[str, List[MedicalAlert]]:
        """Parse batch validation response into alerts per medication"""
        alerts_dict = {}
        
        try:
            sections = response.split("MEDICATION ")
            
            for i, section in enumerate(sections[1:], 1):  # Skip first empty section
                if i <= len(medications):
                    med = medications[i-1]
                    drug_name = med.get("matched_name", med.get("name_candidate", ""))
                    
                    alerts = []
                    lines = section.split('\n')
                    
                    safe = "yes"
                    concerns = []
                    recommendations = []
                    
                    for line in lines:
                        line = line.strip()
                        if line.startswith("SAFE:"):
                            safe = line.split(":", 1)[1].strip().lower()
                        elif line.startswith("CONCERNS:"):
                            concerns_text = line.split(":", 1)[1].strip()
                            if concerns_text.lower() not in ["none", ""]:
                                # Split on semicolons or periods, not commas (to preserve phrases with commas)
                                concerns = [c.strip() for c in concerns_text.replace(";", ".").split(".") if c.strip()]
                                if not concerns:  # fallback to single concern
                                    concerns = [concerns_text]
                        elif line.startswith("RECOMMENDATIONS:"):
                            rec_text = line.split(":", 1)[1].strip()
                            if rec_text.lower() not in ["none", ""]:
                                # Keep full recommendation text, don't split on commas
                                recommendations = [rec_text]
                    
                    # Convert to alerts
                    if safe == "no" or concerns:
                        for j, concern in enumerate(concerns):
                            if concern:
                                rec = recommendations[j] if j < len(recommendations) else "Consult healthcare provider"
                                
                                # Smarter alert level detection
                                concern_lower = concern.lower()
                                if any(word in concern_lower for word in ["overdose", "toxic", "deadly", "severe interaction", "absolute contraindication"]):
                                    level = AlertLevel.CRITICAL
                                elif any(word in concern_lower for word in ["allergy", "history", "verify", "check", "monitor", "caution"]):
                                    level = AlertLevel.INFO
                                elif any(word in concern_lower for word in ["high", "low", "moderate", "interaction", "contraindicated"]):
                                    level = AlertLevel.WARNING
                                else:
                                    level = AlertLevel.WARNING
                                
                                alerts.append(MedicalAlert(
                                    level=level,
                                    category="batch_validation",
                                    message=f"{drug_name}: {concern}",
                                    recommendation=rec,
                                    source="llm_batch_pharmacist",
                                    confidence=0.85
                                ))
                    
                    alerts_dict[drug_name] = alerts
                    
        except Exception as e:
            print(f"Failed to parse batch validation response: {e}")
        
        return alerts_dict

    def _llm_validate_medication(self, drug_name: str, original_name: str, dosage: str, 
                                frequency: str, duration: str, patient_info: Optional[Dict] = None,
                                all_medications: Optional[List] = None) -> List[MedicalAlert]:
        """LLM-powered comprehensive medication validation"""
        alerts = []
        
        try:
            from . import gemini
            if not hasattr(gemini, 'analyze_with_gemini') or not gemini.available():
                return alerts
            
            # Prepare context for LLM
            other_drugs = []
            if all_medications:
                other_drugs = [med.get("matched_name", med.get("name_candidate", "")) 
                              for med in all_medications 
                              if med.get("matched_name") != drug_name]
            
            patient_context = ""
            if patient_info:
                age = patient_info.get("age", "")
                weight = patient_info.get("weight", "")
                if age: patient_context += f"Age: {age}. "
                if weight: patient_context += f"Weight: {weight}. "
            
            # Create comprehensive validation prompt
            validation_prompt = f"""
You are a clinical pharmacist reviewing a prescription. Analyze this medication for safety concerns.

MEDICATION DETAILS:
- Drug: {drug_name}
- Original prescription text: {original_name}
- Dosage: {dosage}
- Frequency: {frequency}
- Duration: {duration}

PATIENT CONTEXT:
{patient_context if patient_context else "No patient information provided"}

OTHER MEDICATIONS:
{', '.join(other_drugs) if other_drugs else "None listed"}

ANALYZE FOR:
1. Dosage appropriateness (too high/low for typical use)
2. Frequency format issues (unclear timing)
3. Drug interactions with other medications listed
4. Common contraindications or warnings
5. Any red flags requiring immediate attention

RULES:
- Only flag genuine safety concerns, not routine checks
- CRITICAL: Reserved for immediate dangers (overdose, severe interactions, absolute contraindications)
- WARNING: For dosage concerns, moderate interactions, relative contraindications  
- INFO: For routine allergy checks, monitoring requirements, general precautions
- DO NOT flag routine allergy history verification as CRITICAL unless patient has known allergies
- For dosage: Only warn if clearly outside normal ranges
- For frequency: Only warn if format is unclear or dangerous
- For interactions: Only major/moderate interactions
- Be concise and specific

RESPOND IN THIS FORMAT:
SAFE: [yes/no]
CONCERNS: [list major concerns only, or "none"]
RECOMMENDATIONS: [specific actionable recommendations, or "none"]

Example response:
SAFE: yes
CONCERNS: none  
RECOMMENDATIONS: none

OR

SAFE: no
CONCERNS: dosage appears high for pediatric use, unclear frequency format
RECOMMENDATIONS: verify pediatric dosing guidelines, clarify frequency as "every 8 hours" instead of "TDS"
"""
            
            response = gemini.analyze_with_gemini(validation_prompt)
            
            if response:
                alerts.extend(self._parse_llm_validation_response(response, drug_name))
                
        except Exception as e:
            print(f"LLM validation failed for {drug_name}: {e}")
        
        return alerts
    
    def _parse_llm_validation_response(self, response: str, drug_name: str) -> List[MedicalAlert]:
        """Parse LLM validation response into alerts"""
        alerts = []
        
        try:
            lines = response.strip().split('\n')
            safe = "yes"
            concerns = []
            recommendations = []
            
            for line in lines:
                line = line.strip()
                if line.startswith("SAFE:"):
                    safe = line.split(":", 1)[1].strip().lower()
                elif line.startswith("CONCERNS:"):
                    concerns_text = line.split(":", 1)[1].strip()
                    if concerns_text.lower() not in ["none", ""]:
                        # Split on semicolons or periods, not commas (to preserve phrases with commas)
                        concerns = [c.strip() for c in concerns_text.replace(";", ".").split(".") if c.strip()]
                        if not concerns:  # fallback to single concern
                            concerns = [concerns_text]
                elif line.startswith("RECOMMENDATIONS:"):
                    rec_text = line.split(":", 1)[1].strip()
                    if rec_text.lower() not in ["none", ""]:
                        # Keep full recommendation text, don't split on commas
                        recommendations = [rec_text]
            
            # Convert to alerts
            if safe == "no" or concerns:
                for i, concern in enumerate(concerns):
                    if concern:
                        rec = recommendations[i] if i < len(recommendations) else "Consult healthcare provider"
                        
                        # Determine alert level with smarter categorization
                        concern_lower = concern.lower()
                        if any(word in concern_lower for word in ["overdose", "toxic", "deadly", "severe interaction", "absolute contraindication"]):
                            level = AlertLevel.CRITICAL
                        elif any(word in concern_lower for word in ["allergy", "history", "verify", "check", "monitor", "caution"]):
                            level = AlertLevel.INFO
                        elif any(word in concern_lower for word in ["high", "low", "moderate", "interaction", "contraindicated"]):
                            level = AlertLevel.WARNING
                        else:
                            level = AlertLevel.WARNING
                        
                        alerts.append(MedicalAlert(
                            level=level,
                            category="llm_validation",
                            message=f"{drug_name}: {concern}",
                            recommendation=rec,
                            source="llm_pharmacist",
                            confidence=0.85
                        ))
                        
        except Exception as e:
            print(f"Failed to parse LLM validation response: {e}")
        
        return alerts
    
    def validate_medication(
        self, 
        medication: Dict[str, Any], 
        all_medications: List[Dict[str, Any]] = None,
        patient_info: Dict[str, Any] = None,
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> Dict[str, Any]:
        """Comprehensive medication validation"""
        
        alerts = []
        original_name = medication.get("name_candidate", "")
        matched_name = medication.get("matched_name", "")
        dosage = medication.get("dosage", "") or medication.get("complete_dosage", "")
        frequency = medication.get("frequency", "")
        duration = medication.get("duration", "")
        
        # Use LLM-powered validation for comprehensive safety analysis
        if matched_name and self.enhanced_db_available:
            # For individual validation, check if we can batch with others
            # This is used for single medication validation
            llm_alerts = self._llm_validate_medication(
                drug_name=matched_name,
                original_name=original_name,
                dosage=dosage,
                frequency=frequency,
                duration=duration,
                patient_info=patient_info,
                all_medications=all_medications
            )
            alerts.extend(llm_alerts)
            
            # Check for critical drug monitoring
            critical_info = self.critical_drugs.get(matched_name.lower())
            if critical_info:
                alerts.append(MedicalAlert(
                    level=AlertLevel.WARNING,
                    category="monitoring",
                    message=f"High-risk medication: {matched_name} requires monitoring",
                    recommendation=f"Monitor: {', '.join(critical_info['monitoring'])}",
                    source="critical_drug_list",
                    confidence=0.9
                ))
        
        elif not matched_name:
            # No match found - drug not recognized
            alerts.append(MedicalAlert(
                level=AlertLevel.WARNING,
                category="drug_recognition",
                message=f"Drug '{original_name}' not recognized",
                recommendation="Verify drug name spelling and check against standard formularies",
                source="enhanced_database",
                confidence=0.8
            ))
        
        # Calculate overall safety score
        critical_alerts = sum(1 for alert in alerts if alert.level == AlertLevel.CRITICAL)
        warning_alerts = sum(1 for alert in alerts if alert.level == AlertLevel.WARNING)
        info_alerts = sum(1 for alert in alerts if alert.level == AlertLevel.INFO)
        
        safety_score = max(0, 100 - (critical_alerts * 40) - (warning_alerts * 15) - (info_alerts * 5))
        
        return {
            "medication_name": matched_name or original_name,
            "original_name": original_name,
            "validation_status": "validated" if not any(a.level == AlertLevel.CRITICAL for a in alerts) else "critical_issues",
            "safety_score": safety_score,
            "alerts": [
                {
                    "level": alert.level.value,
                    "category": alert.category,
                    "message": alert.message,
                    "recommendation": alert.recommendation,
                    "source": alert.source,
                    "confidence": alert.confidence
                }
                for alert in alerts
            ],
            "enhanced_db_used": self.enhanced_db_available and bool(matched_name),
            "validation_method": "llm_powered" if matched_name and self.enhanced_db_available else "basic"
        }
    
    def validate_dosage_basic(self, dosage_text: str, drug_info: Dict) -> DosageValidation:
        """Validate dosage against standard ranges"""
        alerts = []
        
        if not dosage_text or dosage_text.lower() in ["not specified", "none", ""]:
            alerts.append(MedicalAlert(
                level=AlertLevel.WARNING,
                category="dosage_missing",
                message="Dosage not specified",
                recommendation="Specify appropriate dosage based on indication and patient factors",
                source="dosage_validator",
                confidence=0.9
            ))
            return DosageValidation(False, None, None, None, alerts)
        
        # Parse dosage
        dosage_pattern = r'(\d+(?:\.\d+)?)\s*(mg|ml|g|mcg|iu|units?|μg)'
        match = re.search(dosage_pattern, dosage_text.lower())
        
        if not match:
            alerts.append(MedicalAlert(
                level=AlertLevel.WARNING,
                category="dosage_format",
                message=f"Could not parse dosage format: '{dosage_text}'",
                recommendation="Use standard dosage format (e.g., '500 mg', '10 ml')",
                source="dosage_parser",
                confidence=0.8
            ))
            return DosageValidation(False, None, None, None, alerts)
        
        value = float(match.group(1))
        unit = match.group(2)
        
        # Convert to standard units (mg)
        conversion_factor = self.unit_conversions.get(unit, 1)
        standardized_value = value * conversion_factor
        
        # Check against standard ranges
        standard_dosages = drug_info.get("standard_dosages", {})
        adult_range = standard_dosages.get("adult", {})
        
        if adult_range:
            min_dose = adult_range.get("min", 0)
            max_dose = adult_range.get("max", float('inf'))
            standard_unit = adult_range.get("unit", "mg")
            
            if standardized_value < min_dose:
                alerts.append(MedicalAlert(
                    level=AlertLevel.WARNING,
                    category="dosage_low",
                    message=f"Dosage {value} {unit} is below typical range ({min_dose}-{max_dose} {standard_unit})",
                    recommendation="Consider if low dose is intentional or if adjustment needed",
                    source="dosage_validator",
                    confidence=0.8
                ))
            
            elif standardized_value > max_dose:
                alerts.append(MedicalAlert(
                    level=AlertLevel.CRITICAL,
                    category="dosage_high",
                    message=f"Dosage {value} {unit} exceeds typical range ({min_dose}-{max_dose} {standard_unit})",
                    recommendation="Verify dosage and consider potential for overdose",
                    source="dosage_validator",
                    confidence=0.9
                ))
        
        # Check maximum daily dose
        max_daily = drug_info.get("max_daily_dose")
        if max_daily and standardized_value > max_daily:
            alerts.append(MedicalAlert(
                level=AlertLevel.CRITICAL,
                category="max_daily_exceeded",
                message=f"Single dose {value} {unit} exceeds maximum daily dose ({max_daily} mg)",
                recommendation="Reduce dose or verify calculation",
                source="dosage_validator",
                confidence=0.9
            ))
        
        return DosageValidation(
            is_valid=len([a for a in alerts if a.level == AlertLevel.CRITICAL]) == 0,
            parsed_value=value,
            parsed_unit=unit,
            standard_range=(adult_range.get("min"), adult_range.get("max")) if adult_range else None,
            alerts=alerts
        )
    
    def validate_frequency(self, frequency_text: str, drug_info: Dict) -> List[MedicalAlert]:
        """Validate dosing frequency"""
        alerts = []
        
        if not frequency_text or frequency_text.lower() in ["not specified", "none", ""]:
            alerts.append(MedicalAlert(
                level=AlertLevel.WARNING,
                category="frequency_missing",
                message="Dosing frequency not specified",
                recommendation="Specify appropriate dosing frequency",
                source="frequency_validator",
                confidence=0.9
            ))
            return alerts
        
        # Parse frequency
        freq_lower = frequency_text.lower()
        times_per_day = None
        
        for pattern, daily_freq in self.frequency_conversions.items():
            if pattern in freq_lower:
                times_per_day = daily_freq
                break
        
        if times_per_day is None:
            alerts.append(MedicalAlert(
                level=AlertLevel.WARNING,
                category="frequency_format",
                message=f"Could not parse frequency format: '{frequency_text}'",
                recommendation="Use standard frequency format (e.g., 'twice daily', 'every 8 hours')",
                source="frequency_parser",
                confidence=0.8
            ))
            return alerts
        
        # Validate against typical frequencies for this drug
        # This would be expanded with drug-specific frequency guidelines
        if times_per_day > 6:
            alerts.append(MedicalAlert(
                level=AlertLevel.WARNING,
                category="frequency_high",
                message=f"Frequency '{frequency_text}' ({times_per_day}x/day) is unusually high",
                recommendation="Verify dosing frequency is appropriate",
                source="frequency_validator",
                confidence=0.7
            ))
        
        return alerts
    
    def check_drug_interactions(self, drug_name: str, all_medications: List[Dict]) -> List[MedicalAlert]:
        """Check for drug-drug interactions"""
        alerts = []
        
        drug_interactions = self.interaction_matrix.get(drug_name, {})
        
        for other_med in all_medications:
            other_drug = other_med.get("name_candidate", "").lower()
            if other_drug != drug_name and other_drug in drug_interactions:
                interaction = drug_interactions[other_drug]
                
                severity = interaction["severity"]
                alert_level = AlertLevel.CRITICAL if severity == "major" else AlertLevel.WARNING
                
                alerts.append(MedicalAlert(
                    level=alert_level,
                    category="drug_interaction",
                    message=f"Interaction between {drug_name} and {other_drug}: {interaction['effect']}",
                    recommendation=f"Monitor patient closely" if severity == "moderate" else "Consider alternative therapy",
                    source="interaction_database",
                    confidence=0.9
                ))
        
        return alerts
    
    def validate_for_patient(self, drug_info: Dict, patient_info: Dict) -> List[MedicalAlert]:
        """Validate medication for specific patient"""
        alerts = []
        
        # Check contraindications
        contraindications = drug_info.get("contraindications", [])
        patient_conditions = patient_info.get("conditions", [])
        
        for contraindication in contraindications:
            for condition in patient_conditions:
                if contraindication.lower() in condition.lower():
                    alerts.append(MedicalAlert(
                        level=AlertLevel.CRITICAL,
                        category="contraindication",
                        message=f"Contraindication: {contraindication} (patient has {condition})",
                        recommendation="Consider alternative therapy",
                        source="contraindication_checker",
                        confidence=0.9
                    ))
        
        # Check pregnancy
        pregnancy_category = drug_info.get("pregnancy_category", "")
        is_pregnant = patient_info.get("is_pregnant", False)
        
        if is_pregnant and pregnancy_category in ["D", "X"]:
            alerts.append(MedicalAlert(
                level=AlertLevel.CRITICAL,
                category="pregnancy_risk",
                message=f"Pregnancy Category {pregnancy_category}: {self.pregnancy_categories.get(pregnancy_category)}",
                recommendation="Avoid use in pregnancy" if pregnancy_category == "X" else "Use only if benefits outweigh risks",
                source="pregnancy_database",
                confidence=1.0
            ))
        
        # Age-specific checks
        age = patient_info.get("age")
        if age is not None:
            if age < 18:
                alerts.append(MedicalAlert(
                    level=AlertLevel.INFO,
                    category="pediatric_use",
                    message="Pediatric patient - verify pediatric dosing",
                    recommendation="Confirm dose is appropriate for pediatric use",
                    source="age_validator",
                    confidence=0.8
                ))
            elif age > 65:
                alerts.append(MedicalAlert(
                    level=AlertLevel.INFO,
                    category="geriatric_use",
                    message="Geriatric patient - consider dose adjustment",
                    recommendation="Consider reduced dosing for elderly patients",
                    source="age_validator",
                    confidence=0.8
                ))
        
        return alerts
    
    def calculate_safety_score(self, alerts: List[MedicalAlert]) -> float:
        """Calculate overall safety score (0-100)"""
        if not alerts:
            return 100.0
        
        # Weight alerts by severity
        severity_weights = {
            AlertLevel.INFO: 0.1,
            AlertLevel.WARNING: 0.5,
            AlertLevel.CRITICAL: 1.0
        }
        
        total_weight = sum(severity_weights[alert.level] * alert.confidence for alert in alerts)
        max_possible_weight = len(alerts) * 1.0  # All critical alerts
        
        # Invert score (higher weight = lower safety)
        safety_score = max(0, 100 - (total_weight / max_possible_weight * 100))
        
        return round(safety_score, 1)
    
    def generate_safety_report(self, all_medications: List[Dict], patient_info: Dict = None) -> Dict[str, Any]:
        """Generate comprehensive safety report for all medications"""
        
        medication_validations = []
        overall_alerts = []
        
        # Validate each medication
        for med in all_medications:
            validation = self.validate_medication(med, all_medications, patient_info)
            medication_validations.append(validation)
            overall_alerts.extend(validation["alerts"])
        
        # Calculate overall safety metrics
        critical_alerts = [a for a in overall_alerts if a["level"] == "critical"]
        warning_alerts = [a for a in overall_alerts if a["level"] == "warning"]
        
        overall_safety_score = self.calculate_safety_score([
            MedicalAlert(
                level=AlertLevel.CRITICAL if a["level"] == "critical" else 
                      AlertLevel.WARNING if a["level"] == "warning" else AlertLevel.INFO,
                category=a["category"],
                message=a["message"],
                recommendation=a["recommendation"],
                source=a["source"],
                confidence=a["confidence"]
            )
            for a in overall_alerts
        ])
        
        return {
            "overall_safety_score": overall_safety_score,
            "total_medications": len(all_medications),
            "medications_with_issues": len([m for m in medication_validations if m["alerts"]]),
            "critical_alerts_count": len(critical_alerts),
            "warning_alerts_count": len(warning_alerts),
            "medication_validations": medication_validations,
            "summary_recommendations": self._generate_summary_recommendations(overall_alerts),
            "monitoring_requirements": list(set(
                req for med in medication_validations 
                for req in med.get("monitoring_required", [])
            )),
            "validation_timestamp": time.time()
        }
    
    def _generate_summary_recommendations(self, alerts: List[Dict]) -> List[str]:
        """Generate summary recommendations based on alerts"""
        recommendations = []
        
        critical_count = len([a for a in alerts if a["level"] == "critical"])
        warning_count = len([a for a in alerts if a["level"] == "warning"])
        
        if critical_count > 0:
            recommendations.append(f"{critical_count} critical safety issue(s) require immediate attention")
        
        if warning_count > 0:
            recommendations.append(f"{warning_count} warning(s) should be reviewed")
        
        # Category-specific recommendations
        categories = {}
        for alert in alerts:
            category = alert["category"]
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        
        if categories.get("drug_interaction", 0) > 0:
            recommendations.append("Review drug interactions and consider monitoring")
        
        if categories.get("dosage_high", 0) > 0:
            recommendations.append("Verify high dosages and consider dose reduction")
        
        if categories.get("contraindication", 0) > 0:
            recommendations.append("Address contraindications - consider alternative therapy")
        
        return recommendations
    
    def generate_safety_report_from_validations(self, validation_results: List[Dict[str, Any]], medications: List[Dict]) -> Dict[str, Any]:
        """Generate safety report from existing validation results to avoid double processing"""
        
        # Collect all alerts from validation results
        all_alerts = []
        for validation in validation_results:
            all_alerts.extend(validation.get("alerts", []))
        
        # Count alerts by level
        critical_alerts = [a for a in all_alerts if a["level"] == "critical"]
        warning_alerts = [a for a in all_alerts if a["level"] == "warning"]
        info_alerts = [a for a in all_alerts if a["level"] == "info"]
        
        # Calculate overall safety score using the same logic
        overall_safety_score = self.calculate_safety_score([
            MedicalAlert(
                level=AlertLevel.CRITICAL if a["level"] == "critical" else 
                      AlertLevel.WARNING if a["level"] == "warning" else AlertLevel.INFO,
                category=a["category"],
                message=a["message"],
                recommendation=a["recommendation"],
                source=a["source"],
                confidence=a["confidence"]
            )
            for a in all_alerts
        ])
        
        return {
            "overall_safety_score": overall_safety_score,
            "total_medications": len(medications),
            "medications_with_issues": len([v for v in validation_results if v["alerts"]]),
            "critical_alerts_count": len(critical_alerts),
            "warning_alerts_count": len(warning_alerts),
            "info_alerts_count": len(info_alerts),
            "medication_validations": validation_results,
            "summary_recommendations": self._generate_summary_recommendations(all_alerts),
            "monitoring_requirements": list(set(
                req for validation in validation_results 
                for req in validation.get("monitoring_required", [])
            )),
            "validation_timestamp": time.time()
        }
    
    def batch_validate_medications(self, medications: List[Dict], patient_info: Optional[Dict] = None, 
                                  validation_level: ValidationLevel = ValidationLevel.STANDARD) -> List[Dict[str, Any]]:
        """
        Batch validate multiple medications for much faster processing
        """
        if not medications:
            return []
        
        results = []
        
        # Use batch LLM validation if enhanced DB is available
        if self.enhanced_db_available and len(medications) > 1:
            try:
                # Get batch validation results
                batch_alerts = self._llm_batch_validate_medications(medications, patient_info)
                print(f"✅ Batch validated {len(medications)} medications in single LLM call")
                
                # Process each medication with its batch alerts
                for medication in medications:
                    original_name = medication.get("name_candidate", "")
                    matched_name = medication.get("matched_name", "")
                    drug_name = matched_name or original_name
                    
                    # Get alerts from batch processing
                    alerts = batch_alerts.get(matched_name, [])
                    
                    # Add critical drug monitoring alerts
                    if matched_name:
                        critical_info = self.critical_drugs.get(matched_name.lower())
                        if critical_info:
                            alerts.append(MedicalAlert(
                                level=AlertLevel.WARNING,
                                category="monitoring",
                                message=f"High-risk medication: {matched_name} requires monitoring",
                                recommendation=f"Monitor: {', '.join(critical_info['monitoring'])}",
                                source="critical_drug_list",
                                confidence=0.9
                            ))
                    
                    # Calculate safety score
                    critical_alerts = sum(1 for alert in alerts if alert.level == AlertLevel.CRITICAL)
                    warning_alerts = sum(1 for alert in alerts if alert.level == AlertLevel.WARNING)
                    info_alerts = sum(1 for alert in alerts if alert.level == AlertLevel.INFO)
                    
                    safety_score = max(0, 100 - (critical_alerts * 40) - (warning_alerts * 15) - (info_alerts * 5))
                    
                    result = {
                        "medication_name": matched_name or original_name,
                        "original_name": original_name,
                        "validation_status": "validated" if not any(a.level == AlertLevel.CRITICAL for a in alerts) else "critical_issues",
                        "safety_score": safety_score,
                        "alerts": [
                            {
                                "level": alert.level.value,
                                "category": alert.category,
                                "message": alert.message,
                                "recommendation": alert.recommendation,
                                "source": alert.source,
                                "confidence": alert.confidence
                            }
                            for alert in alerts
                        ],
                        "enhanced_db_used": self.enhanced_db_available and bool(matched_name),
                        "validation_method": "llm_batch_powered"
                    }
                    results.append(result)
                
                return results
                
            except Exception as e:
                print(f"Batch validation failed, falling back to individual: {e}")
        
        # Fallback to individual validation
        for medication in medications:
            result = self.validate_medication(medication, medications, patient_info, validation_level)
            results.append(result)
        
        return results