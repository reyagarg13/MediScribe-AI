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
    """Comprehensive medical validation system"""
    
    def __init__(self):
        self.load_medical_databases()
    
    def load_medical_databases(self):
        """Load medical validation databases"""
        
        # Drug database with safety information
        self.drug_database = {
            "amoxicillin": {
                "generic_name": "amoxicillin",
                "category": "antibiotic",
                "standard_dosages": {
                    "adult": {"min": 250, "max": 1000, "unit": "mg"},
                    "pediatric": {"min": 20, "max": 50, "unit": "mg/kg"}
                },
                "max_daily_dose": 3000,  # mg
                "contraindications": [
                    "penicillin allergy",
                    "severe renal impairment"
                ],
                "drug_interactions": [
                    {"drug": "warfarin", "severity": "moderate", "effect": "increased bleeding risk"},
                    {"drug": "methotrexate", "severity": "major", "effect": "increased toxicity"}
                ],
                "pregnancy_category": "B",
                "common_side_effects": ["nausea", "diarrhea", "rash"],
                "black_box_warnings": []
            },
            
            "warfarin": {
                "generic_name": "warfarin",
                "category": "anticoagulant",
                "standard_dosages": {
                    "adult": {"min": 1, "max": 10, "unit": "mg"},
                },
                "max_daily_dose": 15,  # mg
                "contraindications": [
                    "active bleeding",
                    "pregnancy",
                    "severe liver disease"
                ],
                "drug_interactions": [
                    {"drug": "aspirin", "severity": "major", "effect": "increased bleeding risk"},
                    {"drug": "amoxicillin", "severity": "moderate", "effect": "increased INR"}
                ],
                "pregnancy_category": "X",
                "monitoring_required": ["INR", "PT"],
                "black_box_warnings": ["Can cause major or fatal bleeding"]
            },
            
            "insulin": {
                "generic_name": "insulin",
                "category": "antidiabetic",
                "standard_dosages": {
                    "adult": {"min": 0.1, "max": 2.0, "unit": "units/kg"},
                },
                "contraindications": [
                    "hypoglycemia"
                ],
                "drug_interactions": [
                    {"drug": "beta-blockers", "severity": "moderate", "effect": "masked hypoglycemia"}
                ],
                "pregnancy_category": "B",
                "monitoring_required": ["blood glucose", "HbA1c"],
                "black_box_warnings": []
            }
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
        
        # Drug interaction database
        self.interaction_matrix = self._build_interaction_matrix()
        
        # Pregnancy categories
        self.pregnancy_categories = {
            "A": "No risk in controlled studies",
            "B": "No risk in animal studies",
            "C": "Risk cannot be ruled out",
            "D": "Positive evidence of risk",
            "X": "Contraindicated in pregnancy"
        }
    
    def _build_interaction_matrix(self) -> Dict[str, Dict[str, Dict]]:
        """Build comprehensive drug interaction matrix"""
        interactions = {}
        
        # Extract interactions from drug database
        for drug_name, drug_info in self.drug_database.items():
            interactions[drug_name] = {}
            for interaction in drug_info.get("drug_interactions", []):
                interactions[drug_name][interaction["drug"]] = {
                    "severity": interaction["severity"],
                    "effect": interaction["effect"]
                }
        
        return interactions
    
    def validate_medication(
        self, 
        medication: Dict[str, Any], 
        all_medications: List[Dict[str, Any]] = None,
        patient_info: Dict[str, Any] = None,
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> Dict[str, Any]:
        """Comprehensive medication validation"""
        
        alerts = []
        drug_name = medication.get("name_candidate", "").lower()
        dosage = medication.get("dosage", "")
        frequency = medication.get("frequency", "")
        
        # Get drug information
        drug_info = self.drug_database.get(drug_name, {})
        
        # Validate drug exists
        if not drug_info:
            alerts.append(MedicalAlert(
                level=AlertLevel.WARNING,
                category="drug_recognition",
                message=f"Drug '{drug_name}' not found in database",
                recommendation="Verify drug name spelling and check against standard formularies",
                source="drug_database",
                confidence=0.8
            ))
        
        # Validate dosage
        dosage_validation = self.validate_dosage(dosage, drug_info)
        alerts.extend(dosage_validation.alerts)
        
        # Validate frequency
        frequency_validation = self.validate_frequency(frequency, drug_info)
        alerts.extend(frequency_validation)
        
        # Check drug interactions
        if all_medications and drug_info:
            interaction_alerts = self.check_drug_interactions(drug_name, all_medications)
            alerts.extend(interaction_alerts)
        
        # Patient-specific validations
        if patient_info and drug_info:
            patient_alerts = self.validate_for_patient(drug_info, patient_info)
            alerts.extend(patient_alerts)
        
        # Check for black box warnings
        if drug_info.get("black_box_warnings"):
            for warning in drug_info["black_box_warnings"]:
                alerts.append(MedicalAlert(
                    level=AlertLevel.CRITICAL,
                    category="black_box_warning",
                    message=f"Black Box Warning: {warning}",
                    recommendation="Review prescribing information and ensure appropriate monitoring",
                    source="fda_warnings",
                    confidence=1.0
                ))
        
        # Calculate overall safety score
        safety_score = self.calculate_safety_score(alerts)
        
        return {
            "medication_name": drug_name,
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
            "drug_info": drug_info,
            "dosage_validation": {
                "is_valid": dosage_validation.is_valid,
                "parsed_value": dosage_validation.parsed_value,
                "parsed_unit": dosage_validation.parsed_unit,
                "standard_range": dosage_validation.standard_range
            },
            "monitoring_required": drug_info.get("monitoring_required", []),
            "pregnancy_category": drug_info.get("pregnancy_category", "Unknown")
        }
    
    def validate_dosage(self, dosage_text: str, drug_info: Dict) -> DosageValidation:
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