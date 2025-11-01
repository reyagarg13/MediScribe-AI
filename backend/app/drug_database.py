"""
Enhanced Drug Database System using RxNorm API + OpenFDA
Provides medical-grade drug information with local caching
"""
import requests
import sqlite3
import json
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DrugInfo:
    """Standardized drug information structure"""
    rxcui: str  # RxNorm unique identifier
    generic_name: str
    brand_names: List[str]
    drug_class: str
    dosage_forms: List[str]
    strengths: List[str]
    routes: List[str]
    interactions: List[Dict[str, Any]]
    contraindications: List[str]
    warnings: List[str]
    source: str
    last_updated: float


class DrugDatabase:
    """
    Medical-grade drug database using free NIH/FDA APIs
    - RxNorm API for standardized drug names and relationships
    - OpenFDA API for warnings, dosages, and detailed info
    - Local SQLite caching for performance and reliability
    """
    
    def __init__(self, db_path: str = "drug_cache.db"):
        self.db_path = db_path
        self.rxnorm_base = "https://rxnav.nlm.nih.gov/REST"
        self.openfda_base = "https://api.fda.gov/drug/label.json"
        self.cache_expiry = 7 * 24 * 3600  # 7 days
        self.init_database()
    
    def init_database(self):
        """Initialize local SQLite cache database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create drug cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drug_cache (
                rxcui TEXT PRIMARY KEY,
                generic_name TEXT NOT NULL,
                brand_names TEXT,  -- JSON array
                drug_class TEXT,
                dosage_forms TEXT,  -- JSON array
                strengths TEXT,     -- JSON array
                routes TEXT,        -- JSON array
                interactions TEXT,  -- JSON array
                contraindications TEXT,  -- JSON array
                warnings TEXT,      -- JSON array
                source TEXT,
                last_updated REAL,
                UNIQUE(rxcui)
            )
        """)
        
        # Create search index
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_generic_name 
            ON drug_cache(generic_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_brand_names 
            ON drug_cache(brand_names)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Initialized drug database at {self.db_path}")
    
    def search_drug(self, drug_name: str) -> Optional[DrugInfo]:
        """
        Search for drug information by name
        1. Check local cache first
        2. If not found or expired, fetch from APIs
        3. Cache the result
        """
        drug_name = drug_name.strip().lower()
        
        # First check cache
        cached_info = self._get_from_cache(drug_name)
        if cached_info and not self._is_cache_expired(cached_info.last_updated):
            logger.info(f"Cache hit for drug: {drug_name}")
            return cached_info
        
        # Not in cache or expired - fetch from APIs
        logger.info(f"Fetching drug info from APIs: {drug_name}")
        try:
            drug_info = self._fetch_from_apis(drug_name)
            if drug_info:
                self._save_to_cache(drug_info)
                return drug_info
        except Exception as e:
            logger.error(f"API fetch failed for {drug_name}: {e}")
            # Return stale cache if available
            if cached_info:
                logger.info(f"Returning stale cache for {drug_name}")
                return cached_info
        
        return None
    
    def _get_from_cache(self, drug_name: str) -> Optional[DrugInfo]:
        """Get drug info from local cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Search by generic name or brand names
        cursor.execute("""
            SELECT * FROM drug_cache 
            WHERE generic_name = ? 
            OR brand_names LIKE ?
        """, (drug_name, f'%"{drug_name}"%'))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return DrugInfo(
            rxcui=row[0],
            generic_name=row[1],
            brand_names=json.loads(row[2]) if row[2] else [],
            drug_class=row[3] or "",
            dosage_forms=json.loads(row[4]) if row[4] else [],
            strengths=json.loads(row[5]) if row[5] else [],
            routes=json.loads(row[6]) if row[6] else [],
            interactions=json.loads(row[7]) if row[7] else [],
            contraindications=json.loads(row[8]) if row[8] else [],
            warnings=json.loads(row[9]) if row[9] else [],
            source=row[10] or "",
            last_updated=row[11] or 0
        )
    
    def _is_cache_expired(self, last_updated: float) -> bool:
        """Check if cache entry is expired"""
        return (time.time() - last_updated) > self.cache_expiry
    
    def _fetch_from_apis(self, drug_name: str) -> Optional[DrugInfo]:
        """Fetch drug information from RxNorm + OpenFDA APIs"""
        
        # Step 1: Get RxCUI from RxNorm
        rxcui = self._get_rxcui(drug_name)
        if not rxcui:
            return None
        
        # Step 2: Get drug details from RxNorm
        rxnorm_data = self._get_rxnorm_details(rxcui)
        if not rxnorm_data:
            return None
        
        # Step 3: Enrich with OpenFDA data
        fda_data = self._get_openfda_details(rxnorm_data.get('generic_name', drug_name))
        
        # Step 3.5: If OpenFDA failed, try with original drug name
        if not fda_data.get('drug_class') and drug_name != rxnorm_data.get('generic_name', ''):
            fda_data = self._get_openfda_details(drug_name)
        
        # Step 4: Combine and structure the data
        drug_info = DrugInfo(
            rxcui=rxcui,
            generic_name=rxnorm_data.get('generic_name', drug_name),
            brand_names=rxnorm_data.get('brand_names', []),
            drug_class=fda_data.get('drug_class', ''),
            dosage_forms=rxnorm_data.get('dosage_forms', []),
            strengths=rxnorm_data.get('strengths', []),
            routes=fda_data.get('routes', []),
            interactions=fda_data.get('interactions', []),
            contraindications=fda_data.get('contraindications', []),
            warnings=fda_data.get('warnings', []),
            source="rxnorm+openfda",
            last_updated=time.time()
        )
        
        return drug_info
    
    def _get_rxcui(self, drug_name: str) -> Optional[str]:
        """Get RxNorm Concept Unique Identifier (RXCUI) for a drug name"""
        try:
            url = f"{self.rxnorm_base}/rxcui.json"
            params = {"name": drug_name}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            id_group = data.get("idGroup", {})
            rxnorm_ids = id_group.get("rxnormId", [])
            
            if rxnorm_ids:
                return rxnorm_ids[0]  # Return first RXCUI
            
        except Exception as e:
            logger.error(f"RxNorm RXCUI lookup failed for {drug_name}: {e}")
        
        return None
    
    def _get_rxnorm_details(self, rxcui: str) -> Optional[Dict[str, Any]]:
        """Get detailed drug information from RxNorm"""
        try:
            # Get drug properties
            url = f"{self.rxnorm_base}/rxcui/{rxcui}/properties.json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            properties = data.get("properties", {})
            
            if not properties:
                return None
            
            # Get related drugs (brand names, etc.)
            related_url = f"{self.rxnorm_base}/rxcui/{rxcui}/related.json"
            related_response = requests.get(related_url, timeout=10)
            related_data = {}
            
            if related_response.status_code == 200:
                related_data = related_response.json()
            
            # Extract brand names
            brand_names = []
            related_group = related_data.get("relatedGroup", {})
            concept_group = related_group.get("conceptGroup", [])
            
            for group in concept_group:
                if group.get("tty") == "BN":  # Brand Name
                    concept_properties = group.get("conceptProperties", [])
                    for concept in concept_properties:
                        name = concept.get("name", "").strip()
                        if name and name not in brand_names:
                            brand_names.append(name)
            
            # Get dosage forms
            dosage_forms = []
            for group in concept_group:
                if group.get("tty") in ["DF", "DFG"]:  # Dosage Form
                    concept_properties = group.get("conceptProperties", [])
                    for concept in concept_properties:
                        form = concept.get("name", "").strip()
                        if form and form not in dosage_forms:
                            dosage_forms.append(form)
            
            # Get strengths
            strengths = []
            for group in concept_group:
                if group.get("tty") in ["SCD", "SBD"]:  # Semantic Clinical Drug
                    concept_properties = group.get("conceptProperties", [])
                    for concept in concept_properties:
                        name = concept.get("name", "")
                        # Extract strength information using regex
                        strength_match = re.search(r'(\d+(?:\.\d+)?)\s*(mg|ml|g|mcg|iu|%)', name, re.I)
                        if strength_match:
                            strength = strength_match.group(0)
                            if strength not in strengths:
                                strengths.append(strength)
            
            return {
                "generic_name": properties.get("name", "").strip(),
                "brand_names": brand_names,
                "dosage_forms": dosage_forms,
                "strengths": strengths,
                "synonym": properties.get("synonym", "")
            }
            
        except Exception as e:
            logger.error(f"RxNorm details lookup failed for RXCUI {rxcui}: {e}")
        
        return None
    
    def _get_openfda_details(self, generic_name: str) -> Dict[str, Any]:
        """Get detailed drug information from OpenFDA"""
        fda_data = {
            "drug_class": "",
            "routes": [],
            "interactions": [],
            "contraindications": [],
            "warnings": []
        }
        
        # Try multiple search strategies for OpenFDA (based on working API tests)
        search_terms = [
            generic_name,  # Simple text search (works best)
            f'active_ingredient:"{generic_name}"',  # This works!
            f'brand_name:"{generic_name}"',
            f'substance_name:"{generic_name}"',
            f'openfda.generic_name:"{generic_name}"',  # Try with openfda prefix
        ]
        
        for search_term in search_terms:
            try:
                url = self.openfda_base
                params = {
                    "search": search_term,
                    "limit": 1
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    if results:
                        logger.info(f"OpenFDA success for {generic_name} with search: {search_term}")
                        return self._parse_openfda_result(results[0])
                else:
                    logger.debug(f"OpenFDA {response.status_code} for {generic_name} with search: {search_term}")
                    
            except Exception as e:
                logger.debug(f"OpenFDA search failed for {generic_name} with {search_term}: {e}")
        
        logger.warning(f"All OpenFDA searches failed for {generic_name}")
        return fda_data
    
    def _parse_openfda_result(self, result: Dict) -> Dict[str, Any]:
        """Parse OpenFDA API result"""
        fda_data = {
            "drug_class": "",
            "routes": [],
            "interactions": [],
            "contraindications": [],
            "warnings": []
        }
        
        try:
            # Extract drug class
            openfda = result.get("openfda", {})
            pharm_class = openfda.get("pharm_class_epc", [])
            if not pharm_class:
                pharm_class = openfda.get("pharm_class_cs", [])
            if not pharm_class:
                pharm_class = openfda.get("pharm_class_moa", [])
            
            if pharm_class:
                fda_data["drug_class"] = pharm_class[0]
            
            # Extract routes of administration
            routes = openfda.get("route", [])
            fda_data["routes"] = routes
            
            # Extract warnings and contraindications
            warnings = result.get("warnings", [])
            if not warnings:
                warnings = result.get("boxed_warning", [])
            if not warnings:
                warnings = result.get("warnings_and_cautions", [])
            
            contraindications = result.get("contraindications", [])
            if not contraindications:
                contraindications = result.get("contraindications_table", [])
            
            fda_data["warnings"] = warnings[:3] if warnings else []
            fda_data["contraindications"] = contraindications[:3] if contraindications else []
            
            # Extract drug interactions
            drug_interactions = result.get("drug_interactions", [])
            if not drug_interactions:
                drug_interactions = result.get("drug_and_or_laboratory_test_interactions", [])
            
            if drug_interactions:
                interactions = []
                for interaction_text in drug_interactions[:3]:
                    if isinstance(interaction_text, str) and len(interaction_text.strip()) > 10:
                        interactions.append({
                            "description": interaction_text[:200] + "..." if len(interaction_text) > 200 else interaction_text,
                            "severity": "unknown"
                        })
                fda_data["interactions"] = interactions
            
        except Exception as e:
            logger.error(f"OpenFDA parsing failed: {e}")
        
        return fda_data
    
    def _save_to_cache(self, drug_info: DrugInfo):
        """Save drug information to local cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO drug_cache 
                (rxcui, generic_name, brand_names, drug_class, dosage_forms, 
                 strengths, routes, interactions, contraindications, warnings, 
                 source, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                drug_info.rxcui,
                drug_info.generic_name,
                json.dumps(drug_info.brand_names),
                drug_info.drug_class,
                json.dumps(drug_info.dosage_forms),
                json.dumps(drug_info.strengths),
                json.dumps(drug_info.routes),
                json.dumps(drug_info.interactions),
                json.dumps(drug_info.contraindications),
                json.dumps(drug_info.warnings),
                drug_info.source,
                drug_info.last_updated
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Cached drug info: {drug_info.generic_name}")
            
        except Exception as e:
            logger.error(f"Failed to cache drug info: {e}")
    
    def fuzzy_search_drug(self, drug_name: str, threshold: int = 60) -> List[Tuple[str, int]]:
        """
        Fuzzy search for drug names using local cache + fallback to APIs
        Returns list of (drug_name, confidence_score) tuples
        """
        drug_name_clean = drug_name.strip().lower()
        
        # First try exact search
        exact_match = self.search_drug(drug_name_clean)
        if exact_match:
            return [(exact_match.generic_name, 100)]
        
        # Fuzzy search in cache
        matches = []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT generic_name, brand_names FROM drug_cache")
        rows = cursor.fetchall()
        conn.close()
        
        for row in rows:
            generic_name = row[0]
            brand_names = json.loads(row[1]) if row[1] else []
            
            # Check generic name similarity
            score = self._calculate_similarity(drug_name_clean, generic_name.lower())
            if score >= threshold:
                matches.append((generic_name, score))
            
            # Check brand names
            for brand in brand_names:
                score = self._calculate_similarity(drug_name_clean, brand.lower())
                if score >= threshold:
                    matches.append((brand, score))
        
        # Sort by confidence score
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:5]  # Return top 5 matches
    
    def _calculate_similarity(self, str1: str, str2: str) -> int:
        """Calculate similarity score between two strings (0-100)"""
        if str1 == str2:
            return 100
        
        if str1 in str2 or str2 in str1:
            return max(80, int((min(len(str1), len(str2)) / max(len(str1), len(str2))) * 100))
        
        # Simple character-based similarity
        common_chars = set(str1) & set(str2)
        total_chars = set(str1) | set(str2)
        
        if not total_chars:
            return 0
        
        return int((len(common_chars) / len(total_chars)) * 100)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM drug_cache")
        total_drugs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM drug_cache WHERE last_updated > ?", 
                      (time.time() - self.cache_expiry,))
        fresh_drugs = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_cached_drugs": total_drugs,
            "fresh_entries": fresh_drugs,
            "stale_entries": total_drugs - fresh_drugs,
            "cache_hit_rate": "Unknown - need to track",
            "database_path": self.db_path
        }


# Global instance
drug_db = DrugDatabase()


def search_drug_enhanced(drug_name: str) -> Optional[Dict[str, Any]]:
    """
    Enhanced drug search function that replaces the old hardcoded lookup
    Returns comprehensive drug information or None
    """
    if not drug_name or len(drug_name.strip()) < 2:
        return None
    
    drug_info = drug_db.search_drug(drug_name)
    
    if not drug_info:
        return None
    
    return {
        "match": drug_info.generic_name,
        "score": 100,  # Exact match from database
        "candidates": drug_info.brand_names,
        "drug_class": drug_info.drug_class,
        "dosage_forms": drug_info.dosage_forms,
        "strengths": drug_info.strengths,
        "routes": drug_info.routes,
        "interactions": drug_info.interactions,
        "contraindications": drug_info.contraindications,
        "warnings": drug_info.warnings,
        "source": drug_info.source
    }


def _llm_batch_drug_normalization(drug_names: List[str]) -> Dict[str, str]:
    """
    Batch normalize multiple drug names in a single LLM call for speed
    """
    if not drug_names:
        return {}
        
    try:
        # Import gemini here to avoid circular imports
        from . import gemini
        if not hasattr(gemini, 'analyze_with_gemini') or not gemini.available():
            return {}
        
        # Create batch normalization prompt
        drugs_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(drug_names)])
        
        batch_prompt = f"""
You are a pharmaceutical expert. Normalize these drug names to US FDA-approved generic names.

DRUG NAMES TO NORMALIZE:
{drugs_list}

RULES:
1. Remove formulation prefixes: TAB, SYP, CAP, INJ, etc.
2. Remove dosage information: mg, ml, numbers
3. Identify core drug from brand/combination names
4. Convert to US generic equivalent

COMMON PATTERNS:
- "PanD" = Pantoprazole + Domperidone → "pantoprazole"
- "Calpol" → "acetaminophen"
- "Brufen" → "ibuprofen"
- "Levolin" → "albuterol"
- "Meftal-P" → "mefenamic acid"

OUTPUT FORMAT - Return ONLY this format:
1. [generic_name]
2. [generic_name]
3. [generic_name]

Example:
1. pantoprazole
2. acetaminophen
3. ibuprofen

Normalize the drugs:"""
        
        result = gemini.analyze_with_gemini(batch_prompt)
        
        if result:
            # Parse the results
            lines = result.strip().split('\n')
            normalized_dict = {}
            
            for i, line in enumerate(lines):
                if i < len(drug_names):
                    # Extract normalized name from "1. drugname" format
                    normalized = re.sub(r'^\d+\.\s*', '', line.strip()).lower()
                    if normalized and normalized != "unknown" and len(normalized) > 2:
                        normalized_dict[drug_names[i]] = normalized
                        logger.info(f"Batch normalized '{drug_names[i]}' → '{normalized}'")
            
            return normalized_dict
            
    except Exception as e:
        logger.debug(f"Batch drug normalization failed: {e}")
    
    return {}

def _llm_drug_name_normalization(drug_name: str) -> Optional[str]:
    """
    Use LLM to normalize international/brand drug names to US generic names
    """
    try:
        # Import gemini here to avoid circular imports
        from . import gemini
        if not hasattr(gemini, 'analyze_with_gemini') or not gemini.available():
            return None
        
        # Enhanced prompt for drug name normalization
        normalization_prompt = f"""
You are a pharmaceutical expert. Convert this drug name to its US FDA-approved generic name.

Drug name: "{drug_name}"

ANALYSIS RULES:
1. Remove formulation prefixes: TAB, SYP, CAP, INJ, etc.
2. Remove dosage information: mg, ml, numbers
3. Identify the core drug name from brand/combination names
4. Convert to US generic equivalent

COMMON PATTERNS:
- "PanD" = Pantoprazole + Domperidone → "pantoprazole" (main active ingredient)
- "Tab. PanD40mg" → "pantoprazole"
- "SYP CALPOL" → "acetaminophen" 
- "Brufen" → "ibuprofen"
- "Levolin" → "albuterol"
- "Meftal-P" → "mefenamic acid"
- "Disprin" → "aspirin"

COMBINATION DRUGS - Return the PRIMARY active ingredient:
- Acid reducers with "Pan": Usually pantoprazole
- Pain relievers with "P": Usually paracetamol/acetaminophen  
- Anti-spasmodics: Check for dicyclomine, hyoscine

EXAMPLES:
- "Tab. PanD40mg" → "pantoprazole"
- "SYP CALPOL (250/5)" → "acetaminophen"
- "TAB BRUFEN 400mg" → "ibuprofen"
- "Cap. Omez DSR" → "omeprazole"

Return ONLY the primary generic drug name. If uncertain, analyze the brand pattern.

Generic name:"""
        
        result = gemini.analyze_with_gemini(normalization_prompt)
        
        if result:
            # Clean up the response
            normalized = result.strip().lower()
            # Remove common response prefixes
            normalized = normalized.replace("generic name:", "").strip()
            normalized = normalized.replace("the generic name is", "").strip()
            normalized = normalized.replace("answer:", "").strip()
            
            # Validate it's a reasonable drug name (not "unknown" or too long)
            if (normalized and 
                normalized != "unknown" and 
                len(normalized) > 2 and 
                len(normalized) < 50 and
                normalized.replace(" ", "").isalpha()):
                logger.info(f"LLM normalized '{drug_name}' → '{normalized}'")
                return normalized
        
    except Exception as e:
        logger.debug(f"LLM drug normalization failed for '{drug_name}': {e}")
    
    return None


def fuzzy_match_drug_enhanced(drug_name: str) -> Dict[str, Any]:
    """
    Enhanced fuzzy matching with LLM-powered international drug name processing
    """
    if not drug_name or len(drug_name.strip()) < 2:
        return {"match": None, "score": 0, "candidates": [], "normalization": None}
    
    original_name = drug_name.strip()
    normalization_info = None
    
    # First try direct search
    matches = drug_db.fuzzy_search_drug(original_name)
    
    # If no good matches found, try LLM normalization
    if not matches or (matches and matches[0][1] < 70):
        logger.info(f"Low confidence match for '{original_name}', trying LLM normalization...")
        
        normalized_name = _llm_drug_name_normalization(original_name)
        if normalized_name and normalized_name != original_name.lower():
            # Try search with normalized name
            normalized_matches = drug_db.fuzzy_search_drug(normalized_name)
            
            # Always store normalization info if LLM provided a result
            normalization_info = {
                "original_name": original_name,
                "normalized_name": normalized_name,
                "method": "llm_powered",
                "reasoning": f"AI recognized '{original_name}' as brand/international name for '{normalized_name}'"
            }
            
            # Use normalized results if they're better, or if we had no matches at all
            if normalized_matches and (not matches or normalized_matches[0][1] > matches[0][1]):
                logger.info(f"LLM normalization improved match: '{original_name}' → '{normalized_name}' → '{normalized_matches[0][0]}'")
                matches = normalized_matches
            elif normalized_matches and not matches:
                # No original matches, use normalized matches
                logger.info(f"LLM normalization found match: '{original_name}' → '{normalized_name}' → '{normalized_matches[0][0]}'")
                matches = normalized_matches
            elif normalized_matches:
                # Keep original matches but still show normalization info
                logger.info(f"LLM normalized '{original_name}' → '{normalized_name}', but keeping original match")
    
    if not matches:
        return {"match": None, "score": 0, "candidates": [], "normalization": normalization_info}
    
    best_match = matches[0]
    all_candidates = [match[0] for match in matches]
    
    return {
        "match": best_match[0],
        "score": best_match[1],
        "candidates": all_candidates,
        "normalization": normalization_info
    }


def batch_fuzzy_match_drugs_enhanced(drug_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Batch process multiple drug names for much faster processing
    """
    if not drug_names:
        return {}
    
    results = {}
    
    # Skip individual lookups, go straight to batch normalization for speed
    print(f"🚀 Batch processing {len(drug_names)} drugs without individual checks")
    needs_normalization = [name for name in drug_names if name and len(name.strip()) >= 2]
    
    # Batch normalize all drugs in one shot
    if needs_normalization:
        logger.info(f"Batch normalizing {len(needs_normalization)} drugs: {needs_normalization}")
        normalized_dict = _llm_batch_drug_normalization(needs_normalization)
        
        # Process all drugs with batch normalization results
        for original_name in needs_normalization:
            normalized_name = normalized_dict.get(original_name)
            normalization_info = None
            
            if normalized_name:
                # Use normalized name for search
                matches = drug_db.fuzzy_search_drug(normalized_name)
                normalization_info = {
                    "original_name": original_name,
                    "normalized_name": normalized_name,
                    "method": "llm_powered_batch",
                    "reasoning": f"AI recognized '{original_name}' as brand/international name for '{normalized_name}'"
                }
            else:
                # Fallback to original name
                matches = drug_db.fuzzy_search_drug(original_name)
            
            if matches:
                best_match = matches[0]
                all_candidates = [match[0] for match in matches]
                results[original_name] = {
                    "match": best_match[0],
                    "score": best_match[1],
                    "candidates": all_candidates,
                    "normalization": normalization_info
                }
            else:
                results[original_name] = {
                    "match": None,
                    "score": 0,
                    "candidates": [],
                    "normalization": normalization_info
                }
    
    return results