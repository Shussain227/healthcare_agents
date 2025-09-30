"""
Personalized Treatment Recommendation Agent
===========================================
A modular agentic AI system for healthcare that recommends personalized cancer treatments.
Configurable for any hospital API.

Installation:
pip install streamlit pandas openai chromadb langchain langgraph requests python-dotenv

Usage:
streamlit run treatment_agent.py
"""

import streamlit as st
import pandas as pd
import json
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import os
from enum import Enum

# ============================================================================
# CONFIGURATION MODULE - Customize for different hospitals
# ============================================================================

@dataclass
class HospitalAPIConfig:
    """Configuration for hospital-specific API integration"""
    base_url: str
    api_key: str
    auth_type: str = "bearer"  # bearer, basic, apikey
    endpoints: Dict[str, str] = None
    
    def __post_init__(self):
        if self.endpoints is None:
            # Default FHIR-compliant endpoints
            self.endpoints = {
                "patient": "/Patient/{id}",
                "observation": "/Observation?patient={id}",
                "medication": "/MedicationStatement?patient={id}",
                "condition": "/Condition?patient={id}"
            }

class DataSource(Enum):
    """Available data sources"""
    EHR = "ehr"
    GENOMICS = "genomics"
    WEARABLES = "wearables"
    CLINICAL_TRIALS = "clinical_trials"

# ============================================================================
# DATA INGESTION MODULE
# ============================================================================

class HospitalAPIClient:
    """Generic client for hospital API integration"""
    
    def __init__(self, config: HospitalAPIConfig):
        self.config = config
        self.session = requests.Session()
        self._setup_auth()
    
    def _setup_auth(self):
        """Setup authentication headers"""
        if self.config.auth_type == "bearer":
            self.session.headers.update({
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            })
        elif self.config.auth_type == "apikey":
            self.session.headers.update({
                "X-API-Key": self.config.api_key,
                "Content-Type": "application/json"
            })
    
    def fetch_patient_data(self, patient_id: str, endpoint_type: str) -> Dict:
        """Fetch data from hospital API"""
        try:
            endpoint = self.config.endpoints.get(endpoint_type, "")
            url = f"{self.config.base_url}{endpoint.format(id=patient_id)}"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching {endpoint_type} data: {str(e)}")
            return {}

class DataIngestionEngine:
    """Handles data ingestion from multiple sources"""
    
    def __init__(self, api_client: HospitalAPIClient):
        self.api_client = api_client
    
    def fetch_ehr_data(self, patient_id: str) -> Dict:
        """Fetch Electronic Health Records"""
        patient = self.api_client.fetch_patient_data(patient_id, "patient")
        observations = self.api_client.fetch_patient_data(patient_id, "observation")
        medications = self.api_client.fetch_patient_data(patient_id, "medication")
        conditions = self.api_client.fetch_patient_data(patient_id, "condition")
        
        return {
            "patient": patient,
            "observations": observations,
            "medications": medications,
            "conditions": conditions
        }
    
    def parse_genomic_data(self, vcf_data: str) -> Dict:
        """Parse genomic data (VCF format)"""
        # Simplified parser - in production, use cyvcf2
        genomic_variants = []
        
        for line in vcf_data.split('\n'):
            if line.startswith('#') or not line.strip():
                continue
            
            fields = line.split('\t')
            if len(fields) >= 5:
                genomic_variants.append({
                    "chromosome": fields[0],
                    "position": fields[1],
                    "ref": fields[3],
                    "alt": fields[4],
                    "gene": self._identify_gene(fields[0], fields[1])
                })
        
        return {
            "variants": genomic_variants,
            "pathogenic_variants": self._identify_pathogenic(genomic_variants)
        }
    
    def _identify_gene(self, chromosome: str, position: str) -> str:
        """Map position to gene (simplified)"""
        gene_map = {
            "17:41000000-42000000": "BRCA1",
            "13:32000000-33000000": "BRCA2",
        }
        return gene_map.get(f"{chromosome}:{position}", "Unknown")
    
    def _identify_pathogenic(self, variants: List[Dict]) -> List[Dict]:
        """Identify pathogenic variants"""
        pathogenic = []
        cancer_genes = ["BRCA1", "BRCA2", "TP53", "PTEN", "ATM"]
        
        for variant in variants:
            if variant["gene"] in cancer_genes:
                pathogenic.append(variant)
        
        return pathogenic
    
    def fetch_wearable_data(self, patient_id: str, days: int = 7) -> pd.DataFrame:
        """Fetch wearable device data (glucose, heart rate, etc.)"""
        # In production, integrate with Dexcom, Fitbit, Apple Health APIs
        # Simulated data for demo
        data = {
            "timestamp": pd.date_range(end=datetime.now(), periods=days*24, freq='H'),
            "glucose_level": [100 + i % 50 for i in range(days*24)],
            "heart_rate": [70 + i % 30 for i in range(days*24)]
        }
        return pd.DataFrame(data)

# ============================================================================
# AGENT SYSTEM
# ============================================================================

class DiagnosticAgent:
    """Agent for cancer diagnosis and classification"""
    
    def __init__(self, llm_api_key: str):
        self.api_key = llm_api_key
    
    def diagnose(self, ehr_data: Dict, genomic_data: Dict) -> Dict:
        """Classify cancer subtype and stage"""
        # Extract relevant features
        conditions = ehr_data.get("conditions", {})
        pathogenic_variants = genomic_data.get("pathogenic_variants", [])
        
        # Simplified classification logic
        # In production, use fine-tuned BioBERT or similar
        diagnosis = {
            "cancer_type": "Breast Cancer",
            "subtype": "Unknown",
            "stage": "Unknown",
            "molecular_markers": [],
            "confidence": 0.0
        }
        
        # Check for BRCA mutations
        for variant in pathogenic_variants:
            if variant["gene"] in ["BRCA1", "BRCA2"]:
                diagnosis["molecular_markers"].append(variant["gene"])
                diagnosis["subtype"] = "Triple Negative" if variant["gene"] == "BRCA1" else "Luminal A"
                diagnosis["confidence"] = 0.85
        
        return diagnosis

class TreatmentAgent:
    """Agent for treatment recommendation"""
    
    def __init__(self, llm_api_key: str):
        self.api_key = llm_api_key
        self.clinical_trials_db = self._load_clinical_trials()
    
    def _load_clinical_trials(self) -> List[Dict]:
        """Load clinical trials database"""
        # In production, scrape from ClinicalTrials.gov
        return [
            {
                "nct_id": "NCT12345678",
                "title": "Olaparib for BRCA-mutated Breast Cancer",
                "drug": "Olaparib",
                "indication": "BRCA1/2 mutations",
                "phase": "III",
                "efficacy": 0.72
            },
            {
                "nct_id": "NCT87654321",
                "title": "Pembrolizumab + Chemotherapy",
                "drug": "Pembrolizumab",
                "indication": "Triple Negative Breast Cancer",
                "phase": "III",
                "efficacy": 0.65
            }
        ]
    
    def recommend_treatment(self, diagnosis: Dict) -> List[Dict]:
        """Recommend evidence-based treatments"""
        recommendations = []
        
        # Match diagnosis with clinical trials
        for trial in self.clinical_trials_db:
            if self._matches_indication(diagnosis, trial):
                recommendations.append({
                    "drug": trial["drug"],
                    "trial_id": trial["nct_id"],
                    "efficacy": trial["efficacy"],
                    "rationale": self._generate_rationale(diagnosis, trial)
                })
        
        # Sort by efficacy
        recommendations.sort(key=lambda x: x["efficacy"], reverse=True)
        return recommendations
    
    def _matches_indication(self, diagnosis: Dict, trial: Dict) -> bool:
        """Check if diagnosis matches trial indication"""
        for marker in diagnosis.get("molecular_markers", []):
            if marker in trial["indication"]:
                return True
        if diagnosis["subtype"] in trial["indication"]:
            return True
        return False
    
    def _generate_rationale(self, diagnosis: Dict, trial: Dict) -> str:
        """Generate explanation for recommendation"""
        return f"Recommended based on {trial['indication']} match. Phase {trial['phase']} trial showing {trial['efficacy']:.0%} efficacy."

class SafetyAgent:
    """Agent for drug interaction and safety checks"""
    
    def __init__(self):
        self.interaction_db = self._load_interactions()
    
    def _load_interactions(self) -> Dict:
        """Load drug interaction database"""
        # In production, integrate with DrugBank API
        return {
            "Olaparib": {
                "contraindications": ["Strong CYP3A4 inhibitors"],
                "warnings": ["Myelosuppression risk"],
                "interactions": ["Warfarin", "Simvastatin"]
            },
            "Pembrolizumab": {
                "contraindications": ["Active autoimmune disease"],
                "warnings": ["Immune-related adverse events"],
                "interactions": ["Corticosteroids"]
            }
        }
    
    def check_safety(self, treatment: Dict, medications: List[str]) -> Dict:
        """Check for drug interactions and contraindications"""
        drug = treatment["drug"]
        drug_info = self.interaction_db.get(drug, {})
        
        interactions_found = []
        for med in medications:
            if med in drug_info.get("interactions", []):
                interactions_found.append({
                    "drug": med,
                    "severity": "moderate",
                    "description": f"Potential interaction between {drug} and {med}"
                })
        
        return {
            "safe": len(interactions_found) == 0,
            "interactions": interactions_found,
            "warnings": drug_info.get("warnings", []),
            "contraindications": drug_info.get("contraindications", [])
        }

class AgentOrchestrator:
    """Orchestrates multi-agent workflow"""
    
    def __init__(self, llm_api_key: str):
        self.diagnostic_agent = DiagnosticAgent(llm_api_key)
        self.treatment_agent = TreatmentAgent(llm_api_key)
        self.safety_agent = SafetyAgent()
    
    def process_patient(self, patient_id: str, ehr_data: Dict, 
                       genomic_data: Dict, current_medications: List[str]) -> Dict:
        """Execute complete agentic workflow"""
        
        # Step 1: Diagnosis
        diagnosis = self.diagnostic_agent.diagnose(ehr_data, genomic_data)
        
        # Step 2: Treatment recommendation
        treatments = self.treatment_agent.recommend_treatment(diagnosis)
        
        # Step 3: Safety checks
        safe_treatments = []
        for treatment in treatments:
            safety_check = self.safety_agent.check_safety(treatment, current_medications)
            treatment["safety"] = safety_check
            if safety_check["safe"]:
                safe_treatments.append(treatment)
        
        return {
            "patient_id": patient_id,
            "diagnosis": diagnosis,
            "recommended_treatments": safe_treatments,
            "all_treatments": treatments,
            "timestamp": datetime.now().isoformat()
        }

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="Treatment Recommendation Agent",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Personalized Treatment Recommendation Agent")
    st.markdown("*Agentic AI system for cancer treatment recommendations*")
    
    # Sidebar - Hospital Configuration
    with st.sidebar:
        st.header("⚙️ Hospital Configuration")
        
        # API Configuration
        hospital_name = st.text_input("Hospital Name", "Demo Hospital")
        api_base_url = st.text_input("API Base URL", "https://hapi.fhir.org/baseR4")
        api_key = st.text_input("API Key", type="password", value="demo_key")
        
        st.divider()
        
        # LLM Configuration
        st.subheader("LLM Configuration")
        llm_api_key = st.text_input("OpenAI/Claude API Key", type="password", value="demo_llm_key")
        
        st.divider()
        
        # Data Sources
        st.subheader("Data Sources")
        use_ehr = st.checkbox("EHR Data", value=True)
        use_genomics = st.checkbox("Genomic Data", value=True)
        use_wearables = st.checkbox("Wearable Data", value=False)
    
    # Main Content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Patient Data", "🧬 Analysis", "💊 Recommendations", "📈 Monitoring"])
    
    with tab1:
        st.header("Patient Data Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            patient_id = st.text_input("Patient ID", "P12345")
            
            if use_ehr:
                st.subheader("Current Medications")
                medications_text = st.text_area(
                    "Enter medications (one per line)",
                    "Metformin\nLisinopril\nAspirin"
                )
                current_medications = [m.strip() for m in medications_text.split('\n') if m.strip()]
        
        with col2:
            if use_genomics:
                st.subheader("Genomic Data (VCF)")
                vcf_data = st.text_area(
                    "Paste VCF data or upload file",
                    "17\t41000000\t.\tA\tG\tBRCA1\n13\t32000000\t.\tC\tT\tBRCA2",
                    height=150
                )
        
        if st.button("🔍 Fetch & Analyze Patient Data", type="primary"):
            with st.spinner("Processing patient data..."):
                # Initialize system
                config = HospitalAPIConfig(
                    base_url=api_base_url,
                    api_key=api_key
                )
                api_client = HospitalAPIClient(config)
                ingestion_engine = DataIngestionEngine(api_client)
                
                # Fetch data
                ehr_data = ingestion_engine.fetch_ehr_data(patient_id) if use_ehr else {}
                genomic_data = ingestion_engine.parse_genomic_data(vcf_data) if use_genomics else {}
                wearable_data = ingestion_engine.fetch_wearable_data(patient_id) if use_wearables else None
                
                # Store in session state
                st.session_state['ehr_data'] = ehr_data
                st.session_state['genomic_data'] = genomic_data
                st.session_state['wearable_data'] = wearable_data
                st.session_state['current_medications'] = current_medications
                st.session_state['patient_id'] = patient_id
                
                st.success("✅ Data fetched successfully!")
    
    with tab2:
        st.header("Diagnostic Analysis")
        
        if 'genomic_data' in st.session_state:
            genomic_data = st.session_state['genomic_data']
            
            # Display genomic variants
            st.subheader("🧬 Genomic Variants Detected")
            if genomic_data.get('pathogenic_variants'):
                df = pd.DataFrame(genomic_data['pathogenic_variants'])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No pathogenic variants detected")
            
            # Run diagnosis
            if st.button("🔬 Run Diagnostic Analysis"):
                with st.spinner("Running diagnostic agents..."):
                    orchestrator = AgentOrchestrator(llm_api_key)
                    
                    diagnosis = orchestrator.diagnostic_agent.diagnose(
                        st.session_state.get('ehr_data', {}),
                        genomic_data
                    )
                    
                    st.session_state['diagnosis'] = diagnosis
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Cancer Type", diagnosis['cancer_type'])
                    
                    with col2:
                        st.metric("Subtype", diagnosis['subtype'])
                    
                    with col3:
                        st.metric("Confidence", f"{diagnosis['confidence']:.0%}")
                    
                    if diagnosis['molecular_markers']:
                        st.success(f"**Molecular Markers:** {', '.join(diagnosis['molecular_markers'])}")
        else:
            st.warning("⚠️ Please fetch patient data first from the 'Patient Data' tab")
    
    with tab3:
        st.header("Treatment Recommendations")
        
        if 'diagnosis' in st.session_state:
            diagnosis = st.session_state['diagnosis']
            
            if st.button("💊 Generate Treatment Recommendations"):
                with st.spinner("Analyzing treatment options..."):
                    orchestrator = AgentOrchestrator(llm_api_key)
                    
                    results = orchestrator.process_patient(
                        st.session_state.get('patient_id', 'Unknown'),
                        st.session_state.get('ehr_data', {}),
                        st.session_state.get('genomic_data', {}),
                        st.session_state.get('current_medications', [])
                    )
                    
                    st.session_state['results'] = results
                    
                    # Display recommendations
                    st.subheader("✅ Safe Treatment Options")
                    
                    for i, treatment in enumerate(results['recommended_treatments'], 1):
                        with st.expander(f"Option {i}: {treatment['drug']} (Efficacy: {treatment['efficacy']:.0%})", expanded=i==1):
                            st.write(f"**Rationale:** {treatment['rationale']}")
                            st.write(f"**Clinical Trial:** {treatment['trial_id']}")
                            
                            safety = treatment['safety']
                            if safety['warnings']:
                                st.warning("⚠️ **Warnings:**")
                                for warning in safety['warnings']:
                                    st.write(f"- {warning}")
                    
                    # Display unsafe options
                    if len(results['all_treatments']) > len(results['recommended_treatments']):
                        st.subheader("⚠️ Options with Safety Concerns")
                        
                        unsafe = [t for t in results['all_treatments'] if not t['safety']['safe']]
                        for treatment in unsafe:
                            with st.expander(f"❌ {treatment['drug']}", expanded=False):
                                st.write(f"**Interactions Found:**")
                                for interaction in treatment['safety']['interactions']:
                                    st.error(f"- {interaction['description']}")
        else:
            st.warning("⚠️ Please run diagnostic analysis first from the 'Analysis' tab")
    
    with tab4:
        st.header("Patient Monitoring")
        
        if st.session_state.get('wearable_data') is not None:
            wearable_data = st.session_state['wearable_data']
            
            st.subheader("📊 Wearable Data Trends")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.line_chart(wearable_data.set_index('timestamp')['glucose_level'])
                st.caption("Glucose Levels (mg/dL)")
            
            with col2:
                st.line_chart(wearable_data.set_index('timestamp')['heart_rate'])
                st.caption("Heart Rate (bpm)")
        else:
            st.info("Enable wearable data in the sidebar to view monitoring dashboard")
    
    # Footer
    st.divider()
    st.markdown("""
    **System Status:** 🟢 Active | **Hospital:** {} | **Agent Version:** 1.0
    
    *This is a demo system. Always consult with healthcare professionals for medical decisions.*
    """.format(hospital_name))

if __name__ == "__main__":
    main()
