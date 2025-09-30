"""
Autonomous Remote Patient Monitoring Agent
==========================================
Real-time monitoring system for chronic disease patients with automatic alerting.
Configurable for any hospital API.

Installation:
pip install streamlit pandas numpy scikit-learn plotly requests python-dotenv twilio

Usage:
streamlit run monitoring_agent.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import time

# ============================================================================
# CONFIGURATION MODULE
# ============================================================================

@dataclass
class MonitoringConfig:
    """Configuration for patient monitoring system"""
    hospital_api_url: str
    api_key: str
    alert_phone: str = ""
    alert_email: str = ""
    
    # Thresholds for different conditions
    glucose_thresholds: Dict[str, float] = None
    heart_rate_thresholds: Dict[str, float] = None
    bp_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.glucose_thresholds is None:
            self.glucose_thresholds = {
                "critical_low": 54,      # mg/dL
                "low": 70,
                "target_low": 80,
                "target_high": 180,
                "high": 250,
                "critical_high": 300
            }
        
        if self.heart_rate_thresholds is None:
            self.heart_rate_thresholds = {
                "critical_low": 40,      # bpm
                "low": 60,
                "normal_low": 60,
                "normal_high": 100,
                "high": 120,
                "critical_high": 150
            }
        
        if self.bp_thresholds is None:
            self.bp_thresholds = {
                "systolic_low": 90,
                "systolic_high": 140,
                "diastolic_low": 60,
                "diastolic_high": 90
            }

class RiskLevel(Enum):
    """Risk level classification"""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Types of alerts"""
    GLUCOSE_HIGH = "glucose_high"
    GLUCOSE_LOW = "glucose_low"
    HEART_RATE = "heart_rate"
    BLOOD_PRESSURE = "blood_pressure"
    NO_DATA = "no_data"
    TREND = "trend"

# ============================================================================
# DATA STREAMING SIMULATOR
# ============================================================================

class WearableDataSimulator:
    """Simulates real-time data from wearable devices"""
    
    def __init__(self, patient_id: str, condition: str = "diabetes"):
        self.patient_id = patient_id
        self.condition = condition
        self.base_glucose = 120 if condition == "diabetes" else 90
        self.base_hr = 75
        
    def generate_glucose_reading(self, timestamp: datetime, 
                                 inject_anomaly: bool = False) -> float:
        """Generate realistic glucose reading"""
        # Base pattern: higher in morning, lower at night
        hour = timestamp.hour
        circadian_factor = 1.2 if 6 <= hour <= 9 else 0.9 if 22 <= hour <= 6 else 1.0
        
        # Random variation
        noise = np.random.normal(0, 10)
        
        glucose = self.base_glucose * circadian_factor + noise
        
        # Inject anomaly for testing
        if inject_anomaly:
            glucose = np.random.choice([50, 280])  # Hypo or hyperglycemia
        
        return max(40, min(400, glucose))  # Realistic bounds
    
    def generate_heart_rate(self, timestamp: datetime) -> int:
        """Generate realistic heart rate reading"""
        # Higher during day, lower at night
        hour = timestamp.hour
        activity_factor = 1.3 if 9 <= hour <= 17 else 0.85 if 22 <= hour <= 6 else 1.0
        
        noise = np.random.normal(0, 5)
        hr = int(self.base_hr * activity_factor + noise)
        
        return max(40, min(180, hr))
    
    def generate_blood_pressure(self, timestamp: datetime) -> Tuple[int, int]:
        """Generate realistic blood pressure reading"""
        systolic = int(np.random.normal(120, 10))
        diastolic = int(np.random.normal(80, 7))
        
        return (max(70, min(200, systolic)), max(40, min(120, diastolic)))
    
    def generate_stream(self, duration_hours: int = 24, 
                       interval_minutes: int = 15) -> pd.DataFrame:
        """Generate a stream of readings"""
        timestamps = pd.date_range(
            end=datetime.now(),
            periods=int(duration_hours * 60 / interval_minutes),
            freq=f'{interval_minutes}min'
        )
        
        data = []
        for ts in timestamps:
            glucose = self.generate_glucose_reading(ts)
            hr = self.generate_heart_rate(ts)
            bp_sys, bp_dia = self.generate_blood_pressure(ts)
            
            data.append({
                'timestamp': ts,
                'glucose': glucose,
                'heart_rate': hr,
                'systolic_bp': bp_sys,
                'diastolic_bp': bp_dia
            })
        
        return pd.DataFrame(data)

# ============================================================================
# ANOMALY DETECTION ENGINE
# ============================================================================

class AnomalyDetector:
    """Detects anomalies in patient vital signs"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alert_history = []
    
    def check_glucose(self, reading: float, history: List[float]) -> Dict:
        """Check glucose levels for anomalies"""
        thresholds = self.config.glucose_thresholds
        
        risk_level = RiskLevel.NORMAL
        alert = None
        
        if reading <= thresholds['critical_low']:
            risk_level = RiskLevel.CRITICAL
            alert = {
                'type': AlertType.GLUCOSE_LOW,
                'severity': 'critical',
                'message': f'CRITICAL: Severe hypoglycemia detected ({reading:.0f} mg/dL)',
                'action': 'Immediate intervention required'
            }
        elif reading <= thresholds['low']:
            risk_level = RiskLevel.HIGH
            alert = {
                'type': AlertType.GLUCOSE_LOW,
                'severity': 'high',
                'message': f'LOW: Hypoglycemia warning ({reading:.0f} mg/dL)',
                'action': 'Consume 15g fast-acting carbs'
            }
        elif reading >= thresholds['critical_high']:
            risk_level = RiskLevel.CRITICAL
            alert = {
                'type': AlertType.GLUCOSE_HIGH,
                'severity': 'critical',
                'message': f'CRITICAL: Severe hyperglycemia ({reading:.0f} mg/dL)',
                'action': 'Check for ketones, contact doctor'
            }
        elif reading >= thresholds['high']:
            risk_level = RiskLevel.HIGH
            alert = {
                'type': AlertType.GLUCOSE_HIGH,
                'severity': 'high',
                'message': f'HIGH: Hyperglycemia detected ({reading:.0f} mg/dL)',
                'action': 'Increase insulin per sliding scale'
            }
        
        # Check for prolonged elevated levels
        if len(history) >= 8 and all(h >= thresholds['high'] for h in history[-8:]):
            if alert is None or alert['severity'] != 'critical':
                alert = {
                    'type': AlertType.TREND,
                    'severity': 'high',
                    'message': 'Sustained hyperglycemia for 2+ hours',
                    'action': 'Contact healthcare provider'
                }
                risk_level = RiskLevel.HIGH
        
        return {
            'reading': reading,
            'risk_level': risk_level.value,
            'alert': alert,
            'timestamp': datetime.now()
        }
    
    def check_heart_rate(self, hr: int) -> Dict:
        """Check heart rate for anomalies"""
        thresholds = self.config.heart_rate_thresholds
        
        risk_level = RiskLevel.NORMAL
        alert = None
        
        if hr <= thresholds['critical_low'] or hr >= thresholds['critical_high']:
            risk_level = RiskLevel.CRITICAL
            alert = {
                'type': AlertType.HEART_RATE,
                'severity': 'critical',
                'message': f'CRITICAL: Abnormal heart rate ({hr} bpm)',
                'action': 'Seek immediate medical attention'
            }
        elif hr < thresholds['normal_low'] or hr > thresholds['normal_high']:
            risk_level = RiskLevel.ELEVATED
            alert = {
                'type': AlertType.HEART_RATE,
                'severity': 'elevated',
                'message': f'Heart rate outside normal range ({hr} bpm)',
                'action': 'Monitor closely'
            }
        
        return {
            'reading': hr,
            'risk_level': risk_level.value,
            'alert': alert,
            'timestamp': datetime.now()
        }
    
    def check_data_gaps(self, last_reading_time: datetime) -> Optional[Dict]:
        """Check for data gaps indicating device issues"""
        time_since_last = datetime.now() - last_reading_time
        
        if time_since_last > timedelta(hours=6):
            return {
                'type': AlertType.NO_DATA,
                'severity': 'high',
                'message': f'No data received for {time_since_last.seconds//3600} hours',
                'action': 'Check device connectivity'
            }
        
        return None
    
    def predict_trend(self, history: List[float], metric: str) -> Dict:
        """Predict if metric is trending towards danger zone"""
        if len(history) < 4:
            return {'trending': False}
        
        # Simple linear regression
        recent = history[-4:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        
        # Check if trending towards danger
        if metric == "glucose":
            thresholds = self.config.glucose_thresholds
            last_value = recent[-1]
            predicted_next = last_value + slope
            
            if slope > 0 and predicted_next > thresholds['high']:
                return {
                    'trending': True,
                    'direction': 'increasing',
                    'message': f'Glucose trending up (slope: {slope:.1f} mg/dL per reading)',
                    'predicted': predicted_next
                }
            elif slope < 0 and predicted_next < thresholds['low']:
                return {
                    'trending': True,
                    'direction': 'decreasing',
                    'message': f'Glucose trending down (slope: {slope:.1f} mg/dL per reading)',
                    'predicted': predicted_next
                }
        
        return {'trending': False}

# ============================================================================
# ALERT MANAGEMENT
# ============================================================================

class AlertManager:
    """Manages alert generation and notification"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.active_alerts = []
        self.alert_history = []
        self.suppression_time = {}  # Prevent alert fatigue
    
    def should_send_alert(self, alert_type: AlertType) -> bool:
        """Check if enough time has passed since last alert of this type"""
        if alert_type.value not in self.suppression_time:
            return True
        
        time_since = datetime.now() - self.suppression_time[alert_type.value]
        
        # Don't repeat same alert within 30 minutes
        return time_since > timedelta(minutes=30)
    
    def send_alert(self, alert: Dict, patient_id: str) -> bool:
        """Send alert via configured channels"""
        if not self.should_send_alert(alert['type']):
            return False
        
        # Log alert
        alert_record = {
            'patient_id': patient_id,
            'timestamp': datetime.now(),
            'alert': alert,
            'status': 'sent'
        }
        
        self.alert_history.append(alert_record)
        self.active_alerts.append(alert)
        self.suppression_time[alert['type'].value] = datetime.now()
        
        # Send notifications (in production, use Twilio, SendGrid, etc.)
        if self.config.alert_phone:
            self._send_sms(alert, patient_id)
        
        if self.config.alert_email:
            self._send_email(alert, patient_id)
        
        return True
    
    def _send_sms(self, alert: Dict, patient_id: str):
        """Send SMS alert (placeholder)"""
        # In production:
        # from twilio.rest import Client
        # client = Client(account_sid, auth_token)
        # message = client.messages.create(...)
        pass
    
    def _send_email(self, alert: Dict, patient_id: str):
        """Send email alert (placeholder)"""
        # In production: use SendGrid, AWS SES, etc.
        pass
    
    def dismiss_alert(self, alert_id: int):
        """Dismiss an active alert"""
        if alert_id < len(self.active_alerts):
            dismissed = self.active_alerts.pop(alert_id)
            dismissed['status'] = 'dismissed'
            dismissed['dismissed_at'] = datetime.now()

# ============================================================================
# CARE ADJUSTMENT AGENT
# ============================================================================

class CareAdjustmentAgent:
    """Agent that suggests care plan adjustments"""
    
    def __init__(self):
        self.intervention_database = self._load_interventions()
    
    def _load_interventions(self) -> Dict:
        """Load evidence-based interventions"""
        return {
            'glucose_high': [
                {
                    'condition': 'glucose > 250 for 2+ hours',
                    'intervention': 'Increase rapid-acting insulin by 2 units',
                    'evidence': 'ADA Guidelines 2024'
                },
                {
                    'condition': 'glucose > 300',
                    'intervention': 'Check for ketones; if positive, contact physician',
                    'evidence': 'ADA Guidelines 2024'
                }
            ],
            'glucose_low': [
                {
                    'condition': 'glucose < 70',
                    'intervention': 'Rule of 15: 15g fast carbs, recheck in 15 min',
                    'evidence': 'ADA Guidelines 2024'
                },
                {
                    'condition': 'glucose < 54',
                    'intervention': 'Glucagon injection if unable to consume carbs',
                    'evidence': 'ADA Guidelines 2024'
                }
            ]
        }
    
    def suggest_intervention(self, anomaly_result: Dict, 
                           patient_history: Dict) -> Optional[Dict]:
        """Suggest care plan adjustment based on anomaly"""
        if not anomaly_result.get('alert'):
            return None
        
        alert = anomaly_result['alert']
        alert_type = str(alert['type'].value) if isinstance(alert['type'], AlertType) else alert['type']
        
        interventions = self.intervention_database.get(alert_type, [])
        
        for intervention in interventions:
            # Match condition (simplified - in production use ML)
            if self._matches_condition(intervention['condition'], anomaly_result):
                return {
                    'intervention': intervention['intervention'],
                    'rationale': f"Based on {intervention['evidence']}",
                    'confidence': 0.9,
                    'requires_approval': True
                }
        
        return None
    
    def _matches_condition(self, condition: str, result: Dict) -> bool:
        """Check if condition matches current situation"""
        # Simplified matching logic
        reading = result.get('reading', 0)
        
        if 'glucose > 250' in condition and reading > 250:
            return True
        if 'glucose > 300' in condition and reading > 300:
            return True
        if 'glucose < 70' in condition and reading < 70:
            return True
        if 'glucose < 54' in condition and reading < 54:
            return True
        
        return False

# ============================================================================
# STREAMLIT UI
# ============================================================================

def create_vitals_chart(data: pd.DataFrame, metric: str, thresholds: Dict) -> go.Figure:
    """Create interactive chart with threshold zones"""
    fig = go.Figure()
    
    # Add threshold zones
    if metric == "glucose":
        fig.add_hrect(y0=0, y1=thresholds['critical_low'], 
                     fillcolor="red", opacity=0.1, line_width=0)
        fig.add_hrect(y0=thresholds['critical_low'], y1=thresholds['low'],
                     fillcolor="orange", opacity=0.1, line_width=0)
        fig.add_hrect(y0=thresholds['target_low'], y1=thresholds['target_high'],
                     fillcolor="green", opacity=0.1, line_width=0)
        fig.add_hrect(y0=thresholds['high'], y1=thresholds['critical_high'],
                     fillcolor="orange", opacity=0.1, line_width=0)
        fig.add_hrect(y0=thresholds['critical_high'], y1=500,
                     fillcolor="red", opacity=0.1, line_width=0)
    
    # Add data line
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data[metric],
        mode='lines+markers',
        name=metric.replace('_', ' ').title(),
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title=metric.replace('_', ' ').title(),
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode='x unified',
        height=300
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="Remote Patient Monitoring",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Remote Patient Monitoring Agent")
    st.markdown("*Real-time monitoring and alerting for chronic disease management*")
    
    # Initialize session state
    if 'monitoring_active' not in st.session_state:
        st.session_state['monitoring_active'] = False
    if 'patient_data' not in st.session_state:
        st.session_state['patient_data'] = None
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # Hospital API
        hospital_api = st.text_input("Hospital API URL", "https://api.hospital.com")
        api_key = st.text_input("API Key", type="password", value="demo_key")
        
        st.divider()
        
        # Alert Configuration
        st.subheader("🚨 Alert Settings")
        alert_phone = st.text_input("Alert Phone Number", "+1234567890")
        alert_email = st.text_input("Alert Email", "doctor@hospital.com")
        
        st.divider()
        
        # Monitoring Parameters
        st.subheader("📏 Thresholds")
        
        with st.expander("Glucose Thresholds"):
            glucose_low = st.number_input("Low (mg/dL)", value=70)
            glucose_high = st.number_input("High (mg/dL)", value=250)
            glucose_critical_low = st.number_input("Critical Low", value=54)
            glucose_critical_high = st.number_input("Critical High", value=300)
        
        with st.expander("Heart Rate Thresholds"):
            hr_low = st.number_input("Low HR (bpm)", value=60)
            hr_high = st.number_input("High HR (bpm)", value=100)
    
    # Main Content
    tab1, tab2, tab3, tab4 = st.tabs(["👤 Patient Setup", "📈 Live Monitoring", "🚨 Alerts", "📋 Reports"])
    
    with tab1:
        st.header("Patient Setup")
        
        col1, col2 = st.columns(2)
        
        with col1:
            patient_id = st.text_input("Patient ID", "P67890")
            patient_name = st.text_input("Patient Name", "John Doe")
            condition = st.selectbox("Primary Condition", 
                                    ["Diabetes Type 1", "Diabetes Type 2", 
                                     "Hypertension", "Heart Disease"])
        
        with col2:
            age = st.number_input("Age", value=55, min_value=0, max_value=120)
            monitoring_duration = st.slider("Monitoring Duration (hours)", 
                                          1, 48, 24)
            data_interval = st.slider("Data Interval (minutes)", 
                                     5, 60, 15)
        
        if st.button("🚀 Initialize Monitoring", type="primary"):
            with st.spinner("Setting up monitoring system..."):
                # Create configuration
                config = MonitoringConfig(
                    hospital_api_url=hospital_api,
                    api_key=api_key,
                    alert_phone=alert_phone,
                    alert_email=alert_email
                )
                
                config.glucose_thresholds.update({
                    'low': glucose_low,
                    'high': glucose_high,
                    'critical_low': glucose_critical_low,
                    'critical_high': glucose_critical_high
                })
                
                # Generate patient data stream
                simulator = WearableDataSimulator(patient_id, "diabetes")
                data = simulator.generate_stream(
                    duration_hours=monitoring_duration,
                    interval_minutes=data_interval
                )
                
                # Initialize components
                detector = AnomalyDetector(config)
                alert_manager = AlertManager(config)
                care_agent = CareAdjustmentAgent()
                
                # Store in session state
                st.session_state['config'] = config
                st.session_state['patient_data'] = data
                st.session_state['detector'] = detector
                st.session_state['alert_manager'] = alert_manager
                st.session_state['care_agent'] = care_agent
                st.session_state['patient_id'] = patient_id
                st.session_state['patient_name'] = patient_name
                st.session_state['monitoring_active'] = True
                
                st.success(f"✅ Monitoring initialized for {patient_name}")
                st.balloons()
    
    with tab2:
        st.header("Live Monitoring Dashboard")
        
        if st.session_state.get('monitoring_active'):
            data = st.session_state['patient_data']
            config = st.session_state['config']
            detector = st.session_state['detector']
            
            # Current vitals
            latest = data.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                glucose_result = detector.check_glucose(
                    latest['glucose'],
                    data['glucose'].tail(10).tolist()
                )
                st.metric(
                    "Glucose",
                    f"{latest['glucose']:.0f} mg/dL",
                    delta=f"{latest['glucose'] - data.iloc[-2]['glucose']:.0f}",
                    delta_color="inverse"
                )
                if glucose_result['alert']:
                    st.error(f"⚠️ {glucose_result['alert']['severity'].upper()}")
            
            with col2:
                hr_result = detector.check_heart_rate(int(latest['heart_rate']))
                st.metric(
                    "Heart Rate",
                    f"{int(latest['heart_rate'])} bpm"
                )
                if hr_result['alert']:
                    st.error(f"⚠️ {hr_result['alert']['severity'].upper()}")
            
            with col3:
                st.metric(
                    "Blood Pressure",
                    f"{int(latest['systolic_bp'])}/{int(latest['diastolic_bp'])}"
                )
            
            with col4:
                st.metric(
                    "Last Reading",
                    latest['timestamp'].strftime("%H:%M")
                )
            
            st.divider()
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig_glucose = create_vitals_chart(
                    data,
                    'glucose',
                    config.glucose_thresholds
                )
                st.plotly_chart(fig_glucose, use_container_width=True)
            
            with col2:
                fig_hr = go.Figure()
                fig_hr.add_trace(go.Scatter(
                    x=data['timestamp'],
                    y=data['heart_rate'],
                    mode='lines',
                    name='Heart Rate',
                    line=dict(color='red')
                ))
                fig_hr.update_layout(
                    title="Heart Rate Trend",
                    xaxis_title="Time",
                    yaxis_title="BPM",
                    height=300
                )
                st.plotly_chart(fig_hr, use_container_width=True)
            
            # Trend analysis
            st.subheader("📊 Trend Analysis")
            
            trend = detector.predict_trend(
                data['glucose'].tail(8).tolist(),
                'glucose'
            )
            
            if trend.get('trending'):
                st.warning(f"⚠️ {trend['message']}")
                st.info(f"Predicted next reading: {trend['predicted']:.0f} mg/dL")
            else:
                st.success("✅ No concerning trends detected")
            
        else:
            st.warning("⚠️ Please initialize monitoring in the 'Patient Setup' tab")
    
    with tab3:
        st.header("Alert Management")
        
        if st.session_state.get('monitoring_active'):
            alert_manager = st.session_state['alert_manager']
            detector = st.session_state['detector']
            care_agent = st.session_state['care_agent']
            data = st.session_state['patient_data']
            
            # Process recent readings for alerts
            st.subheader("🔔 Active Alerts")
            
            recent_readings = data.tail(5)
            active_alerts = []
            
            for _, row in recent_readings.iterrows():
                glucose_result = detector.check_glucose(
                    row['glucose'],
                    data['glucose'].tolist()
                )
                
                if glucose_result.get('alert'):
                    alert = glucose_result['alert']
                    
                    with st.container():
                        col1, col2, col3 = st.columns([2, 2, 1])
                        
                        with col1:
                            severity_color = {
                                'critical': '🔴',
                                'high': '🟠',
                                'elevated': '🟡'
                            }.get(alert['severity'], '🔵')
                            
                            st.markdown(f"### {severity_color} {alert['message']}")
                            st.write(f"**Action:** {alert['action']}")
                            st.caption(f"Time: {row['timestamp'].strftime('%H:%M:%S')}")
                        
                        with col2:
                            # Get care suggestion
                            suggestion = care_agent.suggest_intervention(
                                glucose_result,
                                {}
                            )
                            
                            if suggestion:
                                st.info(f"💡 **Suggested:** {suggestion['intervention']}")
                                st.caption(f"Evidence: {suggestion['rationale']}")
                        
                        with col3:
                            if st.button("Dismiss", key=f"dismiss_{row['timestamp']}"):
                                st.success("Alert dismissed")
                    
                    st.divider()
            
            # Alert history
            st.subheader("📜 Alert History")
            
            if alert_manager.alert_history:
                history_df = pd.DataFrame([
                    {
                        'Time': a['timestamp'].strftime('%Y-%m-%d %H:%M'),
                        'Type': a['alert']['type'].value if isinstance(a['alert']['type'], AlertType) else a['alert'].get('type', 'Unknown'),
                        'Severity': a['alert']['severity'],
                        'Message': a['alert']['message'],
                        'Status': a.get('status', 'active')
                    }
                    for a in alert_manager.alert_history
                ])
                st.dataframe(history_df, use_container_width=True)
            else:
                st.info("No alerts in history")
        
        else:
            st.warning("⚠️ Please initialize monitoring first")
    
    with tab4:
        st.header("Monitoring Reports")
        
        if st.session_state.get('monitoring_active'):
            data = st.session_state['patient_data']
            
            # Summary statistics
            st.subheader("📊 Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Glucose", f"{data['glucose'].mean():.0f} mg/dL")
                st.metric("Std Dev", f"{data['glucose'].std():.1f}")
            
            with col2:
                st.metric("Time in Range", 
                         f"{((data['glucose'] >= 80) & (data['glucose'] <= 180)).mean()*100:.0f}%")
                st.metric("Target: 70%+", "")
            
            with col3:
                st.metric("Hypoglycemia Events", 
                         f"{(data['glucose'] < 70).sum()}")
                st.metric("Hyperglycemia Events",
                         f"{(data['glucose'] > 250).sum()}")
            
            with col4:
                st.metric("Avg Heart Rate", 
                         f"{data['heart_rate'].mean():.0f} bpm")
                st.metric("Variability",
                         f"{data['heart_rate'].std():.1f}")
            
            # Distribution
            st.subheader("📈 Glucose Distribution")
            
            fig_dist = px.histogram(
                data,
                x='glucose',
                nbins=50,
                title="Glucose Level Distribution"
            )
            fig_dist.add_vline(x=80, line_dash="dash", line_color="green")
            fig_dist.add_vline(x=180, line_dash="dash", line_color="green")
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Export report
            if st.button("📥 Export Report"):
                report = {
                    'patient_id': st.session_state['patient_id'],
                    'patient_name': st.session_state['patient_name'],
                    'period': f"{data['timestamp'].min()} to {data['timestamp'].max()}",
                    'statistics': {
                        'avg_glucose': float(data['glucose'].mean()),
                        'std_glucose': float(data['glucose'].std()),
                        'time_in_range': float(((data['glucose'] >= 80) & (data['glucose'] <= 180)).mean()),
                        'hypo_events': int((data['glucose'] < 70).sum()),
                        'hyper_events': int((data['glucose'] > 250).sum())
                    }
                }
                
                st.download_button(
                    "Download JSON Report",
                    data=json.dumps(report, indent=2),
                    file_name=f"monitoring_report_{st.session_state['patient_id']}.json",
                    mime="application/json"
                )
        
        else:
            st.warning("⚠️ Please initialize monitoring first")
    
    # Footer
    st.divider()
    status = "🟢 ACTIVE" if st.session_state.get('monitoring_active') else "🔴 INACTIVE"
    st.markdown(f"""
    **Monitoring Status:** {status} | **Patient:** {st.session_state.get('patient_name', 'N/A')} | **Version:** 1.0
    
    *This is a demo system. Always verify with healthcare professionals.*
    """)

if __name__ == "__main__":
    main()
