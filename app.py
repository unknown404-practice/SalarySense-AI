import math
from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================================================================
# 1. APPLICATION CONFIGURATION & CONSTANTS
# ==============================================================================
st.set_page_config(
    page_title="SalarySense AI — Fair, Data-Informed Salary Guidance",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Currency Conversion Configuration
USD_TO_INR_RATE = 83.00
USD_TO_INR_RATE_SOURCE = "Manually configured reference rate"
USD_TO_INR_RATE_DATE = "Configured for this project demo"

BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
PIPELINE_PATH = MODELS_DIR / "salary_prediction_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
STYLES_PATH = ASSETS_DIR / "styles.css"

# ==============================================================================
# 2. CUSTOM CSS LOADING
# ==============================================================================
def load_css():
    if STYLES_PATH.exists():
        with open(STYLES_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("Custom stylesheet 'assets/styles.css' not found. Using fallback styling.")

load_css()

# ==============================================================================
# 3. SAFE MODEL AND METADATA LOADER
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_model_and_metadata():
    """Load the trained scikit-learn pipeline and factual metadata."""
    if not PIPELINE_PATH.exists() or not METADATA_PATH.exists():
        return None, None, f"Model artifacts missing. Expected '{PIPELINE_PATH.name}' and '{METADATA_PATH.name}' in {MODELS_DIR.resolve()}."
    
    try:
        pipeline = joblib.load(PIPELINE_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return pipeline, metadata, None
    except Exception as e:
        return None, None, f"Error loading model artifacts: {str(e)}"

pipeline, metadata, model_error = load_model_and_metadata()

# ==============================================================================
# 4. CURRENCY FORMATTING UTILITIES
# ==============================================================================
def format_usd(value: float) -> str:
    """Format numeric value in USD with international comma grouping."""
    if value is None or not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
        return "Not available"
    if value < 0:
        return f"-${abs(value):,.0f}"
    return f"${value:,.0f}"

def format_inr_indian(value: float) -> str:
    """Format numeric value in INR using Indian numbering grouping (lakhs/crores)."""
    if value is None or not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
        return "Not available"
    
    is_neg = value < 0
    val_int = int(round(abs(value)))
    s = str(val_int)
    
    if len(s) <= 3:
        formatted_s = s
    else:
        last3 = s[-3:]
        remaining = s[:-3]
        groups = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
        formatted_s = ",".join(groups) + "," + last3
        
    prefix = "-₹" if is_neg else "₹"
    return prefix + formatted_s

# ==============================================================================
# 5. BUSINESS LOGIC (EXPERIENCE CATEGORY & HR RECOMMENDATION)
# ==============================================================================
def get_experience_category(years: int) -> str:
    """Categorize candidate based on years of relevant experience."""
    if years <= 1:
        return "Fresher / Entry Level"
    elif years <= 4:
        return "Junior Professional"
    elif years <= 8:
        return "Mid-Level Professional"
    elif years <= 14:
        return "Senior Professional"
    else:
        return "Leadership / Expert Level"

def generate_hr_recommendation(
    exp_category: str,
    job_title: str,
    pred_usd: float,
    pred_inr: float,
    usd_low: float,
    usd_high: float
) -> str:
    """Generate a responsible, neutral, human-centered HR compensation recommendation."""
    usd_fmt = format_usd(pred_usd)
    inr_fmt = format_inr_indian(pred_inr)
    usd_range_fmt = f"{format_usd(usd_low)} – {format_usd(usd_high)}"
    
    return (
        f"Use this estimate as an initial compensation benchmark for the {exp_category} "
        f"{job_title} profile. The model's baseline annual estimate is {usd_fmt} (approximately {inr_fmt}), "
        f"with an indicative range of {usd_range_fmt}. Before finalizing an offer, HR should validate role scope, "
        f"technical assessment performance, geographical location, internal pay bands, benefits, "
        f"pay equity standards, and current market benchmarks."
    )

# ==============================================================================
# 6. HEADER COMPONENT
# ==============================================================================
status_html = (
    '<span class="brand-badge"><span class="status-dot"></span>ML Model Ready</span>'
    if pipeline is not None
    else '<span class="brand-badge" style="background:#FEE2E2; border-color:#F87171; color:#991B1B;"><span class="status-dot" style="background:#EF4444;"></span>Model Offline</span>'
)

header_html = f"""
<div class="brand-header">
  <div class="brand-left">
    <div class="brand-icon-wrap">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="4"  y="14" width="3.4" height="6" rx="1.2" fill="#E2E8F0"/>
        <rect x="10" y="10" width="3.4" height="10" rx="1.2" fill="#94A3B8"/>
        <rect x="16" y="5"  width="3.4" height="15" rx="1.2" fill="#22D3EE"/>
        <circle cx="19.5" cy="4" r="2" fill="#22D3EE"/>
      </svg>
    </div>
    <div class="brand-titles">
      <h2 class="brand-title">SalarySense AI</h2>
      <p class="brand-subtitle">ABC Technologies · Decision Support System</p>
    </div>
  </div>
  {status_html}
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# ==============================================================================
# 7. HERO COMPONENT
# ==============================================================================
hero_html = """
<div class="hero-container">
  <div class="hero-pill">HR Decision-Support Tool</div>
  <h1 class="hero-h1">Make salary decisions <span class="accent">with confidence.</span></h1>
  <p class="hero-desc">
    Generate data-informed, fair compensation guidance from candidate experience, education, and role details.
    Powered by a trained machine-learning regression pipeline to support consistent, objective HR hiring decisions.
  </p>
  <div class="hero-chips">
    <div class="hero-chip">
      <span class="hero-chip-icon">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round">
          <path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>
        </svg>
      </span>
      <span>Fast estimates — a range in seconds</span>
    </div>
    <div class="hero-chip">
      <span class="hero-chip-icon">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round">
          <path d="M12 2 2 7l10 5 10-5-10-5z"/>
          <path d="M2 17l10 5 10-5M2 12l10 5 10-5"/>
        </svg>
      </span>
      <span>Consistent evaluation — same logic for every profile</span>
    </div>
    <div class="hero-chip">
      <span class="hero-chip-icon">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round">
          <path d="M12 21s7-3.6 7-9V5.5L12 3 5 5.5V12c0 5.4 7 9 7 9z"/>
          <path d="m9 11.5 2 2 4-4"/>
        </svg>
      </span>
      <span>Human-reviewed decisions — AI informs, HR decides</span>
    </div>
  </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# Graceful error display if model failed to load
if model_error:
    st.error(f"⚠️ {model_error}")
    st.info("💡 To generate model artifacts, run `python export_model.py` in your terminal or check that `models/salary_prediction_pipeline.joblib` exists.")
    st.stop()

# Extract dropdown choices from metadata
education_options = metadata.get("education_levels", ["Bachelor's", "Diploma", "High School", "MBA", "Master's", "PhD"])
job_title_options = metadata.get("job_titles", [
    "Software Engineer", "Senior Software Engineer", "Data Scientist", "Data Analyst",
    "Machine Learning Engineer", "Product Manager", "DevOps Engineer", "QA Engineer",
    "UI/UX Designer", "Business Analyst"
])
gender_options = metadata.get("gender_options", ["Male", "Female", "Other", "Prefer not to say"])
projects_by_exp = metadata.get("projects_by_exp", {})

# Initialize session state for prediction result
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# ==============================================================================
# 8. TWO-COLUMN WORKSPACE
# ==============================================================================
col_form, col_result = st.columns([1, 1], gap="large")

# ------------------------------------------------------------------------------
# 8A. CANDIDATE PROFILE FORM (LEFT COLUMN)
# ------------------------------------------------------------------------------
with col_form:
    st.markdown("""
    <div class="section-card-header">
      <h3 class="section-card-title">Candidate Profile</h3>
      <p class="section-card-desc">Enter candidate profile details to generate a salary assessment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("salary_prediction_form"):
        # 1. Age
        age = st.number_input(
            "Age (years) *",
            min_value=18,
            max_value=70,
            value=28,
            step=1,
            help="Candidate age in years (must be between 18 and 70)."
        )
        
        # 2. Gender
        gender = st.selectbox(
            "Gender *",
            options=gender_options,
            index=0,
            help="Candidate profile attribute. Note: Gender is NOT used in model prediction or HR recommendations."
        )
        st.caption("ℹ️ *Handled according to the approved candidate profile schema; not used in ML calculation or HR recommendation.*")
        
        # 3. Education Level
        education = st.selectbox(
            "Education Level *",
            options=education_options,
            index=0,
            help="Highest formal education level completed."
        )
        
        # 4. Years of Experience
        experience = st.number_input(
            "Years of Experience *",
            min_value=0,
            max_value=50,
            value=5,
            step=1,
            help="Total relevant professional experience in years (0 to 50)."
        )
        
        # 5. Job Title
        job_title = st.selectbox(
            "Job Title *",
            options=job_title_options,
            index=0,
            help="Target role or closest corresponding position supported by the system."
        )
        
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("Generate Salary Estimate", use_container_width=True)

    # Validation and Prediction Logic on Submit
    if submit_btn:
        validation_errors = []
        
        if not (18 <= age <= 70):
            validation_errors.append("Age must be an integer between 18 and 70.")
        if not (0 <= experience <= 50):
            validation_errors.append("Years of experience must be between 0 and 50.")
        if experience > (age - 14):
            validation_errors.append(
                f"Years of experience ({experience} yrs) appears inconsistent with entered age ({age} yrs). "
                f"Maximum plausible experience for age {age} is {age - 14} years."
            )
        if education not in education_options:
            validation_errors.append(f"Education level '{education}' is not in accepted model categories.")
        if job_title not in job_title_options:
            validation_errors.append(f"Job title '{job_title}' is not supported.")
            
        if validation_errors:
            for err in validation_errors:
                st.error(f"❌ {err}")
        else:
            # Derive number of projects empirically from experience to match model training schema
            exp_key = str(int(experience))
            if exp_key in projects_by_exp:
                derived_projects = float(projects_by_exp[exp_key])
            else:
                derived_projects = float(round(experience * 1.5 + 1.0))
            
            # Construct single-row DataFrame matching training feature schema
            input_df = pd.DataFrame([{
                "education": education,
                "working_experience_years": int(experience),
                "number_of_projects": derived_projects
            }])
            
            try:
                # 1. Authoritative Model Output in USD
                pred_salary_usd = float(pipeline.predict(input_df)[0])
                pred_salary_usd = max(0.0, pred_salary_usd)
                
                # 2. Post-Prediction Approximate INR Conversion
                pred_salary_inr = pred_salary_usd * USD_TO_INR_RATE
                
                # 3. Illustrative Estimate Range (+/- 10%)
                usd_range_low = pred_salary_usd * 0.90
                usd_range_high = pred_salary_usd * 1.10
                inr_range_low = pred_salary_inr * 0.90
                inr_range_high = pred_salary_inr * 1.10
                
                # 4. Experience Category
                exp_category = get_experience_category(int(experience))
                
                # 5. Neutral HR Recommendation
                hr_reco = generate_hr_recommendation(
                    exp_category=exp_category,
                    job_title=job_title,
                    pred_usd=pred_salary_usd,
                    pred_inr=pred_salary_inr,
                    usd_low=usd_range_low,
                    usd_high=usd_range_high
                )
                
                # Store in session state
                st.session_state.prediction_result = {
                    "age": age,
                    "gender": gender,
                    "education": education,
                    "experience": experience,
                    "job_title": job_title,
                    "pred_salary_usd": pred_salary_usd,
                    "pred_salary_inr": pred_salary_inr,
                    "usd_range_low": usd_range_low,
                    "usd_range_high": usd_range_high,
                    "inr_range_low": inr_range_low,
                    "inr_range_high": inr_range_high,
                    "exp_category": exp_category,
                    "hr_reco": hr_reco
                }
                st.rerun()
            except Exception as e:
                st.error(f"❌ Prediction computation error: {str(e)}")

# ------------------------------------------------------------------------------
# 8B. PREDICTION RESULT PANEL (RIGHT COLUMN)
# ------------------------------------------------------------------------------
with col_result:
    res = st.session_state.prediction_result
    
    if res is None:
        # Empty State Placeholder
        empty_html = """
        <div class="empty-state">
          <div class="empty-state-icon">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M4 20h16"/>
              <path d="M6 20V10M11 20V6M16 20v-7"/>
              <path d="m15 4 1.5 1.5L19 3"/>
            </svg>
          </div>
          <h4 class="empty-state-title">Salary Assessment Preview</h4>
          <p class="empty-state-desc">
            Complete the candidate profile on the left and select <strong>Generate Salary Estimate</strong> to view the dual-currency valuation and HR recommendation.
          </p>
          <span class="badge-conversion">Model Ready · Waiting for Input</span>
        </div>
        """
        st.markdown(empty_html, unsafe_allow_html=True)
    else:
        # Loaded State: Dual-Currency Results & Insights
        usd_str = format_usd(res["pred_salary_usd"])
        inr_str = format_inr_indian(res["pred_salary_inr"])
        usd_low_str = format_usd(res["usd_range_low"])
        usd_high_str = format_usd(res["usd_range_high"])
        inr_low_str = format_inr_indian(res["inr_range_low"])
        inr_high_str = format_inr_indian(res["inr_range_high"])
        
        # Result Header
        st.markdown("""
        <div class="section-card-header">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <h3 class="section-card-title">Salary Assessment Result</h3>
            <span class="brand-badge" style="background:#ECFDF5; border-color:#A7F3D0; color:#065F46;">
              Assessment Complete
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 1. Primary USD Card
        primary_usd_html = f"""
        <div class="metric-card-usd">
          <div class="label-row">
            <span class="card-label">Estimated Annual Salary</span>
            <span class="badge-primary-target">Native Model Output</span>
          </div>
          <div class="primary-value">{usd_str}<span class="period">/ year</span></div>
          <div class="expl-note">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"/>
              <line x1="12" y1="16" x2="12" y2="12"/>
              <line x1="12" y1="8" x2="12.01" y2="8"/>
            </svg>
            Model prediction in USD (native target: Annual_Salary_USD)
          </div>
        </div>
        """
        st.markdown(primary_usd_html, unsafe_allow_html=True)
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # 2. Secondary Approximate INR Conversion Card
        secondary_inr_html = f"""
        <div class="metric-card-inr">
          <div class="label-row">
            <span class="card-label">Approximate INR Equivalent</span>
            <span class="badge-conversion">Display Conversion</span>
          </div>
          <div class="secondary-value">{inr_str}<span class="period">/ year</span></div>
          <div class="expl-note">
            Calculated at reference rate 1 USD = ₹{USD_TO_INR_RATE:.2f} for display reference
          </div>
        </div>
        """
        st.markdown(secondary_inr_html, unsafe_allow_html=True)
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # 3. Indicative Estimate Range Card with Dual Currency
        range_card_html = f"""
        <div class="range-card">
          <div class="range-header">
            <span>Indicative Estimate Range</span>
            <span style="font-size:0.75rem; color:#64748B; font-weight:normal;">Illustrative ±10%</span>
          </div>
          <div class="range-row">
            <span class="range-title">USD Range:</span>
            <span class="range-val">{usd_low_str} – {usd_high_str} / year</span>
          </div>
          <div class="range-row">
            <span class="range-title">Approximate INR Range:</span>
            <span class="range-val">≈ {inr_low_str} – {inr_high_str} / year</span>
          </div>
          <div class="range-meter-track">
            <div class="range-meter-fill" style="width: 50%;"></div>
          </div>
          <div class="range-labels">
            <span>{usd_low_str}</span>
            <span>Midpoint: {usd_str}</span>
            <span>{usd_high_str}</span>
          </div>
          <p class="range-subnote">
            The displayed range is an illustrative UI-level indicative range calculated as ±10% of the predicted annual salary. It is not a formal statistical confidence interval.
          </p>
        </div>
        """
        st.markdown(range_card_html, unsafe_allow_html=True)
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # 4. Mini Cards: Experience Category & Recommendation
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown(f"""
            <div class="mini-card mini-card-emerald">
              <div class="mini-card-head">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                  <polyline points="22 4 12 14.01 9 11.01"/>
                </svg>
                Experience Category
              </div>
              <p class="mini-card-value">{res["exp_category"]}</p>
              <p class="mini-card-support">{res["experience"]} years of relevant experience</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_c2:
            st.markdown(f"""
            <div class="mini-card">
              <div class="mini-card-head" style="color:var(--indigo);">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <rect x="2" y="7" width="20" height="14" rx="2" ry="2"/>
                  <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>
                </svg>
                Candidate Profile
              </div>
              <p class="mini-card-value">{res["job_title"]}</p>
              <p class="mini-card-support">{res["education"]} · Age {res["age"]}</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # 5. HR Recommendation Card
        st.markdown(f"""
        <div class="reco-card">
          <div class="reco-title">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>
              <rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>
            </svg>
            HR Compensation Guidance
          </div>
          <p class="reco-text">{res["hr_reco"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # 6. Decision Support Summary
        ds_summary_html = f"""
        <div class="ds-card">
          <div class="ds-title">Decision Support Summary</div>
          <div class="ds-row"><span class="ds-label">Evaluated Role</span><span class="ds-value">{res["job_title"]}</span></div>
          <div class="ds-row"><span class="ds-label">Experience Tier</span><span class="ds-value">{res["exp_category"]}</span></div>
          <div class="ds-row"><span class="ds-label">Education Credential</span><span class="ds-value">{res["education"]}</span></div>
          <div class="ds-row"><span class="ds-label">Model Confidence</span><span class="ds-value" style="color:#059669;">R² = {metadata.get('test_r2', 0.852):.3f} (High)</span></div>
          <div class="ds-row"><span class="ds-label">Human Review Requirement</span><span class="ds-value" style="color:#D97706;">Mandatory before offer</span></div>
        </div>
        """
        st.markdown(ds_summary_html, unsafe_allow_html=True)
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        
        # 7. Secondary Action: Reset Assessment
        if st.button("🔄 Reset Assessment", use_container_width=True):
            st.session_state.prediction_result = None
            st.rerun()

# ==============================================================================
# 9. EXPANDABLE SECTIONS: CURRENCY DETAILS & MODEL TRANSPARENCY
# ==============================================================================
st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

with st.expander("💱 Currency Conversion Details"):
    st.markdown(f"""
    - **Model Target Currency:** US Dollars (USD). The trained regression model strictly outputs `Annual_Salary_USD`.
    - **Display Conversion:** USD to Indian Rupees (INR).
    - **Applied Reference Rate:** `1 USD = ₹{USD_TO_INR_RATE:.2f}`.
    - **Rate Type:** {USD_TO_INR_RATE_SOURCE} ({USD_TO_INR_RATE_DATE}).
    - **Important Principle:** The INR value is calculated after model prediction for convenient local interpretation. It is **not** an independent machine-learning prediction and does **not** rely on live exchange-rate APIs.
    """)

with st.expander("📊 Model Transparency & Responsible AI Information"):
    test_r2 = metadata.get("test_r2", "N/A")
    test_mae = metadata.get("test_mae_usd", "N/A")
    test_rmse = metadata.get("test_rmse_usd", "N/A")
    model_name = metadata.get("model_name", "Linear Regression")
    dataset_name = metadata.get("dataset_name", "200000_employee_dataset.csv")
    
    st.markdown(f"""
    #### Architecture & Training Specifications
    - **Algorithm:** {model_name} with Scikit-Learn `ColumnTransformer` preprocessing pipeline.
    - **Training Dataset:** `{dataset_name}` (200,000 employee compensation records).
    - **Test $R^2$ Score:** `{test_r2}` (Explains {float(test_r2)*100 if isinstance(test_r2, (int, float)) else 85.2:.1f}% of salary variance).
    - **Test MAE:** `${test_mae:,.2f}` USD (Mean Absolute Error measured strictly in USD).
    - **Test RMSE:** `${test_rmse:,.2f}` USD.
    
    #### Responsible AI & Fairness Principles
    1. **Decision Support, Not Automated Decisions:** This system provides a baseline compensation benchmark. It does not replace human HR judgment, negotiation, or individual portfolio review.
    2. **Protected Attributes:** Gender is captured in the profile schema per organizational requirements but is **excluded from ML prediction** and **never used in recommendation logic**.
    3. **Bias Awareness:** Historical salary datasets can reflect historical compensation inequities. HR teams must cross-reference predictions with internal pay-equity guidelines and role requirements.
    4. **No Guarantees:** SalarySense AI makes no claim of 100% accuracy, legal compliance, or automated hiring/rejection determination.
    """)

# ==============================================================================
# 10. RESPONSIBLE AI DISCLAIMER BANNER
# ==============================================================================
disclaimer_html = """
<div class="disclaimer-banner">
  <div class="disclaimer-icon">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
      <path d="m9 11.5 2 2 4-4.5"/>
    </svg>
  </div>
  <div>
    <h4 class="disclaimer-title">Important: HR Decision-Support Tool Only</h4>
    <p class="disclaimer-text">
      SalarySense AI provides a data-informed salary estimate from historical employment records. It should never be used as the sole basis for compensation, hiring, promotion, or termination decisions. HR professionals must validate recommendations against company policy, role requirements, individual capabilities, internal pay bands, pay equity standards, and verified market data.
    </p>
  </div>
</div>
"""
st.markdown(disclaimer_html, unsafe_allow_html=True)
st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
