import streamlit as st
import os
import requests
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import yaml
import json
import time
from src.pipeline import run_pipeline, load_config
from src.utils.logging_setup import setup_logging

st.set_page_config(page_title="DeepSea-AI Stage 1", layout="wide")
st.title("DeepSea-AI: Stage 1 Preprocessing & Vectorization")

CONFIG_PATH = 'config/pipeline.yaml'

# Check if we should use API mode
API_MODE = os.getenv('API_MODE', 'false').lower() == 'true'
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')

@st.cache_resource
def get_logger():
    return setup_logging()

logger = get_logger()

def display_pipeline_results(meta, interim_dir, processed_dir, key_suffix=""):
    """Display comprehensive pipeline results on the same page."""
    
    # Results Overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🧬 Processing Mode", meta.get('dada2', {}).get('mode', 'Unknown'))
    
    with col2:
        if os.path.exists(meta.get('dada2', {}).get('asv_table', '')):
            asv_df = pd.read_csv(meta['dada2']['asv_table'])
            st.metric("🔬 ASVs Detected", len(asv_df))
    
    with col3:
        if os.path.exists(meta.get('kmer', {}).get('vectors_csv', '')):
            kmer_df = pd.read_csv(meta['kmer']['vectors_csv'])
            st.metric("📊 K-mer Features", len(kmer_df.columns) - 2)  # Exclude sequence and count
    
    # Tabs for different result views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Processing Stats", "🧬 ASV Results", "📊 K-mer Vectors", "📋 Quality Report", "📁 Files"])
    
    with tab1:
        st.subheader("Pipeline Processing Statistics")
        
        # Display fastp statistics
        fastp_json_path = meta.get('fastp', {}).get('json_report', '')
        if os.path.exists(fastp_json_path):
            with open(fastp_json_path, 'r') as f:
                fastp_data = json.load(f)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**📥 Input Statistics:**")
                if 'summary' in fastp_data:
                    summary = fastp_data['summary']
                    st.write(f"- Total reads: {summary.get('before_filtering', {}).get('total_reads', 'N/A'):,}")
                    st.write(f"- Total bases: {summary.get('before_filtering', {}).get('total_bases', 'N/A'):,}")
                    st.write(f"- Q20 rate: {summary.get('before_filtering', {}).get('q20_rate', 'N/A'):.2%}")
                    st.write(f"- Q30 rate: {summary.get('before_filtering', {}).get('q30_rate', 'N/A'):.2%}")
            
            with col2:
                st.write("**📤 Output Statistics:**")
                if 'summary' in fastp_data:
                    summary = fastp_data['summary']
                    st.write(f"- Filtered reads: {summary.get('after_filtering', {}).get('total_reads', 'N/A'):,}")
                    st.write(f"- Filtered bases: {summary.get('after_filtering', {}).get('total_bases', 'N/A'):,}")
                    st.write(f"- Q20 rate: {summary.get('after_filtering', {}).get('q20_rate', 'N/A'):.2%}")
                    st.write(f"- Q30 rate: {summary.get('after_filtering', {}).get('q30_rate', 'N/A'):.2%}")
        
        # Display DADA2 statistics
        dada2_summary_path = meta.get('dada2', {}).get('summary', '')
        if os.path.exists(dada2_summary_path):
            with open(dada2_summary_path, 'r') as f:
                dada2_data = json.load(f)
            
            st.write("**🔬 DADA2 Results:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"- Total ASVs: {dada2_data.get('total_asvs', 'N/A')}")
                st.write(f"- Total reads retained: {dada2_data.get('total_reads', 'N/A')}")
            with col2:
                st.write(f"- Mean count per ASV: {dada2_data.get('mean_count', 'N/A'):.2f}")
                st.write(f"- Median count per ASV: {dada2_data.get('median_count', 'N/A')}")
    
    with tab2:
        st.subheader("Amplicon Sequence Variants (ASVs)")
        
        asv_path = meta.get('dada2', {}).get('asv_table', '')
        if os.path.exists(asv_path):
            asv_df = pd.read_csv(asv_path)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write("**Top ASVs by abundance:**")
                display_df = asv_df.head(10).copy()
                display_df['sequence_preview'] = display_df['sequence'].str[:50] + '...'
                st.dataframe(display_df[['sequence_preview', 'count']], width='stretch')
            
            with col2:
                st.write("**ASV Length Distribution:**")
                lengths = asv_df['sequence'].str.len()
                length_counts = lengths.value_counts().head(10)
                st.bar_chart(length_counts)
            
            # Download button for ASV table
            csv_data = asv_df.to_csv(index=False)
            st.download_button(
                label="📥 Download ASV Table",
                data=csv_data,
                file_name="asv_table.csv",
                mime="text/csv",
                key=f"download_asv_table{key_suffix}"
            )
    
    with tab3:
        st.subheader("K-mer Vector Analysis")
        
        kmer_path = meta.get('kmer', {}).get('vectors_csv', '')
        if os.path.exists(kmer_path):
            kmer_df = pd.read_csv(kmer_path)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write("**K-mer Vectors Preview:**")
                # Show first few rows and columns
                preview_df = kmer_df.iloc[:10, :10]  # First 10 rows, first 10 k-mer columns
                st.dataframe(preview_df, width='stretch')
            
            with col2:
                st.write("**Dataset Info:**")
                st.write(f"- Sequences: {len(kmer_df)}")
                st.write(f"- K-mer features: {len(kmer_df.columns) - 2}")
                st.write(f"- File size: {os.path.getsize(kmer_path) / 1024 / 1024:.1f} MB")
                
                # Sparsity analysis
                numeric_cols = kmer_df.select_dtypes(include=[float, int]).columns
                if len(numeric_cols) > 0:
                    sparsity = (kmer_df[numeric_cols] == 0).mean().mean()
                    st.write(f"- Sparsity: {sparsity:.1%}")
            
            # Download button for k-mer vectors
            csv_data = kmer_df.to_csv(index=False)
            st.download_button(
                label="📥 Download K-mer Vectors",
                data=csv_data,
                file_name="kmer_vectors.csv",
                mime="text/csv",
                key=f"download_kmer_vectors{key_suffix}"
            )
    
    with tab4:
        st.subheader("Quality Control Report")
        
        # Display fastp HTML report content (extract key sections)
        fastp_html_path = meta.get('fastp', {}).get('html_report', '')
        if os.path.exists(fastp_html_path):
            st.write("**📊 Quality Control Summary**")
            st.info("Full interactive HTML report available for download below.")
            
            # Extract and display key metrics from JSON
            fastp_json_path = meta.get('fastp', {}).get('json_report', '')
            if os.path.exists(fastp_json_path):
                with open(fastp_json_path, 'r') as f:
                    fastp_data = json.load(f)
                
                if 'filtering_result' in fastp_data:
                    filtering = fastp_data['filtering_result']
                    st.write("**Filtering Results:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"- Reads passed: {filtering.get('passed_filter_reads', 'N/A'):,}")
                        st.write(f"- Low quality: {filtering.get('low_quality_reads', 'N/A'):,}")
                    with col2:
                        st.write(f"- Too many N: {filtering.get('too_many_N_reads', 'N/A'):,}")
                        st.write(f"- Too short: {filtering.get('too_short_reads', 'N/A'):,}")
            
            # Download button for HTML report
            with open(fastp_html_path, 'rb') as f:
                html_data = f.read()
            st.download_button(
                label="📥 Download HTML Report",
                data=html_data,
                file_name="fastp_report.html",
                mime="text/html",
                key=f"download_fastp_html{key_suffix}"
            )
    
    with tab5:
        st.subheader("Output Files & Metadata")
        
        st.write("**📁 Generated Files:**")
        file_info = []
        
        # Check all output files
        for category, data in meta.items():
            if isinstance(data, dict):
                for key, filepath in data.items():
                    if isinstance(filepath, str) and os.path.exists(filepath):
                        size = os.path.getsize(filepath) / 1024 / 1024  # MB
                        file_info.append({
                            'Category': category.upper(),
                            'Type': key,
                            'File Path': filepath,
                            'Size (MB)': f"{size:.2f}"
                        })
        
        if file_info:
            files_df = pd.DataFrame(file_info)
            st.dataframe(files_df, width='stretch')
        
        st.write("**🔧 Complete Metadata:**")
        st.json(meta)

def create_job_api(k=6):
    """Create a new job via API."""
    response = requests.post(f"{BACKEND_URL}/jobs", json={
        "name": f"Streamlit Job {int(time.time())}",
        "description": "Job created from Streamlit interface",
        "kmer_k": k
    })
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to create job: {response.text}")
        return None

def upload_files_api(job_id, files):
    """Upload files to a job via API."""
    files_data = []
    for file in files:
        files_data.append(('files', (file.name, file.getvalue(), file.type)))
    
    response = requests.post(f"{BACKEND_URL}/jobs/{job_id}/files", files=files_data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to upload files: {response.text}")
        return None

def run_job_api(job_id):
    """Start pipeline execution via API."""
    response = requests.post(f"{BACKEND_URL}/jobs/{job_id}/run")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to start job: {response.text}")
        return None

def get_job_status_api(job_id):
    """Get job status via API."""
    response = requests.get(f"{BACKEND_URL}/jobs/{job_id}")
    if response.status_code == 200:
        return response.json()
    else:
        return None

def download_vectors_api(job_id):
    """Download vectors CSV via API."""
    response = requests.get(f"{BACKEND_URL}/jobs/{job_id}/vectors")
    if response.status_code == 200:
        return response.content
    else:
        st.error(f"Failed to download vectors: {response.text}")
        return None

# UI Layout
with st.sidebar:
    st.header("Configuration")
    k = st.number_input("k-mer length", min_value=3, max_value=8, value=6, step=1)
    
    if API_MODE:
        st.info("🔌 API Mode Enabled")
        st.text(f"Backend: {BACKEND_URL}")
    else:
        st.info("💻 Direct Mode")
    
    if API_MODE:
        create_job_btn = st.button("Create New Job", type='primary')
    else:
        run_button = st.button("Run Pipeline", type='primary')

st.markdown("Upload FASTQ files to process them through the DeepSea-AI Stage 1 pipeline.")

# File upload section
uploaded_files = st.file_uploader("FASTQ files", type=["fastq", "fq", "gz"], accept_multiple_files=True)

# API Mode workflow
if API_MODE:
    # Session state for job management
    if 'current_job' not in st.session_state:
        st.session_state.current_job = None
    
    if create_job_btn:
        job = create_job_api(k)
        if job:
            st.session_state.current_job = job
            st.success(f"✅ Created job #{job['id']}")
    
    if st.session_state.current_job:
        job = st.session_state.current_job
        st.subheader(f"Job #{job['id']}: {job['name']}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", job.get('status', 'Unknown'))
        with col2:
            st.metric("k-mer length", job.get('kmer_k', k))
        with col3:
            if st.button("🔄 Refresh Status"):
                updated_job = get_job_status_api(job['id'])
                if updated_job:
                    st.session_state.current_job = updated_job
                    st.rerun()
        
        # File upload for current job
        if uploaded_files and job.get('status') == 'PENDING':
            if st.button("📤 Upload Files to Job"):
                upload_result = upload_files_api(job['id'], uploaded_files)
                if upload_result:
                    st.success(f"✅ Uploaded {upload_result['count']} files")
        
        # Run job
        if job.get('status') == 'PENDING' and st.button("🚀 Start Processing"):
            run_result = run_job_api(job['id'])
            if run_result:
                st.success("✅ Pipeline started!")
                time.sleep(1)
                st.rerun()
        
        # Show progress for running jobs
        if job.get('status') == 'RUNNING':
            st.info("🔄 Pipeline is running... Please refresh to check status.")
            if st.button("Check Progress", key="progress_check"):
                st.rerun()
        
        # Show results for completed jobs
        if job.get('status') == 'COMPLETED':
            st.success("✅ Pipeline completed successfully!")
            
            if st.button("📊 Download k-mer Vectors"):
                vectors_data = download_vectors_api(job['id'])
                if vectors_data:
                    st.download_button(
                        label="💾 Download CSV",
                        data=vectors_data,
                        file_name=f"job_{job['id']}_kmer_vectors.csv",
                        mime="text/csv",
                        key=f"download_api_vectors_{job['id']}"
                    )
            
            # Show metadata
            if job.get('meta'):
                st.subheader("Pipeline Results")
                st.json(job['meta'])
                
                # Show k-mer vectors preview
                vectors_csv_path = job.get('meta', {}).get('kmer', {}).get('vectors_csv')
                if vectors_csv_path:
                    try:
                        # Note: In API mode, we'd need to serve this file or embed preview in metadata
                        st.info("💡 Use 'Download k-mer Vectors' button to get the full CSV file.")
                    except Exception as e:
                        st.warning("Preview not available in API mode. Use download button.")
        
        # Show errors for failed jobs
        if job.get('status') == 'FAILED':
            st.error("❌ Pipeline failed!")
            if job.get('error'):
                st.error(f"Error: {job['error']}")

# Direct Mode workflow (original)
else:
    if uploaded_files:
        # Create a unique temporary directory for ONLY uploaded files
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        target_dir = Path(tempfile.gettempdir()) / f'uploaded_files_{unique_id}'
        target_dir.mkdir(parents=True, exist_ok=True)
        
        st.info(f"📁 Processing ONLY your uploaded files in: {target_dir}")
        
        for uf in uploaded_files:
            dest = target_dir / uf.name
            with open(dest, 'wb') as f:
                f.write(uf.read())
            st.write(f"✅ Uploaded: {uf.name} ({uf.size} bytes)")
        
        st.success(f"Uploaded {len(uploaded_files)} file(s) to isolated directory.")

    if run_button:
        if not uploaded_files:
            st.error("❌ Please upload files first!")
            st.stop()
            
        cfg = load_config(CONFIG_PATH)
        cfg['kmer']['k'] = k
        # Write a temp override file
        tmp_cfg_path = Path(tempfile.gettempdir()) / 'pipeline_override.yaml'
        with open(tmp_cfg_path, 'w') as f:
            yaml.safe_dump(cfg, f)
        with st.spinner("Running pipeline... this may take a few minutes depending on data size"):
            try:
                # Create temporary directories for processing
                interim_dir = Path(tempfile.gettempdir()) / 'interim'
                processed_dir = Path(tempfile.gettempdir()) / 'processed'
                interim_dir.mkdir(exist_ok=True)
                processed_dir.mkdir(exist_ok=True)
                
                # Process ONLY the uploaded files directory
                meta = run_pipeline(
                    raw_dir=str(target_dir),  # Use uploaded files directory, NOT data/raw/fastq_dataset
                    interim_dir=str(interim_dir),
                    processed_dir=str(processed_dir),
                    config_path=str(tmp_cfg_path),
                    k=k
                )
                st.success("🎉 Pipeline completed successfully!")
                
                # Store results in session state for persistence
                st.session_state['pipeline_results'] = meta
                st.session_state['interim_dir'] = str(interim_dir)
                st.session_state['processed_dir'] = str(processed_dir)
                st.session_state['current_run'] = True  # Flag to prevent showing as "previous"
                
                # Display comprehensive results
                display_pipeline_results(meta, interim_dir, processed_dir)
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                logger.exception(e)

# Check if we have previous results to display (only if not from current run)
if 'pipeline_results' in st.session_state and not st.session_state.get('current_run', False):
    st.markdown("---")
    st.subheader("📊 Previous Results")
    display_pipeline_results(
        st.session_state['pipeline_results'],
        st.session_state['interim_dir'], 
        st.session_state['processed_dir'],
        key_suffix="_previous"
    )

# Reset the current_run flag for next time
if 'current_run' in st.session_state:
    st.session_state['current_run'] = False

st.markdown("---")
st.caption("DeepSea-AI Stage 1 Module")
