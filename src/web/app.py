import streamlit as st
import os
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import yaml
import json
import time
import sys
import requests
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import run_pipeline, load_config
from src.utils.logging_setup import setup_logging

def display_pipeline_results(meta, interim_dir, processed_dir, key_suffix=""):
    """Display comprehensive pipeline results"""
    
    # Extract stage1 and stage2 data from metadata
    stage1_data = meta.get('stage1', {}) or {}
    stage2_data = meta.get('stage2', {}) or {}
    fastp_data = stage1_data.get('fastp', {}) or {}
    dada2_data = stage1_data.get('dada2', {}) or {}
    discovery_data = stage2_data.get('discovery', {}) or {}
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        if os.path.exists(fastp_data.get('html_report', '')):
            st.metric("📊 Quality Report", "Available")
        else:
            st.metric("📊 Quality Report", "Not found")
    with col2:
        if os.path.exists(dada2_data.get('summary', '')):
            dada2_summary_path = dada2_data.get('summary', '')
            if os.path.exists(dada2_summary_path):
                with open(dada2_summary_path, 'r') as f:
                    dada2_summary = json.load(f)
                st.metric("🧬 ASVs Found", dada2_summary.get('total_asvs', 0))
    with col3:
        if os.path.exists(dada2_data.get('asv_table', '')):
            asv_df = pd.read_csv(dada2_data['asv_table'])
            st.metric("📊 Total Reads", asv_df['count'].sum() if 'count' in asv_df.columns else 0)


st.set_page_config(page_title="DeepSea-AI Pipeline", layout="wide")
st.title("DeepSea-AI: Complete Pipeline (Stage 1 + Stage 2)")

CONFIG_PATH = 'config/pipeline.yaml'

# Check if we should use API mode
API_MODE = os.getenv('API_MODE', 'false').lower() == 'true'
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')

@st.cache_resource
def get_logger():
    return setup_logging()

logger = get_logger()


# API functions for backend integration
def display_pipeline_results(meta, interim_dir, processed_dir, key_suffix=""):
    """Display comprehensive pipeline results on the same page."""
    
    # Extract stage1 and stage2 data from metadata
    stage1_data = meta.get('stage1', {}) or {}
    stage2_data = meta.get('stage2', {}) or {}
    fastp_data = stage1_data.get('fastp', {}) or {}
    dada2_data = stage1_data.get('dada2', {}) or {}
    discovery_data = stage2_data.get('discovery', {}) or {}
    
    # Results Overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🧬 Processing Mode", meta.get('dada2', {}).get('mode', 'Unknown'))
    
    with col2:
        if os.path.exists(meta.get('dada2', {}).get('asv_table', '')):
            asv_df = pd.read_csv(meta['dada2']['asv_table'])
            st.metric("🔬 ASVs Detected", len(asv_df))
    
    with col3:
        if os.path.exists(meta.get('dada2', {}).get('asv_table', '')):
            asv_df = pd.read_csv(meta['dada2']['asv_table'])
            st.metric("📊 Total Reads", asv_df['count'].sum() if 'count' in asv_df.columns else 0)
    
    # Tabs for different result views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Processing Stats", "🧬 ASV Results", "🔍 Discovery Results", "📋 Quality Report", "📁 Files"])
    
    with tab1:
        st.subheader("Pipeline Processing Statistics")
        
        # Display fastp statistics
        fastp_json_path = fastp_data.get('json_report', '')
        if os.path.exists(fastp_json_path):
            with open(fastp_json_path, 'r') as f:
                fastp_json_data = json.load(f)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**📥 Input Statistics:**")
                if 'summary' in fastp_json_data:
                    summary = fastp_json_data['summary']
                    st.write(f"- Total reads: {summary.get('before_filtering', {}).get('total_reads', 'N/A'):,}")
                    st.write(f"- Total bases: {summary.get('before_filtering', {}).get('total_bases', 'N/A'):,}")
                    st.write(f"- Q20 rate: {summary.get('before_filtering', {}).get('q20_rate', 'N/A'):.2%}")
                    st.write(f"- Q30 rate: {summary.get('before_filtering', {}).get('q30_rate', 'N/A'):.2%}")
            
            with col2:
                st.write("**📤 Output Statistics:**")
                if 'summary' in fastp_json_data:
                    summary = fastp_json_data['summary']
                    st.write(f"- Filtered reads: {summary.get('after_filtering', {}).get('total_reads', 'N/A'):,}")
                    st.write(f"- Filtered bases: {summary.get('after_filtering', {}).get('total_bases', 'N/A'):,}")
                    st.write(f"- Q20 rate: {summary.get('after_filtering', {}).get('q20_rate', 'N/A'):.2%}")
                    st.write(f"- Q30 rate: {summary.get('after_filtering', {}).get('q30_rate', 'N/A'):.2%}")
        else:
            st.info("📋 FastP quality statistics not available")
        
        # Display DADA2 statistics
        dada2_summary_path = dada2_data.get('summary', '')
        if os.path.exists(dada2_summary_path):
            with open(dada2_summary_path, 'r') as f:
                dada2_summary_data = json.load(f)
            
            st.write("**🔬 DADA2 Results:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"- Total ASVs: {dada2_summary_data.get('total_asvs', 'N/A')}")
                st.write(f"- Total reads retained: {dada2_summary_data.get('total_reads', 'N/A')}")
            with col2:
                st.write(f"- Mean count per ASV: {dada2_summary_data.get('mean_count', 'N/A'):.2f}")
                st.write(f"- Median count per ASV: {dada2_summary_data.get('median_count', 'N/A')}")
        else:
            st.info("🔬 DADA2 processing statistics not available")
    with tab2:
        st.subheader("Amplicon Sequence Variants (ASVs)")
        
        asv_path = dada2_data.get('asv_table', '')
        if os.path.exists(asv_path):
            asv_df = pd.read_csv(asv_path)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write("**Top ASVs by abundance:**")
                display_df = asv_df.head(10).copy()
                display_df['sequence_preview'] = display_df['sequence'].str[:50] + '...'
                st.dataframe(display_df[['sequence_preview', 'count']], use_container_width=True)
            
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
        else:
            st.info("🧬 ASV table not available. Check if DADA2 completed successfully.")
    
    with tab3:
        st.subheader("🔍 Discovery Engine Results (Stage 2)")
        st.caption("Pipeline: fastp → dada2 → asv → embedding transformer + hdbscan → if failed → k-mer vectorization → umap + hdbscan")
        
        # Check if this is API mode and we have a job ID
        if API_MODE and 'current_job' in st.session_state and st.session_state.current_job:
            job_id = st.session_state.current_job['id']
            discovery_status = get_discovery_status_api(job_id)
            
            if discovery_status:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Stage 2 Enabled", "Yes" if discovery_status.get('stage2_enabled') else "No")
                with col2:
                    st.metric("✅ Discovery Complete", "Yes" if discovery_status.get('discovery_completed') else "No")
                with col3:
                    st.metric("📊 Results Available", "Yes" if discovery_status.get('has_discovery_results') else "No")
                
                if discovery_status.get('error_message'):
                    st.error(f"Discovery Error: {discovery_status['error_message']}")
                
                if discovery_status.get('has_discovery_results'):
                    discovery_results = get_discovery_results_api(job_id)
                    
                    if discovery_results:
                        # Overview metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🧬 Total Sequences", discovery_results['total_sequences'])
                        with col2:
                            st.metric("🎯 Clusters Found", discovery_results['clusters_found'])
                        with col3:
                            st.metric("🔍 Novel Taxa Groups", discovery_results['clusters_found'])
                        with col4:
                            st.metric("⚪ Noise Sequences", discovery_results['noise_sequences'])
                        
                        if discovery_results['clusters']:
                            st.write("### 🎯 Discovered Clusters")
                            
                            # Create cluster summary table
                            cluster_data = []
                            for cluster in discovery_results['clusters']:
                                cluster_data.append({
                                    'Cluster ID': cluster['cluster_id'],
                                    'Sequences': cluster['sequence_count'],
                                    'Total Reads': cluster['total_reads'],
                                    'Representative Sequence': cluster['representative_sequence'][:50] + '...' if cluster['representative_sequence'] else 'N/A'
                                })
                            
                            cluster_df = pd.DataFrame(cluster_data)
                            st.dataframe(cluster_df, use_container_width=True)
                            
                            # Cluster selection for detailed view
                            selected_cluster = st.selectbox(
                                "Select cluster for detailed analysis:",
                                options=[c['cluster_id'] for c in discovery_results['clusters']],
                                format_func=lambda x: f"Cluster {x}"
                            )
                            
                            if selected_cluster is not None:
                                cluster_details = get_cluster_details_api(job_id, selected_cluster)
                                if cluster_details:
                                    st.write(f"### 📊 Cluster {selected_cluster} Details")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Sequences in Cluster", cluster_details['sequence_count'])
                                    with col2:
                                        st.metric("Total Reads", cluster_details['total_reads'])
                                    with col3:
                                        avg_reads = cluster_details['total_reads'] / cluster_details['sequence_count']
                                        st.metric("Avg Reads/Sequence", f"{avg_reads:.1f}")
                                    
                                    # Show top sequences in cluster
                                    st.write("**Top sequences by abundance:**")
                                    seq_data = []
                                    for seq in cluster_details['sequences'][:10]:  # Top 10
                                        seq_data.append({
                                            'Sequence (50bp preview)': seq['sequence'][:50] + '...',
                                            'Full Length': seq['length'],
                                            'Read Count': seq['count']
                                        })
                                    
                                    if seq_data:
                                        seq_df = pd.DataFrame(seq_data)
                                        st.dataframe(seq_df, use_container_width=True)
                        
                        # Download button for discovery results
                        discovery_csv = download_discovery_results_api(job_id)
                        if discovery_csv:
                            st.download_button(
                                label="📥 Download Discovery Results",
                                data=discovery_csv,
                                file_name=f"discovery_results_job_{job_id}.csv",
                                mime="text/csv",
                                key=f"download_discovery_{job_id}"
                            )
                    else:
                        st.info("Discovery results processing failed. Please check the job logs.")
                else:
                    st.info("Discovery results not available. Stage 2 may not be enabled or processing may still be in progress.")
            else:
                st.warning("Unable to fetch discovery status. Please check API connection.")
        
        else:
            # Check for discovery results in direct mode using metadata structure
            discovery_file = None
            discovery_results_info = discovery_data.get('results_file', '')
            
            # Try multiple discovery file locations
            if discovery_results_info and os.path.exists(discovery_results_info):
                discovery_file = discovery_results_info
            elif os.path.exists(os.path.join(processed_dir, "discovery_engine_results.csv")):
                discovery_file = os.path.join(processed_dir, "discovery_engine_results.csv")
            
            if discovery_file and os.path.exists(discovery_file):
                st.success("🎯 Discovery results found!")
                
                discovery_df = pd.read_csv(discovery_file)
                
                # Check if embedding method column exists (new format)
                has_embedding_method = 'embedding_method' in discovery_df.columns
                
                # Get embedding method from discovery results or metadata
                embedding_method = "Unknown"
                if has_embedding_method and not discovery_df.empty:
                    embedding_method = discovery_df['embedding_method'].iloc[0]
                elif 'embedding_method' in discovery_data:
                    embedding_method = discovery_data['embedding_method']
                
                # Overview metrics
                total_sequences = len(discovery_df)
                clusters_found = len(discovery_df[discovery_df['cluster_id'] != -1]['cluster_id'].unique())
                noise_sequences = len(discovery_df[discovery_df['cluster_id'] == -1])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🧬 Total Sequences", total_sequences)
                with col2:
                    st.metric("🎯 Clusters Found", clusters_found)
                with col3:
                    st.metric("⚪ Noise Sequences", noise_sequences)
                with col4:
                    # Display embedding method
                    method_display = {
                        "dnabert": "🤖 DNABERT-S (768D)",
                        "kmer_umap": "🧬 K-mer+UMAP",
                        "kmer": "🔤 K-mer (256D)"
                    }.get(embedding_method, f"🔬 {embedding_method.upper()}" if embedding_method != "Unknown" else "❓ Unknown")
                    st.metric("🔬 Processing Mode", method_display)
                # Pipeline flow status
                st.write("### 🔄 Pipeline Execution Path")
                
                if embedding_method == "dnabert":
                    st.success("✅ **Primary Path**: DNABERT-S transformer → HDBSCAN clustering")
                    st.info("🎯 Used 768-dimensional transformer embeddings from zhihan1996/DNABERT-S model")
                elif embedding_method == "kmer_umap":
                    st.warning("🔄 **Fallback Path**: K-mer vectorization → UMAP → HDBSCAN clustering")
                    st.info("💡 DNABERT-S failed (likely CPU incompatibility), automatically used enhanced k-mer fallback")
                elif embedding_method == "kmer":
                    st.info("🔤 **K-mer Path**: K-mer vectorization → HDBSCAN clustering")
                    st.info("💡 Used 256-dimensional k-mer frequency vectors")
                else:
                    st.info(f"📊 **Method**: {embedding_method}")
                
                if clusters_found > 0:
                    # Cluster summary
                    cluster_summary = discovery_df[discovery_df['cluster_id'] != -1].groupby('cluster_id').agg({
                        'sequence': 'count',
                        'count': 'sum'
                    }).rename(columns={'sequence': 'seq_count', 'count': 'total_reads'})
                    
                    st.write("### 🎯 Discovered Taxonomic Clusters")
                    
                    # Add cluster analysis
                    cluster_summary['avg_reads_per_seq'] = cluster_summary['total_reads'] / cluster_summary['seq_count']
                    cluster_summary = cluster_summary.round(1)
                    
                    st.dataframe(cluster_summary, use_container_width=True)
                    
                    # Cluster interpretation
                    st.write("**🔬 Biological Interpretation:**")
                    st.write("- Each cluster represents a potentially novel taxonomic group")
                    st.write("- Higher read counts suggest more abundant taxa in the environment")
                    st.write("- Noise sequences (-1) may be rare variants or sequencing artifacts")
                
                # Download button
                csv_data = discovery_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Discovery Results",
                    data=csv_data,
                    file_name="discovery_results.csv",
                    mime="text/csv",
                    key=f"download_discovery_direct{key_suffix}"
                )
            else:
                st.info("🔍 No discovery results found. Stage 2 (Discovery Engine) may not be enabled or may not have completed yet.")
    
    with tab4:
        st.subheader("Quality Control Report")
        
        # Display fastp HTML report content (extract key sections)
        fastp_html_path = fastp_data.get('html_report', '')
        if os.path.exists(fastp_html_path):
            st.write("**📊 Quality Control Summary**")
            st.info("Full interactive HTML report available for download below.")
            
            # Extract and display key metrics from JSON
            fastp_json_path = fastp_data.get('json_report', '')
            if os.path.exists(fastp_json_path):
                with open(fastp_json_path, 'r') as f:
                    fastp_json_data = json.load(f)
                
                if 'filtering_result' in fastp_json_data:
                    filtering = fastp_json_data['filtering_result']
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
        else:
            st.info("📋 Quality control report not available. Check if FastP completed successfully.")
    
    with tab5:
        st.subheader("Output Files & Metadata")
        
        st.write("**📁 Generated Files:**")
        file_info = []
        
        # Check stage1 files
        for key, filepath in fastp_data.items():
            if isinstance(filepath, str) and os.path.exists(filepath):
                size = os.path.getsize(filepath) / 1024 / 1024  # MB
                file_info.append({
                    'Category': 'FASTP',
                    'Type': key,
                    'File Path': filepath,
                    'Size (MB)': f"{size:.2f}"
                })
        
        for key, filepath in dada2_data.items():
            if isinstance(filepath, str) and os.path.exists(filepath):
                size = os.path.getsize(filepath) / 1024 / 1024  # MB
                file_info.append({
                    'Category': 'DADA2',
                    'Type': key,
                    'File Path': filepath,
                    'Size (MB)': f"{size:.2f}"
                })
        
        # Check stage2 files
        for key, filepath in discovery_data.items():
            if isinstance(filepath, str) and os.path.exists(filepath):
                size = os.path.getsize(filepath) / 1024 / 1024  # MB
                file_info.append({
                    'Category': 'DISCOVERY',
                    'Type': key,
                    'File Path': filepath,
                    'Size (MB)': f"{size:.2f}"
                })
        
        if file_info:
            files_df = pd.DataFrame(file_info)
            st.dataframe(files_df, use_container_width=True)
        else:
            st.info("📁 No output files found in metadata.")
        
        st.write("**🔧 Complete Metadata:**")
        st.json(meta)

def create_job_api():
    """Create a new job via API."""
    response = requests.post(f"{BACKEND_URL}/jobs", json={
        "name": f"Streamlit Job {int(time.time())}",
        "description": "Job created from Streamlit interface"
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

def download_asv_data_api(job_id):
    """Download ASV metadata via API."""
    response = requests.get(f"{BACKEND_URL}/jobs/{job_id}/metadata")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to download ASV data: {response.text}")
        return None

def get_discovery_status_api(job_id):
    """Get discovery engine status via API."""
    response = requests.get(f"{BACKEND_URL}/jobs/{job_id}/discovery/status")
    if response.status_code == 200:
        return response.json()
    else:
        return None

def get_discovery_results_api(job_id):
    """Get discovery engine results via API."""
    response = requests.get(f"{BACKEND_URL}/jobs/{job_id}/discovery/results")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to get discovery results: {response.text}")
        return None

def get_cluster_details_api(job_id, cluster_id):
    """Get detailed cluster information via API."""
    response = requests.get(f"{BACKEND_URL}/jobs/{job_id}/discovery/clusters/{cluster_id}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to get cluster details: {response.text}")
        return None

def download_discovery_results_api(job_id):
    """Download discovery results CSV via API."""
    response = requests.get(f"{BACKEND_URL}/jobs/{job_id}/discovery/download")
    if response.status_code == 200:
        return response.content
    else:
        st.error(f"Failed to download discovery results: {response.text}")
        return None

# UI Layout
with st.sidebar:
    st.header("Configuration")
    
    if API_MODE:
        st.info("🔌 API Mode Enabled")
        st.text(f"Backend: {BACKEND_URL}")
    else:
        st.info("💻 Direct Mode")
    
    if API_MODE:
        create_job_btn = st.button("Create New Job", type='primary')
    else:
        run_button = st.button("Run Pipeline", type='primary')

st.markdown("Upload FASTQ files to process them through the complete DeepSea-AI pipeline: **Stage 1** (Quality Control + ASV Analysis) and **Stage 2** (AI-powered Discovery Engine for novel taxonomic groups).")

# File upload section
uploaded_files = st.file_uploader("FASTQ files", type=["fastq", "fq", "gz"], accept_multiple_files=True)

# API Mode workflow
if API_MODE:
    # Session state for job management
    if 'current_job' not in st.session_state:
        st.session_state.current_job = None
    if 'job_polling' not in st.session_state:
        st.session_state.job_polling = False

    if create_job_btn:
        job = create_job_api()
        if job:
            st.session_state.current_job = job
            st.session_state.job_polling = False
            st.success(f"✅ Created job #{job['id']}")

    if st.session_state.current_job:
        job = st.session_state.current_job
        st.subheader(f"Job #{job['id']}: {job['name']}")

        status_col, eta_col, actions_col = st.columns([1, 1, 1])
        with status_col:
            st.metric("Status", job.get('status', 'Unknown'))
            if job.get('created_at'):
                created_at = datetime.fromisoformat(job['created_at'])
                st.caption(f"Created {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            if job.get('updated_at'):
                updated_at = datetime.fromisoformat(job['updated_at'])
                st.caption(f"Updated {updated_at.strftime('%Y-%m-%d %H:%M:%S')}")

        with eta_col:
            if job.get('status') == 'RUNNING':
                progress_meta = job.get('meta', {}) or {}
                step = progress_meta.get('progress', {}).get('step', 'fastp')
                step_map = {
                    'fastp': ("Quality control", 0.2),
                    'dada2': ("Denoising", 0.6),
                    'kmer': ("Vectorizing", 0.85),
                    'complete': ("Finalizing", 1.0)
                }
                label, progress_val = step_map.get(step, ("Preparing", 0.1))
                st.progress(progress_val, text=f"🚀 {label}")
            elif job.get('status') == 'COMPLETED':
                st.success("✅ Finished")
            elif job.get('status') == 'FAILED':
                st.error("❌ Failed")
            else:
                st.info("⏳ Waiting to start")

        with actions_col:
            col_refresh, col_poll = st.columns(2)
            with col_refresh:
                if st.button("🔄", help="Refresh now"):
                    updated_job = get_job_status_api(job['id'])
                    if updated_job:
                        st.session_state.current_job = updated_job
                        st.rerun()
            with col_poll:
                if st.toggle("Auto-refresh", value=st.session_state.job_polling, key="poll_toggle"):
                    st.session_state.job_polling = True
                else:
                    st.session_state.job_polling = False

            if st.session_state.job_polling:
                with st.spinner("Polling job status..."):
                    time.sleep(2)
                    updated_job = get_job_status_api(job['id'])
                    if updated_job:
                        st.session_state.current_job = updated_job
                        st.rerun()

        if job.get('status') == 'PENDING':
            upload_col, run_col = st.columns(2)
            with upload_col:
                if uploaded_files and st.button("📤 Upload Files"):
                    upload_result = upload_files_api(job['id'], uploaded_files)
                    if upload_result:
                        st.success(f"✅ Uploaded {upload_result['count']} files")
            with run_col:
                if st.button("🚀 Start Processing"):
                    run_result = run_job_api(job['id'])
                    if run_result:
                        st.success("✅ Pipeline started")
                        st.session_state.job_polling = True
                        time.sleep(1)
                        st.rerun()

        if job.get('status') == 'RUNNING':
            st.info("🔄 Pipeline running. We refresh automatically when auto-refresh is enabled.")

        if job.get('status') == 'COMPLETED':
            st.success("✅ Pipeline completed successfully!")
            asv_data = download_asv_data_api(job['id'])
            if asv_data and asv_data.get('metadata', {}).get('dada2', {}).get('asv_table'):
                asv_table_path = asv_data['metadata']['dada2']['asv_table']
                if os.path.exists(asv_table_path):
                    asv_df = pd.read_csv(asv_table_path)
                    csv_data = asv_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download ASV table",
                        data=csv_data,
                        file_name=f"job_{job['id']}_asv_table.csv",
                        mime="text/csv",
                        key=f"download_api_asv_{job['id']}"
                    )

            if job.get('meta'):
                st.subheader("Pipeline Results")
                meta = job['meta']
                display_pipeline_results(meta, meta.get('interim_dir', ''), meta.get('processed_dir', ''), key_suffix=f"_{job['id']}")

        if job.get('status') == 'FAILED':
            st.error("❌ Pipeline failed. See error details below.")
            if job.get('error'):
                st.exception(RuntimeError(job['error']))

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
                    config_path=str(tmp_cfg_path)
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
