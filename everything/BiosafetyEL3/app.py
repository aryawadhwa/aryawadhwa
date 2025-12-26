
import streamlit as st
import torch
import numpy as np
import io

from src.data_processor import GenomicDataProcessor
from src.vae_model import GenomicVAE

# Page config
st.set_page_config(
    page_title="Biosafety Sequence Screener",
    page_icon="🧬",
    layout="wide"
)

# Title and introduction
st.title("🧬 Biosafety Sequence Screener")
st.markdown("""
This tool uses a Variational Autoencoder (VAE) to screen biological sequences for potential novelty.
High reconstruction error suggests the sequence is "out-of-distribution" relative to the training baseline.
""")

@st.cache_resource
def load_model(model_path="models/vae_best.pth", input_dim=8437):
    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Initialize model
    model = GenomicVAE(input_dim=input_dim, latent_dim=96)
    
    try:
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error(f"Model file not found at `{model_path}`. Please train the model first.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def screen_sequences(sequences, model, device):
    processor = GenomicDataProcessor(k_mers=[3, 4])
    
    results = []
    errors = []
    
    progress_bar = st.progress(0)
    
    for i, seq_record in enumerate(sequences):
        seq = str(seq_record.seq).upper()
        if len(seq) < 50:
            results.append({
                "ID": seq_record.id,
                "Status": "SKIPPED (Too short)",
                "Error": 0.0,
                "Sequence": seq[:50] + "..." if len(seq) > 50 else seq
            })
            continue
            
        # Extract features
        features = processor.extract_features(seq)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            recon, _, _ = model(features_tensor)
            # MSE per feature
            # shape: (1, input_dim)
            diff = features_tensor - recon
            mse = torch.mean(diff ** 2).item()
        
        errors.append(mse)
        results.append({
            "ID": seq_record.id,
            "Error": mse,
            "Sequence": seq[:50] + "..." if len(seq)>50 else seq
        })
        
        progress_bar.progress((i + 1) / len(sequences))
    
    return results, errors

# Sidebar controls
st.sidebar.header("Configuration")
model_path = st.sidebar.text_input("Model Path", "models/vae_best_fast.pth")
threshold_std = st.sidebar.slider("Threshold (Std Devs)", 1.0, 5.0, 3.0)

# Main interface
st.subheader("Input Sequences")
input_option = st.radio("Input method:", ["Upload FASTA File", "Paste FASTA Sequence"])

sequences = []

if input_option == "Upload FASTA File":
    uploaded_file = st.file_uploader("Choose a FASTA file", type=["fasta", "fa"])
    if uploaded_file is not None:
        # Save to temp file or handle in-memory
        # Biopython SeqIO needs a handle
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        try:
           # We need to reuse the load_seqs provided logic or simple SeqIO
           # But load_seqs takes a path. Let's just use SeqIO directly here for simplicity if allowed,
           # or write to temp. The prompt code has `load_seqs` in `train2.py` (which I haven't read yet, only assumed exists from `screen.py` import)
           # Ah, `screen.py` imports `screen_fasta` from `train2`.
           # I'll use Biopython directly to parse the stringIO
           from Bio import SeqIO
           sequences = list(SeqIO.parse(stringio, "fasta"))
           st.success(f"Loaded {len(sequences)} sequences.")
        except Exception as e:
            st.error(f"Error parsing FASTA: {e}")

else:
    fasta_text = st.text_area("Paste FASTA content here", height=200, placeholder=">seq1\nMVLSPADKTN...")
    if fasta_text:
        from Bio import SeqIO
        stringio = io.StringIO(fasta_text)
        try:
            sequences = list(SeqIO.parse(stringio, "fasta"))
            st.info(f"Parsed {len(sequences)} sequences.")
        except Exception as e:
            st.error(f"Error parsing text: {e}")

# Run screening
if sequences:
    if st.button("Screen Sequences"):
        model, device = load_model(model_path)
        
        if model:
            with st.spinner("Screening sequences..."):
                results, errors = screen_sequences(sequences, model, device)
            
            # Calculate threshold
            if errors:
                err_mean = np.mean(errors)
                err_std = np.std(errors)
                threshold = err_mean + (threshold_std * err_std)
                
                st.subheader("Results Analysis")
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean Error", f"{err_mean:.6f}")
                col2.metric("Threshold", f"{threshold:.6f}")
                col3.metric("Max Error", f"{np.max(errors):.6f}")
                
                # Assign status
                threat_count = 0
                for r in results:
                    if "Status" not in r: # if not skipped
                        is_threat = r["Error"] > threshold
                        r["Status"] = "⚠️ THREAT" if is_threat else "✅ Safe"
                        if is_threat: threat_count += 1
                
                if threat_count > 0:
                    st.warning(f"Found {threat_count} potential threats!")
                else:
                    st.success("No threats detected based on current threshold.")
                
                st.dataframe(results)
                
                # Visualization (if more than 1 sequence)
                if len(errors) > 1:
                    st.line_chart(errors)
                    st.caption("Reconstruction error per sequence")
            else:
                st.warning("No valid sequences to screen (all too short?).")

