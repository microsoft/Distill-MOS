    
def test_cli():
    """Tests the command-line interface of the distillmos package on sample data."""
    
    import subprocess
    import os

    DATA_DIR = "./data/Genspeech_LPCNet_listening_test_mfall_dir3/"
    OUTPUT_FILE = "./distillmos_inference.csv"
    EXPECTED_SCORES = {
        "lpcnq.wav": 4.12,
        "lpcnu.wav": 3.0,
        "melp.wav": 3.09,
        "opus.wav": 4.05,
        "ref.wav": 4.55,
        "speex.wav": 1.47,
    }
    TOLERANCE = 0.005

    # Run CLI command
    subprocess.run(["distillmos", DATA_DIR], check=True)

    # Check if the output file exists
    assert os.path.exists(OUTPUT_FILE), "Output file does not exist"

    # Read and validate output
    with open(OUTPUT_FILE, "r") as f:
        lines = f.readlines()

    assert len(lines) > 1, "Output file is empty"

    differences = []
    for line in lines[1:]:  # Skip header
        filepath, score = line.strip().split(",")
        filename = os.path.basename(filepath)

        if filename in EXPECTED_SCORES:
            differences.append(abs(float(score) - EXPECTED_SCORES[filename]))

    assert len(differences) == len(EXPECTED_SCORES), "Incorrect number of entries in output file"
    assert all(diff < TOLERANCE for diff in differences), "Output scores deviate beyond tolerance"
