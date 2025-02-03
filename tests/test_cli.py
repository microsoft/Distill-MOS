def test_cli():
    """Tests the command-line interface of the distillmos package on sample data."""

    import subprocess
    import os
    import requests

    DATA_URL = "https://github.com/QxLabIreland/datasets/raw/597fbf9b60efe555c1f7180e48a508394d817f73/genspeech/Genspeech/LPCNet_listening_test/mfall/dir3/"
    LOCAL_DATA_DIR = "./test_data"
    OUTPUT_FILE = "./distillmos_inference.csv"
    EXPECTED_SCORES = {
        "lpcnq.wav": 3.29,
        "lpcnu.wav": 4.12,
        "melp.wav": 3.09,
        "opus.wav": 4.05,
        "ref.wav": 4.55,
        "speex.wav": 1.47,
    }
    TOLERANCE = 0.005

    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

    # download the files into LOCAL_DATA_DIR
    with requests.Session() as s:
        for filename in EXPECTED_SCORES:
            with open(os.path.join(LOCAL_DATA_DIR, filename), "wb") as f:
                response = s.get(DATA_URL + filename)
                f.write(response.content)
    # Check if the input files exist
    assert all(
        os.path.exists(os.path.join(LOCAL_DATA_DIR, filename))
        for filename in EXPECTED_SCORES
    ), "Input files do not exist"

    # Make sure the output file does not exist
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    # Run CLI command
    subprocess.run(["distillmos", LOCAL_DATA_DIR], check=True)

    # delete the input files
    for filename in EXPECTED_SCORES:
        os.remove(os.path.join(LOCAL_DATA_DIR, filename))
    os.rmdir(LOCAL_DATA_DIR)

    # Check if the output file exists
    assert os.path.exists(OUTPUT_FILE), "Output file does not exist"

    # Read and validate output
    with open(OUTPUT_FILE, "r") as f:
        lines = f.readlines()

    # delete the output file
    os.remove(OUTPUT_FILE)

    assert len(lines) > 1, "Output file is empty"

    differences = []
    for line in lines[1:]:  # Skip header
        filepath, score = line.strip().split(",")
        filename = os.path.basename(filepath)

        if filename in EXPECTED_SCORES:
            differences.append(abs(float(score) - EXPECTED_SCORES[filename]))

    assert len(differences) == len(
        EXPECTED_SCORES
    ), "Incorrect number of entries in output file"
    assert all(
        diff < TOLERANCE for diff in differences
    ), "Output scores deviate beyond tolerance"
