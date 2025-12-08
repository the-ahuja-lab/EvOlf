import subprocess
import os
import argparse

def run_mathfeature(input_fasta: str, output_prefix: str):
    # This is the path to scripts INSIDE the Docker container
    scripts_dir = "/app/methods" 


    print("Running Mappings-Protein (1/3)...")
    subprocess.run([
        "python",
        os.path.join(scripts_dir, "Mappings-Protein.py"),
        "-i", input_fasta,
        "-o", f"{output_prefix}_MF_02.csv",
        "-n", "1", "-r", "2"
    ], check=True)

    print("Running Mappings-Protein (2/3)...")
    subprocess.run([
        "python",
        os.path.join(scripts_dir, "Mappings-Protein.py"),
        "-i", input_fasta,
        "-o", f"{output_prefix}_MF_04.csv",
        "-n", "1", "-r", "4"
    ], check=True)
    
    print("Running Mappings-Protein (3/3)...")
    subprocess.run([
        "python",
        os.path.join(scripts_dir, "Mappings-Protein.py"),
        "-i", input_fasta,
        "-o", f"{output_prefix}_MF_06.csv",
        "-n", "1", "-r", "6"
    ], check=True)

    print("Running EntropyClass (1/2)...")
    subprocess.run([
        "python",
        os.path.join(scripts_dir, "EntropyClass.py"),
        "-i", input_fasta,
        "-o", f"{output_prefix}_MF_08.csv",
        "-l", "1", "-k", "10", "-e", "Shannon"
    ], check=True)

    print("Running EntropyClass (2/2)...")
    subprocess.run([
        "python",
        os.path.join(scripts_dir, "EntropyClass.py"),
        "-i", input_fasta,
        "-o", f"{output_prefix}_MF_09.csv",
        "-l", "1", "-k", "10", "-e", "Tsallis"
    ], check=True)
    
    print("Running ComplexNetworksClass...")
    subprocess.run([
        "python",
        os.path.join(scripts_dir, "ComplexNetworksClass.py"),
        "-i", input_fasta,
        "-o",  f"{output_prefix}_MF_10.csv",
        "-l", "1", "-k", "3"
    ], check=True)
    
    print("Running Kgap...")
    subprocess.run([
        "python",
        os.path.join(scripts_dir, "Kgap.py"),
        "-i", input_fasta,
        "-o",  f"{output_prefix}_MF_11.csv",
        "-l", "1", "-k", "3", "-bef", "1", "-aft", "2", "-seq", "3"
    ], check=True)
    
    print("MathFeature ran successfully")


parser = argparse.ArgumentParser(description="Math Featurizer")
parser.add_argument("--input", required=True, help="Input File Path")
parser.add_argument("--output_prefix", required=True, help="Output File Prefix")

args = parser.parse_args()

run_mathfeature(args.input, args.output_prefix)