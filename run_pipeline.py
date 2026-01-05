import subprocess
import sys
import os

def run_step(step_name, command):
    print(f"\n{'='*60}")
    print(f"▶ Running: {step_name}")
    print(f"{'='*60}")

    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed at step: {step_name}")
        sys.exit(1)

    print(f"✅ Completed: {step_name}")

def main():
    steps = [
        (
            "Training & Experiments",
            "python training/experiment_runner.py"
        ),
        (
            "Confusion Matrix Analysis",
            "python training/confusion_analysis.py"
        ),
        (
            "Grad-CAM Comparison",
            "python training/gradcam_comparison.py"
        )
    ]

    for name, cmd in steps:
        run_step(name, cmd)

    print("\n🎉 Pipeline executed successfully!")
    print("📁 Check logs/ directory for results")

if __name__ == "__main__":
    main()
