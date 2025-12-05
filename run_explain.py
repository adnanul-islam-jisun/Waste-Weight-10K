#!/usr/bin/env python3
"""
Interactive Explanation Pipeline Runner
Provides a user-friendly interface to run the explanation pipeline with various options.
"""

import os
import sys
import subprocess
from typing import Optional


def print_header():
    """Print welcome header."""
    print("\n" + "="*70)
    print("🔬 STAGE 2: EXPLANATION PIPELINE - INTERACTIVE RUNNER")
    print("="*70)


def print_menu(title: str, options: list) -> int:
    """Print a menu and get user selection."""
    print(f"\n📋 {title}")
    print("-" * 50)
    for i, option in enumerate(options, 1):
        print(f"  [{i}] {option}")
    print("-" * 50)
    
    while True:
        try:
            choice = input("Enter your choice (number): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(options):
                return choice_num
            print(f"❌ Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            sys.exit(0)


def get_input(prompt: str, default: Optional[str] = None) -> str:
    """Get user input with optional default value."""
    try:
        if default:
            user_input = input(f"{prompt} [{default}]: ").strip()
            return user_input if user_input else default
        else:
            return input(f"{prompt}: ").strip()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting...")
        sys.exit(0)


def find_checkpoints() -> list:
    """Find available checkpoints."""
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return checkpoints


def main():
    print_header()
    
    # =========================================================================
    # Step 1: Select LLM Mode
    # =========================================================================
    llm_options = [
        "🚫 No LLM (Metrics only - Fast)",
        "📝 Template-based explanations (No LLM needed)",
        "🤖 Full LLM explanations (Requires Llama 3.1 access)"
    ]
    llm_choice = print_menu("Select Explanation Mode", llm_options)
    
    use_llm = True
    use_template_only = False
    
    if llm_choice == 1:
        use_llm = False
        print("✓ Mode: Metrics only (no explanations)")
    elif llm_choice == 2:
        use_template_only = True
        print("✓ Mode: Template-based explanations")
    else:
        print("✓ Mode: Full LLM explanations")
    
    # =========================================================================
    # Step 2: Select Data Mode
    # =========================================================================
    data_options = [
        "🖼️  Single Image (provide image path)",
        "📊 Batch from Test Set (specify number of samples)",
        "📈 Full Test Set Evaluation"
    ]
    data_choice = print_menu("Select Data Mode", data_options)
    
    # =========================================================================
    # Step 3: Get additional parameters based on mode
    # =========================================================================
    image_path = None
    category = None
    numerical = None
    actual_weight = None
    max_samples = 10
    
    if data_choice == 1:
        # Single image mode
        print("\n📸 Single Image Mode")
        print("-" * 50)
        image_path = get_input("Enter image path")
        
        if not os.path.exists(image_path):
            print(f"⚠️  Warning: Image path '{image_path}' does not exist")
            confirm = get_input("Continue anyway? (y/n)", "n")
            if confirm.lower() != 'y':
                print("Exiting...")
                sys.exit(1)
        
        category = get_input("Enter category index (0-10)", "0")
        numerical = get_input("Enter numerical features (comma-separated, or press Enter for defaults)", "")
        actual_weight = get_input("Enter actual weight in kg (optional, press Enter to skip)", "")
        
    elif data_choice == 2:
        # Batch mode
        print("\n📊 Batch Mode")
        print("-" * 50)
        max_samples = get_input("Enter number of samples to analyze", "10")
        try:
            max_samples = int(max_samples)
        except ValueError:
            print("Invalid number, using default: 10")
            max_samples = 10
            
    else:
        # Full test set
        print("\n📈 Full Test Set Mode")
        print("-" * 50)
        max_samples = get_input("Enter max samples (or 'all' for entire test set)", "100")
        if max_samples.lower() == 'all':
            max_samples = 9999  # Large number to process all
        else:
            try:
                max_samples = int(max_samples)
            except ValueError:
                print("Invalid number, using default: 100")
                max_samples = 100
    
    # =========================================================================
    # Step 4: Select Checkpoint
    # =========================================================================
    print("\n📦 Model Checkpoint")
    print("-" * 50)
    
    checkpoints = find_checkpoints()
    if checkpoints:
        print("Available checkpoints:")
        for i, ckpt in enumerate(checkpoints[:5], 1):  # Show top 5
            print(f"  [{i}] {ckpt}")
        if len(checkpoints) > 5:
            print(f"  ... and {len(checkpoints) - 5} more")
        
        ckpt_choice = get_input(f"Select checkpoint (1-{min(5, len(checkpoints))}) or enter path", "1")
        
        try:
            ckpt_idx = int(ckpt_choice) - 1
            if 0 <= ckpt_idx < len(checkpoints):
                checkpoint = os.path.join("checkpoints", checkpoints[ckpt_idx])
            else:
                checkpoint = ckpt_choice
        except ValueError:
            checkpoint = ckpt_choice
    else:
        checkpoint = get_input("Enter checkpoint path")
    
    print(f"✓ Using checkpoint: {checkpoint}")
    
    # =========================================================================
    # Step 5: Additional Options
    # =========================================================================
    print("\n⚙️  Additional Options")
    print("-" * 50)
    
    verbose = get_input("Verbose output? (y/n)", "y").lower() == 'y'
    save_json = get_input("Save results to JSON? (y/n)", "y").lower() == 'y'
    
    # LLM-specific options
    temperature = 0.7
    max_tokens = 512
    if use_llm and not use_template_only:
        advanced = get_input("Configure LLM parameters? (y/n)", "n").lower() == 'y'
        if advanced:
            temp_str = get_input("Temperature (0.0-1.0)", "0.7")
            try:
                temperature = float(temp_str)
            except ValueError:
                temperature = 0.7
            
            tokens_str = get_input("Max tokens", "512")
            try:
                max_tokens = int(tokens_str)
            except ValueError:
                max_tokens = 512
    
    # =========================================================================
    # Step 6: Build and Run Command
    # =========================================================================
    print("\n" + "="*70)
    print("🚀 RUNNING EXPLANATION PIPELINE")
    print("="*70)
    
    # Build command
    cmd = ["python", "explain.py"]
    cmd.extend(["--checkpoint", checkpoint])
    
    if data_choice == 1:
        # Single image
        cmd.extend(["--image", image_path])
        cmd.extend(["--category", str(category)])
        if numerical:
            cmd.extend(["--numerical", numerical])
        if actual_weight:
            cmd.extend(["--actual-weight", actual_weight])
    else:
        # Batch mode
        cmd.extend(["--max-samples", str(max_samples)])
    
    if not use_llm:
        cmd.append("--no-llm")
    
    if use_llm and not use_template_only:
        cmd.extend(["--temperature", str(temperature)])
        cmd.extend(["--max-tokens", str(max_tokens)])
    
    if verbose:
        cmd.append("--verbose")
    
    if save_json:
        cmd.append("--save-json")
    
    # Print command
    print(f"\n📌 Command: {' '.join(cmd)}\n")
    
    # Confirm
    confirm = get_input("Run this command? (y/n)", "y")
    if confirm.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    print("\n" + "-"*70 + "\n")
    
    # Run command
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running command: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
