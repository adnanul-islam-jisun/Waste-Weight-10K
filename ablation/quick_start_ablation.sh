#!/bin/bash

# Quick Start Script for Ablation Study
# This script provides an interactive menu to run the ablation study

echo "=============================================================================="
echo "🔬 MODEL ARCHITECTURE ABLATION STUDY - QUICK START"
echo "=============================================================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

echo "Select an option:"
echo ""
echo "  1) 🧪 Test Setup (Recommended first time)"
echo "  2) 🚀 Run ALL Experiments (4-6 hours)"
echo "  3) 🎯 Run Specific Experiments"
echo "  4) 🔄 Resume Interrupted Run"
echo "  5) 🐛 Debug Mode (Single experiment, fast)"
echo "  6) 📊 Generate Visualizations"
echo "  7) 📋 View Results Summary"
echo "  8) ❓ Help"
echo "  9) 🚪 Exit"
echo ""
read -p "Enter your choice (1-9): " choice

case $choice in
    1)
        echo ""
        echo -e "${BLUE}Testing ablation study setup...${NC}"
        python test_ablation_setup.py
        ;;
    2)
        echo ""
        echo -e "${YELLOW}⚠️  This will run all 6 experiments and take 4-6 hours.${NC}"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            echo -e "${GREEN}Starting full ablation study...${NC}"
            python run_ablation_study.py --all
        else
            echo "Cancelled."
        fi
        ;;
    3)
        echo ""
        echo "Available experiments:"
        echo "  1 - Full Model (Baseline)"
        echo "  2 - Image Only"
        echo "  3 - Metadata Only"
        echo "  4 - No Mutual Attention"
        echo "  5 - ViT-B/32 (Faster)"
        echo "  6 - ViT-L/16 (Larger)"
        echo ""
        read -p "Enter experiment numbers (comma-separated, e.g., 1,2,4): " exp_nums
        echo -e "${GREEN}Running experiments: $exp_nums${NC}"
        python run_ablation_study.py --experiments $exp_nums
        ;;
    4)
        echo ""
        echo -e "${BLUE}Resuming from previous run...${NC}"
        python run_ablation_study.py --resume
        ;;
    5)
        echo ""
        echo "Debug mode runs 1 experiment with 1 batch per epoch (very fast)"
        read -p "Enter experiment number (1-6): " exp_num
        echo -e "${GREEN}Running experiment $exp_num in debug mode...${NC}"
        python run_ablation_study.py --debug --experiments $exp_num
        ;;
    6)
        echo ""
        if [ -d "ablation_results" ]; then
            echo -e "${GREEN}Generating visualizations...${NC}"
            python visualize_ablation_results.py
        else
            echo -e "${RED}❌ No results found. Please run experiments first.${NC}"
        fi
        ;;
    7)
        echo ""
        if [ -f "ablation_results/summary_report.csv" ]; then
            echo -e "${GREEN}📊 Results Summary:${NC}"
            echo ""
            column -t -s, ablation_results/summary_report.csv | head -20
            echo ""
            echo -e "${BLUE}Full report: ablation_results/summary_report.csv${NC}"
        else
            echo -e "${RED}❌ No results found. Please run experiments first.${NC}"
        fi
        ;;
    8)
        echo ""
        echo -e "${BLUE}📖 Ablation Study Help${NC}"
        echo ""
        echo "This ablation study evaluates 6 model architecture variants:"
        echo ""
        echo "1. Full Model - Complete architecture (baseline)"
        echo "2. Image Only - Tests importance of metadata"
        echo "3. Metadata Only - Tests importance of images"
        echo "4. No Attention - Tests mutual attention contribution"
        echo "5. ViT-B/32 - Tests faster ViT variant"
        echo "6. ViT-L/16 - Tests larger ViT variant"
        echo ""
        echo "Typical workflow:"
        echo "  1. Run option 1 to test setup"
        echo "  2. Run option 5 to quickly test one experiment"
        echo "  3. Run option 2 to run all experiments (overnight)"
        echo "  4. Run option 6 to generate visualizations"
        echo "  5. Run option 7 to view results"
        echo ""
        echo "For more details, see: ABLATION_STUDY_README.md"
        ;;
    9)
        echo ""
        echo -e "${GREEN}Goodbye!${NC}"
        exit 0
        ;;
    *)
        echo ""
        echo -e "${RED}Invalid choice. Please run again and select 1-9.${NC}"
        exit 1
        ;;
esac

echo ""
echo "=============================================================================="
echo "✅ Done!"
echo "=============================================================================="
