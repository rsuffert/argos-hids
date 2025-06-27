#!/usr/bin/env python3
"""
Simple LID-DS Setup Script for CVE-2014-0160 Scenario
Uses local CVE scenario folder and runs the test scenario.
"""
import subprocess
import sys
import os

def main():
    print("LID-DS Setup - CVE-2014-0160 Scenario")
    print("=" * 40)
    
    # Check if SCENARIOS folder exists
    scen_folder = os.path.join(os.path.dirname(__file__), "SCENARIOS")
    if os.path.exists(scen_folder):
        print(f"SCENARIOS folder found at: {scen_folder}")
    else:
        print(f"SCENARIOS folder not found at: {scen_folder}")
        print("Please make sure the 'SCENARIOS' folder exists in the same directory as this script")
        return False
    
    # Check if LID-DS directory exists
    if os.path.exists("LID-DS"):
        print("LID-DS framework directory found")
    else:
        print("LID-DS framework directory not found")
        print("Please make sure the 'LID-DS' folder exists in the same directory as this script")
        return False
    
    print("Running test scenario...")
    try:
        subprocess.run([sys.executable, "testscen.py"], check=True)
        print("Test completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
