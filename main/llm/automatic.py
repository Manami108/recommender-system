# now all scripts are csv1 and working 3
# core dumpted will be solved with
# ulimit -c unlimited

import subprocess
import time

# List of scripts to run in order
scripts = [
    # "/home/abhi/Desktop/Manami/recommender-system/main/llm/eval_llm_hop.py",
    # "/home/abhi/Desktop/Manami/recommender-system/main/llm/11.py",
    # "/home/abhi/Desktop/Manami/recommender-system/main/llm/22.py",
    # "/home/abhi/Desktop/Manami/recommender-system/main/llm/33.py",
    "/home/abhi/Desktop/Manami/recommender-system/main/llm/2-1.py",
    # "/home/abhi/Desktop/Manami/recommender-system/main/llm/3-1.py", 
    # "/home/abhi/Desktop/Manami/recommender-system/main/llm/4-1.py",
    "/home/abhi/Desktop/Manami/recommender-system/main/llm/2-2.py", 
    "/home/abhi/Desktop/Manami/recommender-system/main/llm/3-2.py",
    "/home/abhi/Desktop/Manami/recommender-system/main/llm/4-2.py",
    "/home/abhi/Desktop/Manami/recommender-system/main/llm/5-2.py",
]

# Loop through each script
for idx, script in enumerate(scripts):
    print(f"▶️ Running script {idx+1}/{len(scripts)}: {script}")
    
    try:
        subprocess.run(["python", script], check=True)
        print(f"✅ Script {script} finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Script {script} failed with error: {e}")
    
    # Wait 10 minutes before the next script, except after the last script
    if idx < len(scripts) - 1:
        print("⏳ Waiting 10 minutes before running the next script...")
        time.sleep(300)  # 600 seconds = 10 minutes

print("🎉 All scripts have been executed.")
