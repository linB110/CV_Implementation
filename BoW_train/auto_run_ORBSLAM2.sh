#!/bin/bash

# === CONFIGURATION ===
LOOP_COUNT=100
SLAM_BIN="/home/lab605/lab605/ORB_SLAM2/Examples/Monocular/mono_euroc"
CONFIG="/home/lab605/lab605/ORB_SLAM2/Examples/Monocular/EuRoC.yaml"
KF_FILE="/home/lab605/KeyFrameTrajectory.txt"

# === Datasets ===
MH01_DATA="/home/lab605/lab605/dataset/EuRoC/MH_01/mav0/cam0/data"
MH01_STAMPS="/home/lab605/lab605/ORB_SLAM2/Examples/Monocular/EuRoC_TimeStamps/MH01.txt"
MH01_GT="/home/lab605/lab605/dataset/EuRoC/MH_01/groundtruth_tum.txt"

MH05_DATA="/home/lab605/lab605/dataset/EuRoC/MH_05/mav0/cam0/data"
MH05_STAMPS="/home/lab605/lab605/ORB_SLAM2/Examples/Monocular/EuRoC_TimeStamps/MH05.txt"
MH05_GT="/home/lab605/lab605/dataset/EuRoC/MH_05/groundtruth_tum.txt"

# === Vocabulary list ===
VOCAB_LIST=(
    "/home/lab605/lab605/ORB_SLAM2/Vocabulary/ORBvoc.txt"
    "/home/lab605/lab605/ORB_SLAM2/Vocabulary/VG_Boosted.txt"
    "/home/lab605/lab605/ORB_SLAM2/Vocabulary/CoCo_Boosted.txt"
    "/home/lab605/lab605/ORB_SLAM2/Vocabulary/HP_Boosted.txt"
    "/home/lab605/lab605/ORB_SLAM2/Vocabulary/BigVoc_Boosted.txt"
)

# === Function: Run SLAM + APE ===
run_eval() {
    local vocab="$1"
    local data="$2"
    local stamp="$3"
    local gt="$4"
    local log="$5"
    local seq="$6"

    echo "[INFO] Running ORB-SLAM2 on $seq using $(basename "$vocab")"
    "$SLAM_BIN" "$vocab" "$CONFIG" "$data" "$stamp" > /dev/null 2>&1

    if [ ! -f "$KF_FILE" ]; then
        echo "[ERROR] $KF_FILE not found!"
        echo "fail" >> "$log"
        return
    fi

    echo "[INFO] Evaluating $seq with evo_ape"
    rm -f temp_ape_log.txt
    evo_ape tum "$gt" "$KF_FILE" --align --correct_scale --logfile temp_ape_log.txt

    if grep -q "rmse" temp_ape_log.txt; then
        grep -m 1 "rmse" temp_ape_log.txt | awk '{print $2}' >> "$log"
        echo "[✓] RMSE written to $log"
    else
        echo "[✗] evo_ape failed for $seq" >> "$log"
    fi
}

# === Main loop over vocabularies ===
for vocab_path in "${VOCAB_LIST[@]}"; do
    vocab_name=$(basename "$vocab_path" .txt)

    LOG_MH01="${vocab_name}_mh01.txt"
    LOG_MH05="${vocab_name}_mh05.txt"
    > "$LOG_MH01"
    > "$LOG_MH05"

    #$echo "========== Vocab: $vocab_name | MH_01 Loop =========="
    #for i in $(seq 1 $LOOP_COUNT); do
    #    echo "--- Round $i / $LOOP_COUNT: MH_01"
    #    run_eval "$vocab_path" "$MH01_DATA" "$MH01_STAMPS" "$MH01_GT" "$LOG_MH01" "MH_01"
    #done

    echo "========== Vocab: $vocab_name | MH_05 Loop =========="
    for i in $(seq 1 $LOOP_COUNT); do
        echo "--- Round $i / $LOOP_COUNT: MH_05"
        run_eval "$vocab_path" "$MH05_DATA" "$MH05_STAMPS" "$MH05_GT" "$LOG_MH05" "MH_05"
    done

    echo "✅ Done: $vocab_name → $LOG_MH01, $LOG_MH05"
    echo ""
done

echo "🎉 All vocabularies completed."

