# **s8 – End‑Effector Reachability Accuracy**  
### *Benchmark Protocol & Submission Instructions*

This measure quantifies the end-effector reaching accuracy for the robotic control. This task evaluates how accurately a robot can reach **six canonical target poses** in its workspace. The metric is designed to be **hardware‑agnostic**, allowing robots with different kinematics and reachability to participate fairly. To ensure fair and honest evaluation, participants must follow a **strict procedural protocol** with multiple repetitions, continuous execution, and video verification.

---

## **1. Overview**
 
Participants must command their robot to reach **six desired poses**, defined relative to the robot's own workspace dimensions:
 
- **r** — horizontal reachability (millimeters)  
- **h** — vertical reachability (millimeters)

<p align="center">
  <img width="500" src="https://github.com/CORSMAL/2019_RA-L_Benchmark-for-Human-to-Robot-Handovers-of-Unseen-Containers-with-Unknown-Filling/blob/master/offlineScores/s8_end-effector/illustration.png">
</p>
 
For each **sequence of six poses**, the participant:
 
1. Executes **K = 3 repetitions** of the sequence  
2. Records the **reached end‑effector pose** for each attempt  
3. Submits **position and orientation** data  
4. Provides video evidence of the execution  
 
The benchmark computes a **normalised accuracy score** based on the **median position error** (across all repetitions and poses) relative to the desired positions. Orientation is recorded but **not scored**.

---
 
## **2. Workspace Parameters and Desired Poses**
 
Each robot defines its own six target positions using its measured workspace parameters **r** and **h**.
 
Let:
 
- $r$ = maximum horizontal reachability (mm)
- $h$ = maximum vertical reachability (mm)
 
### **2.1 Standard Initial Pose**
 
Before each sequence, the robot **must start** from a standardised initial pose:

**Initial pose in base frame (orientation: identity):**

$$p_{\text{init}} = (0, 0, 0), \quad q_{\text{init}} = (0, 0, 0, 1)$$

Participants must ensure their robot end‑effector can reach this pose reliably. This pose will be verified in video evidence.
 
### **2.2 The Six Target Poses**
 
The six desired positions are:
 
| Pose | $x$ (mm) | $y$ (mm) | $z$ (mm) |
|------|----------|----------|----------|
| 1 | $0$ | $r/2$ | $0$ |
| 2 | $-r/2$ | $r/2$ | $0$ |
| 3 | $+r/2$ | $r/2$ | $0$ |
| 4 | $0$ | $r/2$ | $h/4$ |
| 5 | $-r/2$ | $r/2$ | $h/4$ |
| 6 | $+r/2$ | $r/2$ | $h/4$ |

The end-effector must reach these 6 desired poses. The poses are defined by the locations depicted in the figure above, and the rotation of the end-effector gripper is fixed and pointing to the central location (i.e., end-effector gripper planes perpendicular to the central point).
 
### **2.3 Orientation**  
 
The end‑effector orientation **must point toward** the central location $(0, r/2, 0)$. Participants may use any consistent convention (e.g., quaternion), but must report the reached orientation in the submission.
 
---
 
## **3. Execution Protocol**
 
### **3.1 Single Continuous Run (No Resets, No Pauses)**
 
Participants must execute all **six target poses in a single uninterrupted sequence**, starting from the standard initial pose. The procedure for each repetition is:
 
1. **Start** from the standard initial pose $(0, 0, 0)$ with orientation identity  
2. **Move to target pose 1**, record the reached pose  
3. **Return to standard initial pose**  
4. **Move to target pose 2**, record the reached pose  
5. ... continue for all six poses ...  
6. **Return to standard initial pose** after the final pose  
 
**Return to initial pose** between targets to ensure all starts are from the same reference.
 
### **3.2 Time‑Bound Execution**
 
Each individual motion (from initial pose to target pose, or target to initial pose) must complete within **T_max = 5 seconds**.
 
**Scoring penalty:** If any motion exceeds 5 seconds, the corresponding pose in that repetition receives a **score of 0**.
 
### **3.3 Multiple Repetitions**
 
Participants must execute the **complete sequence 3 times** (K = 3 repetitions). All repetitions must be:
 
- Executed on the same day before executing the 288 handover configurations  
- Without manual recalibration between repetitions  
- Documented in separate video clips or a single continuous video  
 
### **3.4 No Manual Fine‑Tuning**
 
- No manual adjustment of robot pose between trials  
- All execution must be autonomous (i.e., driven by the robot's control system)  
- Manual teleoperation is **not permitted**  
 
---
 
## **4. Video Evidence Requirements**
 
Each submission must include **one or more video files** showing:
 
- **Robot identifier** visible on screen or in filename  
- **Sequence number** and **repetition number** (e.g., "Rep 1 of 3")  
- **Start of each sequence:** robot at the standard initial pose  
- **Target pose index** announced or displayed before motion begins  
- **Full motion:** robot moving autonomously from initial to target pose  
- **Final pose:** clearly visible and stable for ≥ 1 second before returning  
- **Timestamp overlay** or a visible clock in the frame for each motion  
 
**Video specifications:**
- Resolution: ≥ 720p  
- Frame rate: ≥ 30 fps  
- Duration: minimal editing; if edited, gaps must be marked clearly  

**Video submission:** Participants must provide **public video links** (e.g., YouTube, cloud storage with public access) or upload to the competition platform.
 
---
 
## **5. Data to Submit**
 
### **5.1 Workspace Parameters**
 
**Workspace parameters $r$ and $h$ must be reported in the JSON metadata file** at: https://corsmal.github.io/events/rgmc/icra2026/resources/metadata_template.json
 
- **r** (horizontal reachability in millimeters)  
- **h** (vertical reachability in millimeters)  
 
### **5.2 Reached Poses (CSV)**
 
For each of the **K = 3 repetitions** and each of the **6 target poses**, report:
 
- **Target ID** (1–6)  
- **Position**: $x, y, z$ in **millimeters** (base frame)  
- **Orientation**: quaternion $q_x, q_y, q_z, q_w$ (base frame)  
- **Repetition number** (1, 2, or 3)  
- **Start time**: ISO 8601 timestamp (UTC, millisecond precision)
- **End time**: ISO 8601 timestamp (UTC, millisecond precision)
- **Motion time** in **milliseconds** (computed from timestamps for verification)  

### **5.3 Optional: Joint State Logs**
 
Participants **may** submit joint‑state logs (CSV format) containing:
 
- Timestamp  
- All joint positions (radians or degrees)  
- Computed end‑effector position (from forward kinematics, in millimeters)  
- Computed end‑effector orientation (from forward kinematics)  
 
These logs allow organisers to verify consistency between reported poses and forward kinematics.
 
---
 
## **6. Submission Format**
 
### **6.1 CSV Structure**
 
Participants must submit a single CSV file with the following structure (units in **millimeters**, **milliseconds**, and **ISO 8601 UTC timestamps**):
 
```
run_id,repetition,target_id,x,y,z,qx,qy,qz,qw,start_time,end_time,motion_time_ms
1,1,1,<x1>,<y1>,<z1>,<qx1>,<qy1>,<qz1>,<qw1>,<start1>,<end1>,<duration1>
1,1,2,<x2>,<y2>,<z2>,<qx2>,<qy2>,<qz2>,<qw2>,<start2>,<end2>,<duration2>
...
1,1,6,<x6>,<y6>,<z6>,<qx6>,<qy6>,<qz6>,<qw6>,<start6>,<end6>,<duration6>
1,2,1,<x1>,<y1>,<z1>,<qx1>,<qy1>,<qz1>,<qw1>,<start1>,<end1>,<duration1>
...
1,3,6,<x6>,<y6>,<z6>,<qx6>,<qy6>,<qz6>,<qw6>,<start6>,<end6>,<duration6>
```

**Timestamp format:** ISO 8601 with UTC timezone and millisecond precision (e.g., `2024-04-02T14:30:00.000Z`)
 
### **6.2 Field Definitions**
 
| Field | Type | Description |
|-------|------|-------------|
| `run_id` | int | Run identifier (not used in scoring, for traceability) |
| `repetition` | int | Repetition number (1, 2, or 3) |
| `target_id` | int | Target pose ID (1–6) |
| `x, y, z` | float | Position in base frame (**millimeters**) |
| `qx, qy, qz, qw` | float | Quaternion (base frame) |
| `start_time` | string | Motion start time (ISO 8601 UTC, millisecond precision) |
| `end_time` | string | Motion end time (ISO 8601 UTC, millisecond precision) |
| `motion_time_ms` | float | Duration of motion in **milliseconds** (computed as `end_time - start_time`) |

---
 
## **7. Example Submission**
 
Assume:
 
- Team name: `Example_Arm_v1`  
- $r = 800$ mm  
- $h = 600$ mm  
- **3 repetitions**, each with 6 target poses  
 
Desired poses (computed):
- Pose 1: $(0, 400, 0)$ mm
- Pose 2: $(-400, 400, 0)$ mm
- Pose 3: $(400, 400, 0)$ mm
- Pose 4: $(0, 400, 150)$ mm
- Pose 5: $(-400, 400, 150)$ mm
- Pose 6: $(400, 400, 150)$ mm

Example CSV (with positions in **mm**, ISO 8601 timestamps, and computed durations):
 
```
run_id,repetition,target_id,x,y,z,qx,qy,qz,qw,start_time,end_time,motion_time_ms
1,1,1,10,390,20,0,0,0,1,2024-04-02T14:30:00.000Z,2024-04-02T14:30:02.300Z,2300
1,1,2,-410,380,-10,0,0,0,1,2024-04-02T14:30:04.500Z,2024-04-02T14:30:06.600Z,2100
1,1,3,390,410,10,0,0,0,1,2024-04-02T14:30:08.700Z,2024-04-02T14:30:10.700Z,2000
1,1,4,-20,420,160,0,0,0,1,2024-04-02T14:30:12.800Z,2024-04-02T14:30:15.300Z,2500
1,1,5,-380,410,140,0,0,0,1,2024-04-02T14:30:17.400Z,2024-04-02T14:30:19.600Z,2200
1,1,6,420,390,170,0,0,0,1,2024-04-02T14:30:21.700Z,2024-04-02T14:30:24.100Z,2400
1,2,1,20,400,10,0,0,0,1,2024-04-02T14:31:00.000Z,2024-04-02T14:31:02.400Z,2400
1,2,2,-400,390,0,0,0,0,1,2024-04-02T14:31:04.500Z,2024-04-02T14:31:06.700Z,2200
1,2,3,400,420,20,0,0,0,1,2024-04-02T14:31:08.800Z,2024-04-02T14:31:10.900Z,2100
1,2,4,-10,410,170,0,0,0,1,2024-04-02T14:31:13.000Z,2024-04-02T14:31:15.600Z,2600
1,2,5,-390,400,150,0,0,0,1,2024-04-02T14:31:17.700Z,2024-04-02T14:31:20.000Z,2300
1,2,6,410,400,160,0,0,0,1,2024-04-02T14:31:22.100Z,2024-04-02T14:31:24.400Z,2300
1,3,1,0,400,30,0,0,0,1,2024-04-02T14:32:00.000Z,2024-04-02T14:32:02.500Z,2500
1,3,2,-420,370,-20,0,0,0,1,2024-04-02T14:32:04.600Z,2024-04-02T14:32:06.600Z,2000
1,3,3,380,420,0,0,0,0,1,2024-04-02T14:32:08.700Z,2024-04-02T14:32:10.900Z,2200
1,3,4,-30,430,150,0,0,0,1,2024-04-02T14:32:13.000Z,2024-04-02T14:32:15.400Z,2400
1,3,5,-370,420,130,0,0,0,1,2024-04-02T14:32:17.500Z,2024-04-02T14:32:19.600Z,2100
1,3,6,430,380,180,0,0,0,1,2024-04-02T14:32:21.700Z,2024-04-02T14:32:23.900Z,2200
```

**Note:** Timestamps are ISO 8601 UTC format with millisecond precision. The `motion_time_ms` is computed as the difference between `end_time` and `start_time`.
 
---
 
## **8. Scoring Method**
 
### **8.1 Per‑Trial Position Error**

For each pose $i$ and repetition $k$:

$$e_{p,i,k} = \left\| p_{i,k}^{\text{sub}} - p_i^{\text{des}} \right\|$$

where:
- $p_{i,k}^{\text{sub}}$ = reported reached position for pose $i$, repetition $k$ (in mm)
- $p_i^{\text{des}}$ = desired position for pose $i$ (in mm)

### **8.2 Motion Time Verification**

For each pose $i$ and repetition $k$, verify that the reported `motion_time_ms` matches the timestamps:

$$motionTime_{ms} = \text{round}(endTime - startTime) \text{ in milliseconds}$$

**Tolerance:** ±1 ms (to account for rounding)

**If mismatch:** Organisers may request clarification or mark as invalid.

### **8.3 Median Per‑Pose Error**

Compute the **median error across repetitions**:

$$\tilde{e}_{p,i} = \text{median}_{k \in \{1,2,3\}} \, e_{p,i,k}$$

This reduces noise and prevents cherry-picking.

### **8.4 Per‑Pose Position Score**

A fixed threshold $\tau_p = 30$ mm (3 cm) is used:

$$s_{p,i} = \max\left(0, \, 1 - \frac{\tilde{e}_{p,i}}{\tau_p}\right)$$

- Perfect accuracy ($\tilde{e}_{p,i} = 0$) → score = 1  
- Error = 15 mm → score = 0.5  
- Error = 30 mm → score = 0  
- Error > 30 mm → score = 0  

### **8.5 Time Penalty**

If any motion in repetition $k$ exceeds **5000 ms**, that pose receives:

$$s_{p,i,k} = 0$$

The median is then computed over the remaining valid repetitions. If fewer than 2 valid repetitions remain, the pose score is:

$$s_{p,i} = 0$$

### **8.6 Final Score**

Average the per‑pose scores across all six poses:

$$S = \frac{1}{6} \sum_{i=1}^{6} s_{p,i}$$

The final score ranges from **0** (poor accuracy) to **1** (perfect accuracy).

---
 
## **9. Timestamp Collection Guide**

Participants must report motion start and end times using **ISO 8601 UTC format with millisecond precision**.

### **Timestamp Format**

- **Standard:** ISO 8601 (RFC 3339)
- **Timezone:** preferably UTC (Z notation mandatory)
- **Precision:** Milliseconds (3 decimal places minimum)
- **Example:** `2024-04-02T14:30:00.000Z`

### **Collection Methods**

#### **Option 1: Robot Native Timestamps (Python)**
```python
from datetime import datetime, timezone

# At motion START
start_time = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')

# Execute motion (e.g., 2.3 seconds)
robot.move_to_pose(target_pose)

# At motion END
end_time = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')

# Compute and verify duration
motion_time_ms = int((datetime.fromisoformat(end_time.replace('Z', '+00:00')) - 
                       datetime.fromisoformat(start_time.replace('Z', '+00:00'))).total_seconds() * 1000)
```

#### **Option 2: Video Clock Reference**
If no precise system clock is available, synchronise with video timestamp overlay:

1. Record video with visible **countdown clock** or **frame counter**
2. Note clock time when motion starts: `14:30:00.000`
3. Translate to ISO 8601: `2024-04-02T14:30:00.000Z`
4. Note clock time when motion ends
5. Verify `motion_time_ms` matches the difference

### **Verification Rules**

Organisers will validate:

1. **Format:** All timestamps are valid ISO 8601 UTC  
2. **Ordering:** $$\text{endTime} > \text{startTime}$$ (always)  
3. **Duration match:** `motion_time_ms ≈ (end_time - start_time)` (within ±1 ms)  
4. **Sequence integrity:** Reasonable gaps between poses (100 ms to 2 seconds)  
5. **Video synchronisation:** Timestamps align with video clock overlay

**If timestamp is invalid or mismatched:**
- Organisers request clarification
- Motion may be marked invalid
- Score for that pose = 0

---
 
## **10. Example Score Computation**
 
Using the example above, compute position errors for each pose across repetitions:
 
### **Pose 1**
- Rep 1: $e_{p,1,1} = \|(10, 390, 20) - (0, 400, 0)\| = \sqrt{100 + 100 + 400} \approx 22.4$ mm
- Rep 2: $e_{p,1,2} = \|(20, 400, 10) - (0, 400, 0)\| = \sqrt{400 + 0 + 100} \approx 22.4$ mm
- Rep 3: $e_{p,1,3} = \|(0, 400, 30) - (0, 400, 0)\| = 30$ mm
- Median: $\tilde{e}_{p,1} = 22.4$ mm
- Score: $s_{p,1} = \max(0, 1 - 22.4 / 30) = 0.25$

### **Pose 2**
- Rep 1: $e_{p,2,1} = \|(-410, 380, -10) - (-400, 400, 0)\| = \sqrt{100 + 400 + 100} \approx 22.4$ mm
- Rep 2: $e_{p,2,2} = \|(-400, 390, 0) - (-400, 400, 0)\| = 10$ mm
- Rep 3: $e_{p,2,3} = \|(-420, 370, -20) - (-400, 400, 0)\| = \sqrt{400 + 900 + 400} \approx 39.1$ mm
- Median: $\tilde{e}_{p,2} = 22.4$ mm
- Score: $s_{p,2} = \max(0, 1 - 22.4 / 30) = 0.25$

Continuing similarly for poses 3–6...
 
### **Final Aggregation**
 
| Pose | Median Error (mm) | Score |
|------|------------------|-------|
| 1 | 22.4 | 0.25 |
| 2 | 22.4 | 0.25 |
| 3 | 15 | 0.50 |
| 4 | 24 | 0.20 |
| 5 | 18 | 0.40 |
| 6 | 29 | 0.03 |
 
**Final score:**

$$S = \frac{1}{6}(0.25 + 0.25 + 0.50 + 0.20 + 0.40 + 0.03) = 0.27$$
 
---
 
## **11. Interpretation**
 
- **1.0** → robot consistently reaches within 0–10 mm across all repetitions  
- **0.75** → robot often within 5–10 mm  
- **0.5** → robot often within 10–20 mm  
- **0.25** → robot often within 20–30 mm  
- **0.1** → robot rarely within 30 mm  
- **0.0** → robot consistently misses the tolerance window  
 
This metric reflects **functional accuracy and consistency** for handover and manipulation tasks.
 
---
 
## **12. Submission Checklist**
 
Before submitting, verify:
 
- [ ] **Metadata file updated** with $r$ and $h$ values (in mm) at the JSON template  
- [ ] CSV file is properly formatted with all required fields  
- [ ] All three repetitions are included (18 rows of pose data: 3 reps × 6 poses)  
- [ ] All positions are in **millimeters**  
- [ ] All motion times are in **milliseconds** and ≤ 5000 ms  
- [ ] All quaternions are normalised (magnitude ≈ 1.0)  
- [ ] Video links are public and accessible  
- [ ] Video(s) show the full execution of all three repetitions  
- [ ] Video includes timestamp overlays or visible clock  
- [ ] Video clearly shows robot moving from initial pose to target and back  
- [ ] All poses in CSV correspond to those shown in video  
- [ ] Team name is consistent across submission  
- [ ] No manual fine‑tuning between repetitions is visible in video  
- [ ] Initial pose is the same for all repetitions  
 
---
 
## **13. Submission Instructions**
 
Participants must submit:
 
1. **Update metadata JSON file** with workspace parameters $r$ and $h$ (in millimeters)  
   - Location: https://corsmal.github.io/events/rgmc/icra2026/resources/metadata_template.json  
 
2. **CSV file** named `s8_submission_<TEAM_NAME>.csv` containing:
   - All 18 pose records (3 reps × 6 poses)  
   - Positions in **millimeters**  
   - Motion times in **milliseconds**  
 
3. **Video links** in the submission form or metadata:
   - Public links to video files (e.g., YouTube, cloud storage)  
   - One video per repetition or single continuous video  
   - Labels: `s8_video_rep1`, `s8_video_rep2`, `s8_video_rep3`  
 
4. **Optional: Joint-state logs** (CSV format) named `s8_logs_<TEAM_NAME>.csv` for verification  

**Submission deadline:** As specified on the competition website  
**Send by email to the organisers**
 
---
 
## **14. Frequently Asked Questions**
 
### **Q: Can we use teleoperation?**  
**A:** No. All motions must be autonomous (driven by your robot control system).
 
### **Q: What if the robot cannot reach the standard initial pose?**  
**A:** Your robot must be capable of reaching this pose. If not, contact the organisers to discuss accommodations.
 
### **Q: Can we pause between repetitions?**  
**A:** Yes, you may pause for setup or equipment checks between the three full repetitions. But within each repetition, all six poses must be executed continuously without resets.
 
### **Q: What if a motion takes 5100 milliseconds?**  
**A:** That pose receives a score of 0 for that repetition. The median is computed over the remaining valid repetitions (if 2 or more remain).
 
### **Q: How should we define r and h for a non-planar robot?**  
**A:** Use the maximum horizontal reach from the base and maximum vertical reach from the base. If your robot has complex kinematics, document your methodology in a brief readme file.
 
### **Q: Can we submit more than 3 repetitions?**  
**A:** Yes, you may submit additional repetitions. Scoring will use the first K=3 repetitions unless otherwise specified.
 
### **Q: Where do we report r and h?**  
**A:** Workspace parameters $r$ and $h$ must be reported in the metadata JSON file at https://corsmal.github.io/events/rgmc/icra2026/resources/metadata_template.json. They should **not** be in the CSV file.
 
### **Q: How do we submit videos?**  
**A:** Provide public video links (e.g., YouTube, Google Drive with public access, Dropbox) in the submission form or metadata file. The videos must be accessible without authentication.
 
### **Q: Will orientation be scored in future versions?**  
**A:** Not in this version (s8 v1.0). Future versions may include orientation scoring.

---

### **Q: How do I collect ISO 8601 timestamps for my robot?**  
**A:** Use your robot's native timestamp source (ROS time, system clock, middleware timestamps). Convert to ISO 8601 UTC format: `YYYY-MM-DDTHH:MM:SS`. See Section 9 for code examples.

### **Q: What if my robot doesn't have precise clock synchronisation?**  
**A:** Synchronise with your video clock. Overlay a visible timestamp in the video, and match it to your ISO 8601 timestamps in the CSV. Organisers will verify consistency.

### **Q: What if start_time and end_time don't match motion_time_ms?**  
**A:** Report all three fields. Organisers will compute `motion_time_ms = (end_time - start_time)` and compare. If there's a mismatch (> 1 ms), the motion may be flagged for review. Provide video evidence of timestamps if questioned.

---

## **Appendix: Changes from s8 v0 (Original Version)**
 
| Aspect | Original (v0) | Updated (v1.0) |
|--------|---------------|-----------------|
| Repetitions | 1 | 3 (K=3) |
| Aggregation | Single trial | Median across K trials |
| Initial pose | Not specified | Standard fixed pose (0, 0, 0) |
| Time constraints | None | 5 seconds per motion with penalties |
| Video requirement | Optional | Mandatory with timestamps; links provided |
| Position units | Meters | **Millimeters** |
| Time units | Seconds | **Milliseconds** |
| Workspace parameters | Direct in CSV | JSON metadata file |
| Scoring | Position only | Position only (orientation recorded but not scored) |
| Continuous run | Not required | Required (no resets between poses) |
| Manual fine-tuning | Allowed | Not permitted |
 
---
 
**Version:** s8 v1.0  
**Date:** April 2026
