# Stability Analysis of Zoom Lens Calibration Parameters

## A Comprehensive Guide for BSc Thesis Research

---

# Chapter 1: Introduction

## 1.1 Background: The Role of Camera Calibration in Machine Vision

Camera calibration is the foundational process that transforms a camera from a simple image-capturing device into a **precision measurement instrument**. In machine vision applications—such as robotic guidance, 3D reconstruction, quality inspection, and autonomous vehicles—we need to extract accurate geometric information from images. This is only possible if we understand exactly how the camera projects the 3D world onto its 2D sensor.

Consider this: when you take a photo of a building, the parallel vertical edges of the building appear to converge toward the top of the image. This is **perspective distortion**, a geometric transformation governed by the camera's optics. Camera calibration mathematically characterizes this transformation, allowing us to:

1. **Measure real-world distances** from pixel coordinates
2. **Correct lens distortions** that bend straight lines
3. **Reconstruct 3D geometry** from multiple 2D images
4. **Fuse data from multiple cameras** in a common coordinate system

Without calibration, a camera is like a ruler with unknown markings—it can show you something, but you cannot trust the measurements.

### Why Industrial Applications Demand Calibration

| Application | Calibration Requirement | Typical Accuracy Needed |
|-------------|------------------------|------------------------|
| Robot arm guidance | Know where objects are in 3D space | ±0.5 mm |
| Quality inspection | Measure part dimensions from images | ±0.1 mm |
| Autonomous vehicles | Estimate distances to obstacles | ±5 cm at 50m |
| 3D scanning | Reconstruct surface geometry | ±0.05 mm |
| Augmented reality | Overlay virtual objects correctly | ±1 pixel |

---

## 1.2 Problem Statement: The Zoom Lens Challenge

### Standard Calibration Assumes a Fixed Lens

Traditional camera calibration methods assume the camera's optical system is **static**. You calibrate once, obtain a set of parameters, and use them for all subsequent measurements. This works perfectly for:

- Fixed focal length (prime) lenses
- Cameras with no mechanical adjustments
- Controlled laboratory environments

### The Problem with Varifocal (Zoom) Lenses

Modern industrial applications often require **varifocal** or **zoom** lenses for flexibility:

- **Variable field of view**: Zoom out to see the whole scene, zoom in for detail
- **Variable working distance**: Focus on objects at different distances
- **Adaptive inspection**: One camera system for multiple tasks

However, zoom lenses introduce a fundamental problem: **changing the zoom or focus mechanically moves lens elements inside the barrel, which alters ALL intrinsic parameters**.

```
┌─────────────────────────────────────────────────────────────┐
│                    ZOOM LENS PROBLEM                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Fixed Lens:     Calibrate once → Use forever              │
│                                                             │
│   Zoom Lens:      Calibrate at Zoom=1 → Parameters valid    │
│                   Change to Zoom=2   → Parameters INVALID   │
│                   Change to Zoom=3   → Parameters INVALID   │
│                           ...                               │
│                   Return to Zoom=1   → Parameters valid?    │
│                                      ↑                      │
│                                      │                      │
│                            THIS IS YOUR RESEARCH QUESTION   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### The Hysteresis Question

Even if we calibrate at every zoom setting, a deeper question remains: **if we return to the same setting, do we get the same parameters?**

**Hysteresis** is the phenomenon where a system's output depends not just on its current input, but on its history. In mechanical systems like zoom lenses:

- Gear backlash (slack in gear teeth)
- Friction in lens barrel threads
- Elastic deformation of lens mounts
- Motor positioning errors

These effects mean that "Zoom = 50mm" reached by zooming IN from 35mm might not be physically identical to "Zoom = 50mm" reached by zooming OUT from 70mm.

---

## 1.3 Research Objectives

This thesis addresses three specific objectives:

### Objective 1: Design and Build an Automated Test Rig

**Goal**: Create a repeatable, automated system for calibrating a motorized zoom lens at multiple settings.

**Components**:
- Basler industrial camera with motorized zoom lens
- Digital monitor displaying ChArUco calibration patterns
- Python software for automated capture and calibration
- Motorized lens control for programmable zoom/focus adjustment

**Why Automation Matters**: Manual calibration introduces human variability. Automated calibration ensures:
- Consistent capture timing
- Reproducible pattern positioning
- Systematic coverage of all zoom settings
- Large datasets for statistical analysis

### Objective 2: Map Intrinsic Parameters vs. Lens Settings

**Goal**: Characterize how each calibration parameter changes as a function of zoom and focus settings.

**Expected Relationships**:
- **Focal length (fx, fy)**: Should increase approximately linearly with zoom
- **Principal point (cx, cy)**: May shift as optical axis moves
- **Distortion (k1, k2, p1, p2)**: May change non-linearly with zoom

**Deliverable**: Plots showing parameter values across the zoom range, with uncertainty bands.

### Objective 3: Quantify Stability and Hysteresis

**Goal**: Determine if returning to a zoom setting produces consistent calibration parameters.

**Method**: 
1. Calibrate at Zoom Setting A
2. Move to Zoom Setting B
3. Return to Zoom Setting A
4. Calibrate again
5. Compare results statistically

**Key Metrics**:
- Mean parameter difference between "forward" and "return" calibrations
- Standard deviation of repeated calibrations at the same setting
- Correlation between approach direction and parameter values

---

## 1.4 Research Aim

> **To determine if a "per-setting" calibration lookup table for a zoom lens is reliable for accurate measurements, or if mechanical hysteresis introduces errors that exceed acceptable tolerances for precision applications.**

This research will produce:

1. **Quantitative data** on the magnitude of hysteresis effects in a specific zoom lens
2. **Practical guidelines** for when per-setting calibration is sufficient vs. when real-time calibration is needed
3. **A methodology** that can be applied to evaluate other zoom lenses

---

## 1.5 Structure of the Thesis

| Chapter | Title | Content |
|---------|-------|---------|
| 1 | Introduction | Problem context, objectives, and thesis structure |
| 2 | Theoretical Background | Camera models, lens optics, calibration mathematics |
| 3 | Research Methodology | Hardware design, software tools, experimental procedures |
| 4 | Results and Analysis | Parameter maps, hysteresis measurements, statistical analysis |
| 5 | Conclusion | Findings, implications, limitations, future work |

---

# Chapter 2: Theoretical Background

## 2.1 Camera Models: The Pinhole Model

### The Ideal Pinhole Camera

The **pinhole camera model** is the mathematical foundation of camera calibration. It describes how 3D points in the world project onto a 2D image plane.

```
                    World Point P(X, Y, Z)
                           *
                          /|
                         / |
                        /  |
                       /   |
         ────────────*─────────────  Image Plane
                    /p(u,v)
                   /
                  /
                 * ← Optical Center (Camera Origin)
                 
```

### The Projection Equation

A 3D point $\mathbf{P} = (X, Y, Z)$ in camera coordinates projects to image point $\mathbf{p} = (u, v)$ as:

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{Z} \mathbf{K} \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}
$$

Where $\mathbf{K}$ is the **camera intrinsic matrix**:

$$
\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

### Intrinsic Parameters Explained

#### Focal Length: $f_x$ and $f_y$ (pixels)

**Physical meaning**: The distance from the optical center to the image plane, measured in pixels.

**Why two values?** 
- $f_x$: focal length in the horizontal direction
- $f_y$: focal length in the vertical direction

In theory, these should be equal. In practice, they differ slightly due to:
- Non-square pixels (rare in modern sensors)
- Slight manufacturing imperfections
- Anamorphic lens effects

**Units**: Pixels. To convert to millimeters: $f_{mm} = f_{pixels} \times \text{pixel\_size}_{mm}$

**Typical values**: 
- For a 50mm lens on a 5MP sensor with 3.45μm pixels: $f_x \approx f_y \approx 14500$ pixels
- For a wide-angle lens: smaller values
- For a telephoto/zoomed lens: larger values

**What YOUR result showed**:
```
fx = 11696 pixels
fy = 22514 pixels  ← PROBLEM: Nearly 2x difference!
```
This is physically impossible for a normal lens. It indicates the calibration failed, likely due to:
- Insufficient pattern movement (all captures at similar positions)
- Poor corner detection quality
- Numerical instability from bad data

**Good calibration**: $|f_x - f_y| / f_x < 1\%$

---

#### Principal Point: $c_x$ and $c_y$ (pixels)

**Physical meaning**: The point where the optical axis intersects the image sensor.

```
    ┌─────────────────────────────────────┐
    │                                     │
    │                                     │
    │              *(cx, cy)              │  ← Principal Point
    │               Principal Point       │     (ideally at image center)
    │                                     │
    │                                     │
    └─────────────────────────────────────┘
              Image Sensor
```

**Ideal location**: Center of the image
- For a 1920×1080 image: $c_x \approx 960$, $c_y \approx 540$
- For a 5472×3648 image: $c_x \approx 2736$, $c_y \approx 1824$

**Why it's not exactly centered**:
- Manufacturing tolerances in sensor mounting
- Lens decentering
- Optical axis not perfectly aligned with sensor center

**Typical deviation from center**: ±2-5% of image dimensions

**What YOUR result showed**:
```
cx = 2660 pixels  ← Plausible for ~5000 pixel wide sensor
cy = 660 pixels   ← PROBLEM: Very far from center if image is tall
```
If your image is, say, 3648 pixels tall, the center should be ~1824. Having $c_y = 660$ means the principal point is in the upper portion—possible but unusual, and combined with other issues, suggests bad calibration.

---

### Extrinsic Parameters

While intrinsic parameters describe the camera itself, **extrinsic parameters** describe where the camera is in space:

- **Rotation matrix $\mathbf{R}$**: 3×3 matrix describing camera orientation
- **Translation vector $\mathbf{t}$**: 3×1 vector describing camera position

For your thesis focusing on intrinsic parameter stability, extrinsics are computed during calibration but not the primary focus.

---

## 2.2 Lens Optics and Distortions

Real lenses are not ideal pinholes. They introduce **distortions** that bend straight lines into curves.

### Lens Parameters

#### Focal Length (Optical)

**Definition**: The distance from the lens's optical center to the focal point (where parallel rays converge).

```
    Parallel rays →  ───────────┐
                     ───────────┼───→ Focal Point
                     ───────────┘
                          ↑
                    │←───────→│
                    Focal Length f
```

**For zoom lenses**: Focal length changes as you zoom:
- Wide angle (zoomed out): f ≈ 12mm → wider field of view
- Telephoto (zoomed in): f ≈ 50mm → narrower field of view

#### Focus Distance

**Definition**: The distance from the lens to the plane of sharp focus.

**Effect on calibration**: Changing focus slightly moves lens elements, which can affect:
- Effective focal length
- Distortion characteristics
- Principal point position

#### Aperture (f-number)

**Definition**: The ratio of focal length to entrance pupil diameter: $N = f/D$

**Effect on calibration**: 
- Smaller aperture (larger f-number, e.g., f/8) → greater depth of field, sharper calibration images
- Larger aperture (smaller f-number, e.g., f/1.4) → shallower depth of field, potentially blurry corners

**Recommendation**: Use f/5.6 to f/8 for calibration (good sharpness across the image).

---

### Distortion Types

#### Radial Distortion

**Cause**: Rays at different distances from the optical axis are bent by different amounts.

**Types**:

```
    Original Grid        Barrel Distortion       Pincushion Distortion
    ┌───┬───┬───┐        ╭───────────────╮        ┌─────────────────┐
    │   │   │   │        │ ╭───┬───┬───╮ │        │  ╲   │   │   ╱  │
    ├───┼───┼───┤        │ ├───┼───┼───┤ │        │   ╲──┼───┼──╱   │
    │   │   │   │   →    │ ├───┼───┼───┤ │   or   │   │──┼───┼──│   │
    ├───┼───┼───┤        │ ├───┼───┼───┤ │        │   ╱──┼───┼──╲   │
    │   │   │   │        │ ╰───┴───┴───╯ │        │  ╱   │   │   ╲  │
    └───┴───┴───┘        ╰───────────────╯        └─────────────────┘
                         (k1 < 0 typically)       (k1 > 0 typically)
```

**Mathematical model**:

$$
x_{distorted} = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
$$
$$
y_{distorted} = y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
$$

Where $r^2 = x^2 + y^2$ is the squared distance from the principal point.

**Distortion coefficients**:
- $k_1$: Primary radial distortion (most significant)
- $k_2$: Secondary radial distortion
- $k_3$: Tertiary radial distortion (often fixed to 0)

**Typical values**:
| Lens Type | k1 | k2 |
|-----------|-----|-----|
| Wide angle | -0.3 to -0.1 | 0.05 to 0.2 |
| Normal | -0.1 to 0.1 | -0.1 to 0.1 |
| Telephoto | -0.05 to 0.05 | -0.05 to 0.05 |

**What YOUR result showed**:
```
k1 = -3.77    ← EXTREME! Should be -0.5 to 0.5
k2 = -104.9   ← ABSURD! Should be -1 to 1
```
These values are physically impossible. The algorithm produced them because it was trying to compensate for systematic errors in the input data.

---

#### Tangential Distortion

**Cause**: Lens elements are not perfectly parallel to the image sensor.

```
    Ideal:                    Reality:
    
    Lens ═══════              Lens ═══════
           ║                         ╲
           ║                          ╲
    Sensor ═══════            Sensor ═══════
    
    (Parallel)                (Slight tilt → tangential distortion)
```

**Mathematical model**:

$$
x_{distorted} = x + [2p_1 xy + p_2(r^2 + 2x^2)]
$$
$$
y_{distorted} = y + [p_1(r^2 + 2y^2) + 2p_2 xy]
$$

**Distortion coefficients**:
- $p_1$: Tangential distortion in one axis
- $p_2$: Tangential distortion in the perpendicular axis

**Typical values**: $|p_1|, |p_2| < 0.01$ for well-manufactured lenses

**What YOUR result showed**:
```
p1 = 0.286   ← HIGH, but not impossible
p2 = -0.568  ← HIGH, but not impossible
```
These are elevated but less obviously wrong than k1/k2.

---

## 2.3 Principles of Camera Calibration

### Zhang's Method (The Standard Approach)

In 2000, Zhengyou Zhang published "A Flexible New Technique for Camera Calibration" which became the foundation of modern calibration. OpenCV implements this method.

**Key insight**: By observing a planar pattern (like a checkerboard) at multiple orientations, we can solve for all intrinsic parameters without knowing the exact 3D positions of the pattern.

### The Calibration Process

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ZHANG'S CALIBRATION METHOD                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INPUT:                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐       ┌─────────┐           │
│  │ Image 1 │  │ Image 2 │  │ Image 3 │  ...  │ Image N │           │
│  │   📷    │  │   📷    │  │   📷    │       │   📷    │           │
│  └─────────┘  └─────────┘  └─────────┘       └─────────┘           │
│       ↓            ↓            ↓                 ↓                 │
│  STEP 1: Detect pattern corners in each image                      │
│       ↓            ↓            ↓                 ↓                 │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │ Corner coordinates: (u₁,v₁), (u₂,v₂), ..., (uₘ,vₘ)     │       │
│  └─────────────────────────────────────────────────────────┘       │
│       ↓                                                             │
│  STEP 2: Solve for homographies (one per image)                    │
│       ↓                                                             │
│  STEP 3: Extract initial intrinsic parameters from homographies    │
│       ↓                                                             │
│  STEP 4: Estimate extrinsic parameters for each image              │
│       ↓                                                             │
│  STEP 5: Non-linear optimization (bundle adjustment)               │
│       ↓                                                             │
│  OUTPUT:                                                            │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │ Camera Matrix K = [fx, 0, cx; 0, fy, cy; 0, 0, 1]       │       │
│  │ Distortion Coefficients [k1, k2, p1, p2, k3]           │       │
│  │ Reprojection Error (RMS)                                │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Multiple Images Are Needed

The calibration has **many unknowns**:
- 4 intrinsic parameters: $f_x, f_y, c_x, c_y$
- 4-5 distortion coefficients: $k_1, k_2, p_1, p_2, (k_3)$
- 6 extrinsic parameters per image: 3 rotation + 3 translation

Each detected corner provides 2 equations (one for x, one for y).

**Minimum requirement**: 
- At least 3 images with the pattern at different orientations
- At least 6 corners detected per image

**Practical recommendation**: 
- 15-30 images for robust calibration
- Pattern should cover different areas of the image
- Pattern should appear at different scales (simulated depth)
- Pattern should be tilted at various angles

### Reprojection Error: The Quality Metric

**Definition**: After calibration, we use the computed parameters to **project** the known 3D pattern points back onto each image. The reprojection error measures how far these projected points are from the actually detected corners.

$$
\text{RMS Error} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(u_i - \hat{u}_i)^2 + (v_i - \hat{v}_i)^2}
$$

Where:
- $(u_i, v_i)$: detected corner position
- $(\hat{u}_i, \hat{v}_i)$: reprojected corner position
- $N$: total number of corners across all images

**Interpreting reprojection error**:

| Error (pixels) | Quality | Interpretation |
|----------------|---------|----------------|
| < 0.1 | Excellent | Lab-grade calibration |
| 0.1 - 0.3 | Very Good | High-precision applications |
| 0.3 - 0.5 | Good | Most industrial applications |
| 0.5 - 1.0 | Acceptable | General use |
| 1.0 - 2.0 | Poor | Recalibrate recommended |
| > 2.0 | Bad | Calibration failed |

**What YOUR result showed**:
```
Reprojection Error: 9.48 pixels  ← VERY BAD!
```
This means the model doesn't fit the data at all. It's like fitting a straight line to data that forms a circle—the math gives you an answer, but it's meaningless.

---

### OpenCV Calibration Function

```python
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints,    # 3D coordinates of pattern corners
    imagePoints,     # 2D coordinates detected in images
    imageSize,       # (width, height) of images
    cameraMatrix,    # Initial guess (or None)
    distCoeffs,      # Initial guess (or None)
    flags            # Options (fix certain parameters, etc.)
)
```

**Returns**:
- `ret`: Reprojection error (RMS)
- `camera_matrix`: The 3×3 intrinsic matrix K
- `dist_coeffs`: Distortion coefficients [k1, k2, p1, p2, k3]
- `rvecs`: Rotation vectors for each image (extrinsics)
- `tvecs`: Translation vectors for each image (extrinsics)

---

## 2.4 Calibration Targets

### Passive Targets: Printed Patterns

**Checkerboard**:
```
┌───┬───┬───┬───┬───┬───┬───┐
│███│   │███│   │███│   │███│
├───┼───┼───┼───┼───┼───┼───┤
│   │███│   │███│   │███│   │
├───┼───┼───┼───┼───┼───┼───┤
│███│   │███│   │███│   │███│
├───┼───┼───┼───┼───┼───┼───┤
│   │███│   │███│   │███│   │
└───┴───┴───┴───┴───┴───┴───┘
```
- **Pros**: Simple, cheap (print on paper), very accurate corner detection
- **Cons**: All-or-nothing detection (if any corner occluded, fails), no orientation information

**ChArUco Board** (used in your experiment):
```
┌───┬───┬───┬───┬───┬───┬───┐
│███│░░░│███│░░░│███│░░░│███│
│███│░1░│███│░2░│███│░3░│███│  ← ArUco markers in white squares
├───┼───┼───┼───┼───┼───┼───┤
│░░░│███│░░░│███│░░░│███│░░░│
│░4░│███│░5░│███│░6░│███│░7░│
├───┼───┼───┼───┼───┼───┼───┤
│███│░░░│███│░░░│███│░░░│███│
│███│░8░│███│░9░│███│░10│███│
└───┴───┴───┴───┴───┴───┴───┘
```
- **Pros**: 
  - Partial detection works (some corners occluded? Still calibrates!)
  - ArUco markers provide unique IDs (knows which corner is which)
  - Better for automated systems
- **Cons**: Slightly lower corner accuracy than pure checkerboard

### Active Targets: Display-Based Patterns

**Your setup uses a digital monitor as an "active target"**:

**Advantages**:
1. **No physical target to manufacture**: Generate pattern in software
2. **Variable patterns**: Change scale, position programmatically
3. **Automation**: Script can cycle through configurations
4. **No wear**: Physical targets degrade, digital patterns don't

**Disadvantages**:
1. **Screen flatness**: LCD monitors aren't perfectly flat
2. **Pixel grid artifacts**: Moiré patterns possible
3. **Reflections**: Glossy screens cause glare
4. **Refresh rate**: Can cause flickering in images
5. **Limited viewing angle**: Some screens distort at angles

**Mitigation strategies**:
- Use matte screen or anti-glare filter
- Ensure pattern pixels are integer multiples (no sub-pixel artifacts)
- Control ambient lighting to minimize reflections
- Use high-quality monitor with good viewing angles

---

## 2.5 Gaps and Opportunities

### Current Literature Limitations

Most calibration research assumes:
1. **Fixed lens**: Parameters don't change after calibration
2. **One-time process**: Calibrate once, use forever
3. **Mechanical perfection**: Lens settings are deterministic

### The Research Gap

For zoom lenses:
- How much do parameters actually vary with zoom setting?
- Is the variation repeatable (deterministic) or random?
- Is hysteresis (direction-dependent variation) significant?
- Can we create a lookup table of parameters vs. zoom setting?

### Your Contribution

This thesis will provide:
1. **Empirical data** on parameter variation for a specific zoom lens
2. **Hysteresis quantification** methodology applicable to other lenses
3. **Practical guidance** for industrial users of zoom lenses in metrology

---

# Chapter 3: Research Methodology

## 3.1 Experimental Hardware Design

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     EXPERIMENTAL SETUP                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐         ~0.5-1.5m          ┌─────────────────┐   │
│   │             │ ◄─────────────────────────► │                 │   │
│   │   BASLER    │                             │    DIGITAL      │   │
│   │   CAMERA    │        Optical Axis         │    MONITOR      │   │
│   │             │ ════════════════════════════│                 │   │
│   │  [Motorized │                             │   ChArUco       │   │
│   │   Zoom Lens]│                             │   Pattern       │   │
│   │             │                             │                 │   │
│   └──────┬──────┘                             └────────┬────────┘   │
│          │                                             │            │
│          │ USB3/GigE                                   │ HDMI/DP    │
│          │                                             │            │
│          ▼                                             ▼            │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                     CONTROL COMPUTER                        │  │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │  │
│   │  │capture_     │  │display_     │  │ Lens Control        │ │  │
│   │  │calibrate.py │  │pattern.py   │  │ Software            │ │  │
│   │  └─────────────┘  └─────────────┘  └─────────────────────┘ │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Hardware Components

#### 1. Basler Industrial Camera

**Specifications to document in thesis**:
- Model number
- Sensor type and size
- Resolution (e.g., 5472 × 3648)
- Pixel size (e.g., 2.4 μm)
- Interface (USB3 Vision / GigE Vision)

**Why Basler?**
- Industrial-grade: stable, repeatable
- pypylon SDK: precise control
- Global shutter: no rolling shutter artifacts

#### 2. Motorized Zoom Lens

**Specifications to document**:
- Model and manufacturer
- Focal length range (e.g., 12-50mm)
- Aperture range
- Motor type (stepper, DC servo)
- Control interface (serial, I2C)

**Key characteristic for your research**: Can the motor return to exact positions repeatedly?

#### 3. Digital Monitor (Active Target)

**Specifications to document**:
- Size and resolution (e.g., 27" 2560×1440)
- Panel type (IPS preferred for viewing angles)
- Matte vs. glossy surface
- Flatness specification if available

#### 4. Mounting and Positioning

- Camera mount (tripod, optical rail)
- Monitor mount (arm, fixed stand)
- Distance between camera and monitor
- Method for ensuring alignment

---

## 3.2 Data Collection Software

### Software Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   SOFTWARE STACK                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Application Layer:                                         │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ display_pattern │  │capture_calibrate│                  │
│  │      .py        │  │      .py        │                  │
│  └────────┬────────┘  └────────┬────────┘                  │
│           │                    │                            │
│  Library Layer:                │                            │
│  ┌────────▼────────┐  ┌───────▼────────┐  ┌─────────────┐ │
│  │   OpenCV        │  │    pypylon     │  │ screeninfo  │ │
│  │ (cv2.aruco)     │  │ (Basler SDK)   │  │             │ │
│  └────────┬────────┘  └───────┬────────┘  └─────────────┘ │
│           │                   │                            │
│  OS/Hardware Layer:           │                            │
│  ┌────────▼────────┐  ┌──────▼─────────┐                  │
│  │ Display Driver  │  │ Camera Driver  │                  │
│  └─────────────────┘  └────────────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.21 | Numerical computations |
| opencv-contrib-python | ≥4.8 | ChArUco detection, calibration |
| pypylon | ≥2.0 | Basler camera control |
| screeninfo | ≥0.8 | Multi-monitor support |

### ChArUco Board Parameters

**CRITICAL**: These must match between display and capture scripts!

```python
ARUCO_DICT_ID = cv2.aruco.DICT_6X6_250  # 6x6 bit markers, 250 unique IDs
SQUARES_X = 7                            # Columns of squares
SQUARES_Y = 5                            # Rows of squares
SQUARE_LENGTH = 0.04                     # Physical size (meters) - for reference
MARKER_LENGTH = 0.03                     # ArUco marker size (meters)
```

**Note on SQUARE_LENGTH and MARKER_LENGTH**: Since we're using a monitor, these values are somewhat arbitrary—they define the scale of the 3D coordinate system, not actual physical sizes. What matters is the **ratio**: marker should be smaller than square.

---

## 3.3 Experimental Procedure 1: Intrinsic Parameter Mapping

### Objective

**Research Question**: How do intrinsic camera parameters change as a function of zoom lens setting?

**Hypothesis**: Parameters (especially focal length) will show a systematic, potentially non-linear relationship with zoom setting.

### Method

```
┌─────────────────────────────────────────────────────────────┐
│           PROCEDURE 1: PARAMETER MAPPING                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  FOR zoom_setting IN [min_zoom, min_zoom+step, ..., max]:   │
│      │                                                      │
│      ├──► 1. Set lens to zoom_setting                       │
│      │                                                      │
│      ├──► 2. Wait for mechanical settling (e.g., 500ms)     │
│      │                                                      │
│      ├──► 3. Capture N calibration frames:                  │
│      │       FOR position IN all_positions:                 │
│      │           FOR scale IN all_scales:                   │
│      │               - Display pattern at (position, scale) │
│      │               - Capture frame                        │
│      │               - Detect corners                       │
│      │               - Store if valid                       │
│      │                                                      │
│      ├──► 4. Run calibration with collected frames          │
│      │                                                      │
│      └──► 5. Record: [zoom_setting, fx, fy, cx, cy,        │
│                       k1, k2, p1, p2, reproj_error]        │
│                                                             │
│  OUTPUT: CSV file with parameters at each zoom setting      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Recommended Capture Strategy

At each zoom setting, capture frames covering:

| Scale Level | Positions | Frames per Config | Total |
|-------------|-----------|-------------------|-------|
| 40% | 5 (corners + center) | 1 | 5 |
| 60% | 5 | 1 | 5 |
| 80% | 5 | 1 | 5 |
| 100% | 5 | 1 | 5 |
| **Total per zoom setting** | | | **20** |

If you have 10 zoom settings: 10 × 20 = 200 total calibration frames per complete sweep.

### Expected Results

```
Zoom Setting    fx (pixels)    fy (pixels)    k1          Reproj Error
-----------    -----------    -----------    --------    ------------
12mm           8,500          8,510          -0.15       0.35
16mm           11,200         11,180         -0.12       0.32
20mm           14,100         14,090         -0.08       0.28
...
50mm           35,200         35,180         -0.02       0.25
```

---

## 3.4 Experimental Procedure 2: Hysteresis Test

### Objective

**Research Question**: When returning to a zoom setting from different directions, do we get the same calibration parameters?

**Hypothesis**: Mechanical hysteresis will cause measurable differences in parameters depending on the approach direction.

### Method

```
┌─────────────────────────────────────────────────────────────┐
│           PROCEDURE 2: HYSTERESIS TEST                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Define: TARGET = 30mm (or mid-range zoom setting)          │
│          LOW = 12mm (minimum zoom)                          │
│          HIGH = 50mm (maximum zoom)                         │
│                                                             │
│  FOR trial IN [1, 2, 3, ..., N_trials]:                    │
│      │                                                      │
│      ├──► FORWARD PATH (approaching from LOW):              │
│      │    1. Go to LOW setting                              │
│      │    2. Go to TARGET setting                           │
│      │    3. Calibrate → Record as "Forward_trial_N"        │
│      │                                                      │
│      ├──► REVERSE PATH (approaching from HIGH):             │
│      │    1. Go to HIGH setting                             │
│      │    2. Go to TARGET setting                           │
│      │    3. Calibrate → Record as "Reverse_trial_N"        │
│      │                                                      │
│                                                             │
│  ANALYSIS:                                                  │
│  - Compare Forward vs Reverse distributions                 │
│  - Statistical test for significant difference              │
│  - Quantify hysteresis magnitude                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Statistical Analysis

**Metrics to compute**:

1. **Mean difference**: $\bar{x}_{forward} - \bar{x}_{reverse}$ for each parameter

2. **Standard deviation**: $\sigma_{forward}$, $\sigma_{reverse}$ 

3. **Coefficient of variation**: $CV = \sigma / \mu$ (relative variability)

4. **Statistical significance**: 
   - t-test or Mann-Whitney U test
   - p-value < 0.05 indicates significant hysteresis

### Expected Results Table

```
Parameter    Forward Mean    Reverse Mean    Difference    p-value    Significant?
---------    ------------    ------------    ----------    -------    ------------
fx           14,105.2        14,098.7        6.5 px        0.032      Yes*
fy           14,102.8        14,097.1        5.7 px        0.048      Yes*
cx           2,736.4         2,735.9         0.5 px        0.412      No
cy           1,824.1         1,823.8         0.3 px        0.687      No
k1           -0.0821         -0.0819         0.0002        0.891      No
```

---

## 3.5 Data Analysis Methods

### For Procedure 1: Parameter Mapping

**Visualization**:
- Line plots: Parameter vs. Zoom setting
- Error bars: ±1 standard deviation from repeated calibrations
- Polynomial fit: Find functional relationship

```python
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P

# Example: Fitting focal length vs zoom
zoom_settings = [12, 16, 20, 25, 30, 35, 40, 45, 50]
fx_values = [8500, 11200, 14100, 17600, 21100, 24700, 28200, 31800, 35200]

# Fit polynomial
coeffs = np.polyfit(zoom_settings, fx_values, deg=2)
fit_fn = np.poly1d(coeffs)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(zoom_settings, fx_values, label='Measured', s=100)
plt.plot(zoom_settings, fit_fn(zoom_settings), 'r--', label=f'Fit: {coeffs}')
plt.xlabel('Zoom Setting (mm)')
plt.ylabel('Focal Length fx (pixels)')
plt.title('Focal Length vs Zoom Setting')
plt.legend()
plt.grid(True)
plt.savefig('fx_vs_zoom.png', dpi=150)
```

### For Procedure 2: Hysteresis Analysis

**Statistical Tests**:

```python
from scipy import stats

# Example data
forward_fx = [14105, 14108, 14103, 14107, 14104]  # 5 trials
reverse_fx = [14098, 14099, 14097, 14100, 14098]  # 5 trials

# Paired t-test (if same trial conditions)
t_stat, p_value = stats.ttest_rel(forward_fx, reverse_fx)
print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")

# Effect size (Cohen's d)
diff = np.array(forward_fx) - np.array(reverse_fx)
cohens_d = np.mean(diff) / np.std(diff)
print(f"Cohen's d: {cohens_d:.3f}")  # >0.8 is large effect
```

### Measurement Uncertainty

For a thorough thesis, quantify uncertainty sources:

| Source | Typical Magnitude | How to Estimate |
|--------|------------------|-----------------|
| Corner detection | ±0.1 pixels | Repeated detection on same image |
| Pattern flatness | ±0.5 pixels | Compare to reference calibration |
| Lens settling | ±1-5 pixels | Multiple calibrations after single move |
| Camera noise | ±0.05 pixels | Multiple images, same scene |

---

# Chapter 4: Results and Analysis (Template)

## 4.1 Parameter Mapping Results

### 4.1.1 Focal Length vs Zoom Setting

[Insert figure: fx and fy plotted against zoom setting]

**Observations**:
- Focal length shows [linear/quadratic/other] relationship with zoom setting
- fx and fy maintain [excellent/good/poor] agreement (ratio = X.XXX ± 0.XXX)
- The fitted relationship is: $f_x = a \cdot z^2 + b \cdot z + c$ where...

### 4.1.2 Principal Point Variation

[Insert figure: cx and cy plotted against zoom setting]

**Observations**:
- Principal point [does/does not] shift significantly with zoom
- Maximum deviation from center: X pixels (Y% of image dimension)

### 4.1.3 Distortion Coefficient Variation

[Insert figure: k1, k2 plotted against zoom setting]

**Observations**:
- Radial distortion [increases/decreases/varies non-monotonically] with zoom
- At wide angle (12mm): barrel distortion with k1 = -X.XX
- At telephoto (50mm): [minimal/pincushion] distortion with k1 = X.XX

---

## 4.2 Stability and Hysteresis Results

### 4.2.1 Repeatability at Fixed Setting

[Insert table: Statistics for repeated calibrations at same setting]

**Finding**: Standard deviation of repeated calibrations is X pixels, representing Y% of the mean value.

### 4.2.2 Hysteresis Magnitude

[Insert figure: Box plots comparing Forward vs Reverse approaches]

**Finding**: 
- Mean hysteresis in fx: X pixels (Y% of value)
- Statistical significance: p = X.XXX
- Practical significance: [This does/does not] exceed typical measurement uncertainty

---

## 4.3 Reprojection Error Analysis

### 4.3.1 Error vs Zoom Setting

[Insert figure: Reprojection error at each zoom setting]

**Observations**:
- Reprojection error ranges from X to Y pixels
- [Higher/Lower] zoom settings show [better/worse] calibration quality
- Possible causes: [optical quality variation, depth of field, pattern visibility]

### 4.3.2 Error Comparison: Forward vs Reverse

[Insert statistical comparison]

**Finding**: Hysteresis [does/does not] significantly increase reprojection error.

---

# Chapter 5: Conclusion and Future Work

## 5.1 Summary of Key Findings

1. **Parameter Mapping**: [Summarize the functional relationships found]

2. **Hysteresis Quantification**: [State the magnitude and significance]

3. **Practical Implications**: [What does this mean for users of zoom lenses?]

## 5.2 Answer to Research Question

> Can a "per-setting" calibration lookup table for a zoom lens provide reliable measurements?

[Your answer based on the data]

**If hysteresis is small**: "Yes, a lookup table indexed by zoom setting provides calibration accuracy within X pixels, which is [sufficient/insufficient] for applications requiring Y mm accuracy."

**If hysteresis is large**: "Per-setting calibration is unreliable due to significant hysteresis (X pixels). Real-time calibration or mechanical improvements are recommended."

## 5.3 Limitations

1. **Single lens tested**: Results may not generalize to other lens models
2. **Limited zoom range**: Only tested X to Y mm
3. **Temperature not controlled**: Thermal effects not characterized
4. **Display-based target**: Potential artifacts vs. physical targets

## 5.4 Future Work

1. **Test multiple lens models**: Compare hysteresis across manufacturers
2. **Characterize focus effects**: Include focus setting as a variable
3. **Temperature study**: Measure thermal stability
4. **Compensation algorithms**: Develop methods to correct for hysteresis

---

# Appendix A: Good vs Bad Calibration Results

## What Good Results Look Like

```
============================================================
CALIBRATION RESULTS - GOOD EXAMPLE
============================================================
Timestamp: 2025-11-28 15:30:00
------------------------------------------------------------
Intrinsic Parameters (Camera Matrix):
  fx (focal length X): 14523.4521 pixels
  fy (focal length Y): 14518.9873 pixels    ← Within 0.03% of fx ✓
  cx (principal point X): 2738.2341 pixels  ← Near image center ✓
  cy (principal point Y): 1821.5678 pixels  ← Near image center ✓
------------------------------------------------------------
Distortion Coefficients:
  k1 (radial): -0.08234156                  ← Reasonable magnitude ✓
  k2 (radial): 0.12453789                   ← Reasonable magnitude ✓
  p1 (tangential): 0.00045123               ← Very small ✓
  p2 (tangential): -0.00023456              ← Very small ✓
------------------------------------------------------------
Re-projection Error: 0.2847 pixels          ← Excellent! ✓
  [EXCELLENT] Error < 0.5 pixels
============================================================
```

## What Bad Results Look Like (Your Result)

```
============================================================
CALIBRATION RESULTS - BAD EXAMPLE
============================================================
Timestamp: 2025-11-28 15:05:34
------------------------------------------------------------
Intrinsic Parameters (Camera Matrix):
  fx (focal length X): 11696.4687 pixels
  fy (focal length Y): 22514.3981 pixels    ← 93% different from fx ✗
  cx (principal point X): 2660.7352 pixels  ← Might be OK
  cy (principal point Y): 660.9524 pixels   ← Very off-center ✗
------------------------------------------------------------
Distortion Coefficients:
  k1 (radial): -3.77560003                  ← Extreme! ✗
  k2 (radial): -104.92546995                ← Absurd! ✗
  p1 (tangential): 0.28642092               ← High ✗
  p2 (tangential): -0.56840120              ← High ✗
------------------------------------------------------------
Re-projection Error: 9.4807 pixels          ← Very bad ✗
  [FAIR] Consider capturing more diverse frames
============================================================
```

---

# Appendix B: Troubleshooting Checklist

## Before Capturing

- [ ] Camera is in focus (sharp marker edges visible)
- [ ] Pattern fills 50-80% of the image at maximum scale
- [ ] No reflections or glare on the monitor
- [ ] Exposure is correct (not too bright, not too dark)
- [ ] Monitor displaying pattern with good contrast

## During Capture

- [ ] At least 15+ green corners detected before pressing 'C'
- [ ] Corners detected across the ENTIRE board (not just center)
- [ ] Varying scale between captures (use 'S' key on display)
- [ ] Varying position between captures (use 'P' key on display)
- [ ] Minimum 20-30 frames captured

## After Calibration

- [ ] Reprojection error < 1.0 pixel
- [ ] fx and fy within 1% of each other
- [ ] cx and cy near image center (within 5%)
- [ ] k1 and k2 between -1 and +1
- [ ] p1 and p2 between -0.01 and +0.01

---

# Appendix C: Quick Reference - Parameter Ranges

| Parameter | Symbol | Typical Range | Units | Your Goal |
|-----------|--------|---------------|-------|-----------|
| Focal length X | fx | 5,000 - 50,000 | pixels | fx ≈ fy |
| Focal length Y | fy | 5,000 - 50,000 | pixels | fy ≈ fx |
| Principal point X | cx | 45-55% of width | pixels | Near center |
| Principal point Y | cy | 45-55% of height | pixels | Near center |
| Radial distortion 1 | k1 | -0.5 to +0.5 | - | Small |
| Radial distortion 2 | k2 | -0.5 to +0.5 | - | Small |
| Tangential 1 | p1 | -0.01 to +0.01 | - | Very small |
| Tangential 2 | p2 | -0.01 to +0.01 | - | Very small |
| Reprojection error | RMS | 0.1 to 0.5 | pixels | < 0.5 |

---

*Document generated for BSc Thesis: Stability of Zoom Lens Calibration Parameters*
*Date: November 2025*
