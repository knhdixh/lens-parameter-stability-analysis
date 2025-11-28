#!/usr/bin/env python3
"""
capture_calibrate.py - Basler Camera Calibration with ChArUco Detection

This script interfaces with a Basler industrial camera using pypylon,
performs real-time ChArUco board detection, and calculates camera
calibration parameters (intrinsics and distortion coefficients).

Key Features:
- Native pypylon camera control (not cv2.VideoCapture)
- Real-time ChArUco corner detection with visualization
- Interactive capture mode for collecting calibration frames
- Calibration computation with re-projection error
- CSV logging of calibration results for analysis

Author: [Your Name]
Thesis: Stability of Zoom Lens Calibration Parameters
Date: November 2025

Usage:
    python capture_calibrate.py [--exposure <us>] [--min-corners <n>]
    
Controls:
    'q' / ESC   - Quit the application
    'c'         - Capture current frame's corners (if valid detection)
    'k'         - Run calibration using collected corners
    'd'         - Delete last captured frame
    'r'         - Reset (clear all captured corners)
    's'         - Save current frame as image file
    'i'         - Show/hide camera info overlay
"""

import cv2
import numpy as np
from pypylon import pylon
import argparse
import csv
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import sys

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# ChArUco board parameters - MUST match display_pattern.py exactly!
ARUCO_DICT_ID = cv2.aruco.DICT_6X6_250  # Same dictionary as display script
SQUARES_X = 7           # Number of chessboard squares in X direction
SQUARES_Y = 5           # Number of chessboard squares in Y direction
SQUARE_LENGTH = 0.04    # Size of chessboard square (meters)
MARKER_LENGTH = 0.03    # Size of ArUco marker (meters)

# Minimum number of corners required for a valid capture
# More corners = more accurate calibration
MIN_CORNERS_DEFAULT = 10

# Minimum frames required for calibration
# At least 10-15 frames from different angles/positions recommended
MIN_FRAMES_FOR_CALIBRATION = 5

# Display settings
PREVIEW_MAX_WIDTH = 1280    # Maximum width for preview window (resize if larger)
PREVIEW_MAX_HEIGHT = 960    # Maximum height for preview window

# CSV output file
CSV_FILENAME = "calibration_log.csv"

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CalibrationData:
    """
    Stores collected calibration data (corners from multiple frames).
    
    This data structure accumulates corner detections from multiple
    captures, which are then used together for camera calibration.
    """
    # List of corner positions in image coordinates (2D)
    all_corners: List[np.ndarray] = field(default_factory=list)
    
    # List of corner IDs (which chessboard corners were detected)
    all_ids: List[np.ndarray] = field(default_factory=list)
    
    # Image size (width, height) - needed for calibration
    image_size: Optional[Tuple[int, int]] = None
    
    # Number of valid captures
    @property
    def num_captures(self) -> int:
        return len(self.all_corners)
    
    def add_capture(self, corners: np.ndarray, ids: np.ndarray, 
                    image_size: Tuple[int, int]) -> None:
        """Add a new capture to the calibration data."""
        self.all_corners.append(corners)
        self.all_ids.append(ids)
        self.image_size = image_size
    
    def remove_last(self) -> bool:
        """Remove the last capture. Returns True if successful."""
        if self.num_captures > 0:
            self.all_corners.pop()
            self.all_ids.pop()
            return True
        return False
    
    def clear(self) -> None:
        """Clear all captured data."""
        self.all_corners.clear()
        self.all_ids.clear()


@dataclass
class CalibrationResult:
    """
    Stores the results of camera calibration.
    
    These are the intrinsic camera parameters and distortion coefficients
    that describe the camera's optical characteristics.
    """
    # Camera matrix (intrinsic parameters)
    # [[fx, 0, cx],
    #  [0, fy, cy],
    #  [0,  0,  1]]
    camera_matrix: np.ndarray
    
    # Distortion coefficients [k1, k2, p1, p2, k3, ...]
    dist_coeffs: np.ndarray
    
    # Re-projection error (RMS, in pixels)
    # Lower is better; <0.5 pixels is excellent
    reprojection_error: float
    
    # Timestamp of calibration
    timestamp: str
    
    @property
    def fx(self) -> float:
        """Focal length in X (pixels)."""
        return self.camera_matrix[0, 0]
    
    @property
    def fy(self) -> float:
        """Focal length in Y (pixels)."""
        return self.camera_matrix[1, 1]
    
    @property
    def cx(self) -> float:
        """Principal point X coordinate (pixels)."""
        return self.camera_matrix[0, 2]
    
    @property
    def cy(self) -> float:
        """Principal point Y coordinate (pixels)."""
        return self.camera_matrix[1, 2]
    
    @property
    def k1(self) -> float:
        """Radial distortion coefficient k1."""
        return self.dist_coeffs[0, 0] if self.dist_coeffs.ndim > 1 else self.dist_coeffs[0]
    
    @property
    def k2(self) -> float:
        """Radial distortion coefficient k2."""
        return self.dist_coeffs[0, 1] if self.dist_coeffs.ndim > 1 else self.dist_coeffs[1]
    
    @property
    def p1(self) -> float:
        """Tangential distortion coefficient p1."""
        return self.dist_coeffs[0, 2] if self.dist_coeffs.ndim > 1 else self.dist_coeffs[2]
    
    @property
    def p2(self) -> float:
        """Tangential distortion coefficient p2."""
        return self.dist_coeffs[0, 3] if self.dist_coeffs.ndim > 1 else self.dist_coeffs[3]
    
    def to_csv_row(self) -> List:
        """Convert to a list suitable for CSV writing."""
        return [
            self.timestamp,
            f"{self.fx:.6f}",
            f"{self.fy:.6f}",
            f"{self.cx:.6f}",
            f"{self.cy:.6f}",
            f"{self.k1:.8f}",
            f"{self.k2:.8f}",
            f"{self.p1:.8f}",
            f"{self.p2:.8f}",
            f"{self.reprojection_error:.6f}"
        ]
    
    def print_summary(self) -> None:
        """Print a formatted summary of calibration results."""
        print("\n" + "="*60)
        print("CALIBRATION RESULTS")
        print("="*60)
        print(f"Timestamp: {self.timestamp}")
        print("-"*60)
        print("Intrinsic Parameters (Camera Matrix):")
        print(f"  fx (focal length X): {self.fx:.4f} pixels")
        print(f"  fy (focal length Y): {self.fy:.4f} pixels")
        print(f"  cx (principal point X): {self.cx:.4f} pixels")
        print(f"  cy (principal point Y): {self.cy:.4f} pixels")
        print("-"*60)
        print("Distortion Coefficients:")
        print(f"  k1 (radial): {self.k1:.8f}")
        print(f"  k2 (radial): {self.k2:.8f}")
        print(f"  p1 (tangential): {self.p1:.8f}")
        print(f"  p2 (tangential): {self.p2:.8f}")
        print("-"*60)
        print(f"Re-projection Error: {self.reprojection_error:.4f} pixels")
        if self.reprojection_error < 0.5:
            print("  [EXCELLENT] Error < 0.5 pixels")
        elif self.reprojection_error < 1.0:
            print("  [GOOD] Error < 1.0 pixel")
        else:
            print("  [FAIR] Consider capturing more diverse frames")
        print("="*60 + "\n")


# =============================================================================
# CAMERA FUNCTIONS (PYPYLON)
# =============================================================================

def initialize_camera(exposure_time: Optional[float] = None) -> pylon.InstantCamera:
    """
    Initialize and configure a Basler camera using pypylon.
    
    This function:
    1. Finds and connects to the first available Basler camera
    2. Configures it for optimal calibration capture
    3. Sets up proper image format conversion
    
    Args:
        exposure_time: Optional exposure time in microseconds.
                      If None, uses camera's default or auto exposure.
    
    Returns:
        Configured InstantCamera object ready for grabbing
    
    Raises:
        RuntimeError: If no camera is found or connection fails
    """
    # Get the first available camera
    # pylon.TlFactory is the transport layer factory that manages camera discovery
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    
    if len(devices) == 0:
        raise RuntimeError(
            "No Basler camera found!\n"
            "Please check:\n"
            "  1. Camera is connected via USB/GigE\n"
            "  2. Camera drivers are installed (Pylon Viewer works)\n"
            "  3. Camera is not in use by another application"
        )
    
    # Create an InstantCamera object for the first found device
    # InstantCamera provides a convenient high-level interface
    camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
    
    # Open the camera for configuration
    camera.Open()
    
    # Print camera information
    print(f"[CAMERA] Connected to: {camera.GetDeviceInfo().GetModelName()}")
    print(f"         Serial: {camera.GetDeviceInfo().GetSerialNumber()}")
    
    # Configure pixel format for color output
    # We want BGR8 for OpenCV compatibility
    try:
        # Try to set pixel format to BGR8 directly
        camera.PixelFormat.SetValue("BGR8")
        print("         Pixel Format: BGR8 (native)")
    except:
        # If BGR8 not available, we'll convert later
        try:
            camera.PixelFormat.SetValue("RGB8")
            print("         Pixel Format: RGB8 (will convert)")
        except:
            # Fallback to whatever is available (likely Mono8 or BayerRG8)
            print(f"         Pixel Format: {camera.PixelFormat.GetValue()} (will convert)")
    
    # Configure exposure if specified
    if exposure_time is not None:
        try:
            # Disable auto exposure first
            camera.ExposureAuto.SetValue("Off")
            camera.ExposureTime.SetValue(exposure_time)
            print(f"         Exposure: {exposure_time} µs (manual)")
        except Exception as e:
            print(f"[WARNING] Could not set exposure: {e}")
    else:
        try:
            camera.ExposureAuto.SetValue("Continuous")
            print("         Exposure: Auto")
        except:
            print("         Exposure: Default")
    
    # Configure for software triggering (grab on demand)
    try:
        camera.TriggerMode.SetValue("Off")
    except:
        pass
    
    # Get sensor resolution
    width = camera.Width.GetValue()
    height = camera.Height.GetValue()
    print(f"         Resolution: {width}x{height}")
    
    return camera


def create_image_converter() -> pylon.ImageFormatConverter:
    """
    Create an image format converter for Pylon images.
    
    Basler cameras may output various pixel formats (Bayer, Mono, YUV, etc.).
    This converter ensures we always get BGR8 format for OpenCV processing.
    
    Returns:
        Configured ImageFormatConverter
    """
    converter = pylon.ImageFormatConverter()
    
    # Set output format to BGR8packed (OpenCV's native format)
    # BGR8packed means Blue-Green-Red, 8 bits per channel, packed (no padding)
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    
    return converter


def grab_frame(camera: pylon.InstantCamera, 
               converter: pylon.ImageFormatConverter,
               timeout: int = 5000) -> Optional[np.ndarray]:
    """
    Grab a single frame from the camera.
    
    Uses GrabStrategy_LatestImageOnly to ensure we always get the most
    recent frame, which is important for real-time preview and avoiding
    buffer delays.
    
    Args:
        camera: The InstantCamera object
        converter: Image format converter
        timeout: Grab timeout in milliseconds
    
    Returns:
        BGR image as numpy array, or None if grab failed
    """
    try:
        # Start grabbing if not already started
        if not camera.IsGrabbing():
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        
        # Retrieve the latest image
        grab_result = camera.RetrieveResult(timeout, pylon.TimeoutHandling_ThrowException)
        
        if grab_result.GrabSucceeded():
            # Convert to BGR8 format for OpenCV
            image = converter.Convert(grab_result)
            
            # Get the numpy array from the converted image
            # The .Array property gives us direct access to the pixel data
            frame = image.GetArray()
            
            grab_result.Release()
            return frame
        else:
            print(f"[ERROR] Grab failed: {grab_result.ErrorCode} - {grab_result.ErrorDescription}")
            grab_result.Release()
            return None
            
    except Exception as e:
        print(f"[ERROR] Exception during grab: {e}")
        return None


# =============================================================================
# CHARUCO DETECTION FUNCTIONS
# =============================================================================

def create_charuco_detector() -> Tuple[cv2.aruco.CharucoBoard, 
                                        cv2.aruco.ArucoDetector,
                                        cv2.aruco.CharucoDetector]:
    """
    Create ChArUco board and detectors matching the display script.
    
    Returns:
        Tuple containing:
        - CharucoBoard: The board definition
        - ArucoDetector: Detector for ArUco markers
        - CharucoDetector: Detector for ChArUco corners
    """
    # Create ArUco dictionary (must match display script!)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    
    # Create the CharucoBoard with same parameters as display
    board = cv2.aruco.CharucoBoard(
        size=(SQUARES_X, SQUARES_Y),
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGTH,
        dictionary=aruco_dict
    )
    
    # Create detector parameters
    # These can be tuned for better detection in challenging conditions
    detector_params = cv2.aruco.DetectorParameters()
    
    # Refinement for better corner accuracy
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    
    # Create ArucoDetector for marker detection
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    
    # Create CharucoDetector for corner detection
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
    
    return board, aruco_detector, charuco_detector


def detect_charuco_corners(frame: np.ndarray,
                           charuco_detector: cv2.aruco.CharucoDetector
                           ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Detect ChArUco corners in a frame.
    
    Args:
        frame: BGR image to process
        charuco_detector: The CharucoDetector object
    
    Returns:
        Tuple containing:
        - corners: Detected corner positions (Nx1x2 array) or None
        - ids: Corner IDs (Nx1 array) or None  
        - num_corners: Number of detected corners
    """
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ChArUco corners
    # This returns corners and their IDs directly
    charuco_corners, charuco_ids, marker_corners, marker_ids = \
        charuco_detector.detectBoard(gray)
    
    if charuco_corners is not None and len(charuco_corners) > 0:
        return charuco_corners, charuco_ids, len(charuco_corners)
    else:
        return None, None, 0


def draw_detection_overlay(frame: np.ndarray,
                           corners: Optional[np.ndarray],
                           ids: Optional[np.ndarray],
                           num_corners: int,
                           min_corners: int) -> np.ndarray:
    """
    Draw detection results and status on the frame.
    
    Args:
        frame: BGR image to draw on
        corners: Detected corner positions
        ids: Corner IDs
        num_corners: Number of detected corners
        min_corners: Minimum required for valid capture
    
    Returns:
        Frame with overlays drawn
    """
    display_frame = frame.copy()
    
    if corners is not None and ids is not None:
        # Draw detected ChArUco corners
        cv2.aruco.drawDetectedCornersCharuco(display_frame, corners, ids, (0, 255, 0))
        
        # Detection status
        if num_corners >= min_corners:
            status_color = (0, 255, 0)  # Green - valid
            status_text = f"VALID: {num_corners} corners (Press 'C' to capture)"
        else:
            status_color = (0, 165, 255)  # Orange - insufficient
            status_text = f"NEED MORE: {num_corners}/{min_corners} corners"
    else:
        status_color = (0, 0, 255)  # Red - no detection
        status_text = "NO DETECTION - Adjust camera or target"
    
    # Draw status bar at top
    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(display_frame, status_text, (10, 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    return display_frame


def draw_info_panel(frame: np.ndarray, 
                    calib_data: CalibrationData,
                    show_info: bool = True) -> np.ndarray:
    """
    Draw information panel showing capture status and controls.
    
    Args:
        frame: BGR image to draw on
        calib_data: Current calibration data
        show_info: Whether to show the info panel
    
    Returns:
        Frame with info panel drawn
    """
    if not show_info:
        return frame
    
    display_frame = frame.copy()
    h, w = display_frame.shape[:2]
    
    # Info panel at bottom
    panel_height = 80
    cv2.rectangle(display_frame, (0, h - panel_height), (w, h), (40, 40, 40), -1)
    
    # Capture count
    cv2.putText(display_frame, f"Captures: {calib_data.num_captures}", 
                (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Calibration status
    if calib_data.num_captures >= MIN_FRAMES_FOR_CALIBRATION:
        calib_status = f"Ready to calibrate (Press 'K')"
        calib_color = (0, 255, 0)
    else:
        calib_status = f"Need {MIN_FRAMES_FOR_CALIBRATION - calib_data.num_captures} more captures"
        calib_color = (0, 165, 255)
    cv2.putText(display_frame, calib_status, (10, h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, calib_color, 1)
    
    # Controls
    controls = "C=Capture | K=Calibrate | D=Delete | R=Reset | S=Save | Q=Quit"
    cv2.putText(display_frame, controls, (10, h - 8), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    
    return display_frame


# =============================================================================
# CALIBRATION FUNCTIONS
# =============================================================================

def run_calibration(calib_data: CalibrationData,
                    board: cv2.aruco.CharucoBoard) -> Optional[CalibrationResult]:
    """
    Run camera calibration using collected ChArUco corner data.
    
    This function uses OpenCV's calibrateCamera with the ChArUco corner
    detections to compute the camera's intrinsic parameters and distortion
    coefficients.
    
    Args:
        calib_data: Collected calibration data (corners and IDs)
        board: The CharucoBoard used for calibration
    
    Returns:
        CalibrationResult object, or None if calibration failed
    """
    if calib_data.num_captures < MIN_FRAMES_FOR_CALIBRATION:
        print(f"[ERROR] Need at least {MIN_FRAMES_FOR_CALIBRATION} captures for calibration")
        print(f"        Currently have: {calib_data.num_captures}")
        return None
    
    if calib_data.image_size is None:
        print("[ERROR] Image size not recorded")
        return None
    
    print(f"\n[CALIBRATION] Starting calibration with {calib_data.num_captures} frames...")
    
    # Prepare object points (3D points of corners on the board)
    # ChArUco provides this through the board object
    all_obj_points = []
    all_img_points = []
    
    for corners, ids in zip(calib_data.all_corners, calib_data.all_ids):
        # Get the object points for the detected corners
        obj_points, img_points = board.matchImagePoints(corners, ids)
        
        if obj_points is not None and len(obj_points) > 0:
            all_obj_points.append(obj_points)
            all_img_points.append(img_points)
    
    if len(all_obj_points) < MIN_FRAMES_FOR_CALIBRATION:
        print("[ERROR] Not enough valid frames after processing")
        return None
    
    # Run calibration
    # Flags can be adjusted to fix certain parameters or enable rational model
    calibration_flags = 0
    # calibration_flags |= cv2.CALIB_FIX_K3  # Uncomment to fix k3 to 0
    
    try:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=all_obj_points,
            imagePoints=all_img_points,
            imageSize=calib_data.image_size,
            cameraMatrix=None,
            distCoeffs=None,
            flags=calibration_flags
        )
        
        # ret is the RMS re-projection error
        result = CalibrationResult(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            reprojection_error=ret,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Calibration failed: {e}")
        return None


def save_to_csv(result: CalibrationResult, filename: str = CSV_FILENAME) -> None:
    """
    Append calibration results to a CSV file.
    
    Creates the file with headers if it doesn't exist, otherwise appends
    a new row with the calibration data.
    
    Args:
        result: CalibrationResult to save
        filename: Output CSV filename
    """
    file_exists = os.path.exists(filename)
    
    headers = [
        'Timestamp', 'fx', 'fy', 'cx', 'cy', 
        'k1', 'k2', 'p1', 'p2', 'Reprojection_Error'
    ]
    
    try:
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write headers if new file
            if not file_exists:
                writer.writerow(headers)
            
            # Write data row
            writer.writerow(result.to_csv_row())
        
        print(f"[CSV] Results appended to: {filename}")
        
    except Exception as e:
        print(f"[ERROR] Could not save to CSV: {e}")


# =============================================================================
# DISPLAY HELPER FUNCTIONS
# =============================================================================

def resize_for_display(frame: np.ndarray, 
                       max_width: int = PREVIEW_MAX_WIDTH,
                       max_height: int = PREVIEW_MAX_HEIGHT) -> np.ndarray:
    """
    Resize frame for display if it exceeds maximum dimensions.
    
    High-resolution industrial cameras can have very large frames
    (e.g., 4096x3000). This function scales them down for comfortable
    viewing while maintaining aspect ratio.
    
    Args:
        frame: Original frame
        max_width: Maximum display width
        max_height: Maximum display height
    
    Returns:
        Resized frame (or original if already within limits)
    """
    h, w = frame.shape[:2]
    
    # Check if resizing is needed
    if w <= max_width and h <= max_height:
        return frame
    
    # Calculate scale factor
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)
    
    # Resize
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main entry point for the calibration capture application.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Basler camera calibration with ChArUco detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python capture_calibrate.py                     # Auto exposure
    python capture_calibrate.py --exposure 20000   # 20ms exposure
    python capture_calibrate.py --min-corners 15   # Require 15 corners

The script will:
1. Connect to the first available Basler camera
2. Show live preview with ChArUco detection overlay
3. Allow capturing frames with 'C' key
4. Run calibration with 'K' key after enough captures
5. Save results to calibration_log.csv
        """
    )
    parser.add_argument('--exposure', type=float, default=None,
                       help='Exposure time in microseconds (default: auto)')
    parser.add_argument('--min-corners', type=int, default=MIN_CORNERS_DEFAULT,
                       help=f'Minimum corners for valid capture (default: {MIN_CORNERS_DEFAULT})')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Directory for output files (default: current)')
    
    args = parser.parse_args()
    
    # Initialize camera
    print("\n" + "="*60)
    print("BASLER CAMERA CALIBRATION")
    print("="*60 + "\n")
    
    try:
        camera = initialize_camera(args.exposure)
    except RuntimeError as e:
        print(f"\n[FATAL] {e}")
        sys.exit(1)
    
    # Create image converter
    converter = create_image_converter()
    
    # Create ChArUco detector
    board, aruco_detector, charuco_detector = create_charuco_detector()
    print(f"\n[CHARUCO] Board: {SQUARES_X}x{SQUARES_Y} squares")
    print(f"          Dictionary: DICT_6X6_250")
    print(f"          Min corners: {args.min_corners}")
    
    # Initialize calibration data storage
    calib_data = CalibrationData()
    
    # Display state
    show_info = True
    frame_count = 0
    
    print("\n" + "="*60)
    print("LIVE PREVIEW - Press 'Q' to quit")
    print("="*60 + "\n")
    
    # Create display window
    window_name = "Basler Camera - ChArUco Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # Grab frame from camera
            frame = grab_frame(camera, converter)
            
            if frame is None:
                print("[WARNING] Failed to grab frame, retrying...")
                continue
            
            frame_count += 1
            
            # Get image size (needed for calibration)
            h, w = frame.shape[:2]
            image_size = (w, h)
            
            # Detect ChArUco corners
            corners, ids, num_corners = detect_charuco_corners(frame, charuco_detector)
            
            # Create display frame with overlays
            display_frame = draw_detection_overlay(
                frame, corners, ids, num_corners, args.min_corners
            )
            display_frame = draw_info_panel(display_frame, calib_data, show_info)
            
            # Resize for display if needed
            display_frame = resize_for_display(display_frame)
            
            # Show frame
            cv2.imshow(window_name, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Quit
                print("\n[INFO] Exiting...")
                break
                
            elif key == ord('c'):  # Capture
                if corners is not None and num_corners >= args.min_corners:
                    calib_data.add_capture(corners, ids, image_size)
                    print(f"[CAPTURE] Frame captured! Total: {calib_data.num_captures}")
                else:
                    print(f"[CAPTURE] Invalid - need at least {args.min_corners} corners")
                    
            elif key == ord('k'):  # Calibrate
                result = run_calibration(calib_data, board)
                if result is not None:
                    result.print_summary()
                    csv_path = os.path.join(args.output_dir, CSV_FILENAME)
                    save_to_csv(result, csv_path)
                    
            elif key == ord('d'):  # Delete last
                if calib_data.remove_last():
                    print(f"[DELETE] Last capture removed. Total: {calib_data.num_captures}")
                else:
                    print("[DELETE] No captures to delete")
                    
            elif key == ord('r'):  # Reset
                calib_data.clear()
                print("[RESET] All captures cleared")
                
            elif key == ord('s'):  # Save frame
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(args.output_dir, f"frame_{timestamp}.png")
                cv2.imwrite(filename, frame)
                print(f"[SAVE] Frame saved: {filename}")
                
            elif key == ord('i'):  # Toggle info
                show_info = not show_info
                
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        # Cleanup
        print("[INFO] Cleaning up...")
        
        if camera.IsGrabbing():
            camera.StopGrabbing()
        camera.Close()
        
        cv2.destroyAllWindows()
        
        print("[INFO] Done")


if __name__ == "__main__":
    main()
