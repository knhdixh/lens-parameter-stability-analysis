#!/usr/bin/env python3
"""
display_pattern.py - Virtual ChArUco Calibration Target Display

This script generates and displays a ChArUco board on a secondary monitor,
acting as a virtual calibration target for camera calibration experiments.

Key Features:
- Full-screen display with monitor selection support
- Automatic and manual cycling through different scales (zoom) and positions (pan)
- Reproducible sequences using fixed random seed
- High contrast display (white board on black background)

Author: [Your Name]
Thesis: Stability of Zoom Lens Calibration Parameters
Date: November 2025

Usage:
    python display_pattern.py [--monitor <id>] [--auto] [--seed <value>]
    
Controls:
    'q' / ESC   - Quit the application
    'a'         - Toggle automatic cycling mode
    's'         - Cycle to next scale (zoom level)
    'p'         - Cycle to next position (pan)
    'r'         - Reset to default scale and position
    '+' / '='   - Increase scale (zoom in)
    '-'         - Decrease scale (zoom out)
    Arrow keys  - Pan the board manually
    SPACE       - Capture current configuration (prints to console)
"""

import cv2
import numpy as np
import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# ChArUco board parameters - MUST match capture_calibrate.py
ARUCO_DICT_ID = cv2.aruco.DICT_6X6_250  # Dictionary type for ArUco markers
SQUARES_X = 7           # Number of chessboard squares in X direction
SQUARES_Y = 5           # Number of chessboard squares in Y direction
SQUARE_LENGTH = 0.04    # Size of chessboard square (in meters, for reference)
MARKER_LENGTH = 0.03    # Size of ArUco marker (in meters, for reference)

# Display parameters
BACKGROUND_COLOR = (0, 0, 0)       # Black background for high contrast
BOARD_MARGIN = 50                   # Minimum margin from screen edges (pixels)

# Reproducibility seed - CRITICAL for thesis reproducibility
DEFAULT_RANDOM_SEED = 42

# Predefined scale levels (as fraction of maximum fitting size)
# These simulate different "depths" of the calibration target
SCALE_LEVELS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Predefined position offsets (as fraction of available movement range)
# Format: (x_offset, y_offset) where 0.5 = center
POSITION_PRESETS = [
    (0.5, 0.5),   # Center
    (0.3, 0.3),   # Top-left region
    (0.7, 0.3),   # Top-right region
    (0.3, 0.7),   # Bottom-left region
    (0.7, 0.7),   # Bottom-right region
    (0.5, 0.3),   # Top-center
    (0.5, 0.7),   # Bottom-center
    (0.3, 0.5),   # Left-center
    (0.7, 0.5),   # Right-center
]

# Auto-cycling timing (seconds)
AUTO_CYCLE_INTERVAL = 2.0  # Time between automatic transitions


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DisplayState:
    """
    Maintains the current state of the ChArUco board display.
    
    This class tracks scale, position, and cycling state for the virtual target.
    Using a dataclass provides clean, self-documenting code.
    """
    scale_index: int = 6           # Index into SCALE_LEVELS (default: 1.0)
    position_index: int = 0        # Index into POSITION_PRESETS (default: center)
    custom_offset_x: float = 0.5   # Custom X position (0.0-1.0)
    custom_offset_y: float = 0.5   # Custom Y position (0.0-1.0)
    use_custom_position: bool = False  # Whether using custom or preset position
    auto_cycle: bool = False       # Automatic cycling enabled
    last_cycle_time: float = 0.0   # Timestamp of last auto-cycle
    frame_count: int = 0           # Total frames displayed (for logging)
    
    @property
    def current_scale(self) -> float:
        """Get the current scale factor."""
        return SCALE_LEVELS[self.scale_index]
    
    @property
    def current_position(self) -> Tuple[float, float]:
        """Get the current position offset (x, y)."""
        if self.use_custom_position:
            return (self.custom_offset_x, self.custom_offset_y)
        return POSITION_PRESETS[self.position_index]


# =============================================================================
# CHARUCO BOARD GENERATION
# =============================================================================

def create_charuco_board() -> Tuple[cv2.aruco.CharucoBoard, cv2.aruco.Dictionary]:
    """
    Create a ChArUco board with the specified parameters.
    
    A ChArUco board combines a chessboard pattern with ArUco markers,
    providing both the high accuracy of chessboard corner detection
    and the robustness of ArUco marker detection (handles occlusions).
    
    Returns:
        Tuple containing:
        - CharucoBoard: The board object used for detection/calibration
        - Dictionary: The ArUco dictionary used for marker generation
    
    Note: The parameters here MUST match those in capture_calibrate.py
    for successful detection and calibration.
    """
    # Get the ArUco dictionary - this defines the marker patterns
    # DICT_6X6_250 means 6x6 bit markers with 250 unique patterns
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    
    # Create the ChArUco board
    # Parameters:
    #   - size: (squares_x, squares_y) - number of squares in each direction
    #   - squareLength: physical size of each chessboard square
    #   - markerLength: physical size of each ArUco marker (must be < squareLength)
    #   - dictionary: the ArUco dictionary to use for markers
    board = cv2.aruco.CharucoBoard(
        size=(SQUARES_X, SQUARES_Y),
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGTH,
        dictionary=aruco_dict
    )
    
    return board, aruco_dict


def generate_board_image(board: cv2.aruco.CharucoBoard, 
                         width: int, 
                         height: int) -> np.ndarray:
    """
    Generate a high-resolution image of the ChArUco board.
    
    Args:
        board: The CharucoBoard object
        width: Desired image width in pixels
        height: Desired image height in pixels
    
    Returns:
        numpy array containing the board image (grayscale)
    
    The generated image is grayscale with white squares and black markers,
    which provides maximum contrast for detection.
    """
    # Generate the board image
    # marginSize adds a white border around the board
    board_image = board.generateImage(
        outSize=(width, height),
        marginSize=10,
        borderBits=1
    )
    
    return board_image


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def get_monitor_info() -> List[dict]:
    """
    Get information about available monitors.
    
    This function attempts to detect connected monitors using screeninfo
    if available, otherwise falls back to a default configuration.
    
    Returns:
        List of dictionaries containing monitor info (x, y, width, height)
    """
    try:
        # Try to use screeninfo for accurate monitor detection
        from screeninfo import get_monitors
        monitors = []
        for m in get_monitors():
            monitors.append({
                'x': m.x,
                'y': m.y,
                'width': m.width,
                'height': m.height,
                'name': m.name if hasattr(m, 'name') else f"Monitor {len(monitors)}"
            })
        return monitors
    except ImportError:
        # Fallback: assume single 1920x1080 monitor
        print("[WARNING] screeninfo not installed. Using default monitor config.")
        print("         Install with: pip install screeninfo")
        return [{
            'x': 0,
            'y': 0,
            'width': 1920,
            'height': 1080,
            'name': 'Default'
        }]


def create_fullscreen_window(window_name: str, monitor_index: int = 0) -> dict:
    """
    Create a full-screen OpenCV window on the specified monitor.
    
    Args:
        window_name: Name for the OpenCV window
        monitor_index: Index of the monitor to use (0 = primary)
    
    Returns:
        Dictionary with monitor information (x, y, width, height)
    
    Note on Windows 11:
        OpenCV's WINDOW_FULLSCREEN doesn't work well with multi-monitor setups.
        Instead, we create a borderless window sized exactly to the monitor.
    """
    monitors = get_monitor_info()
    
    # Print available monitors for user reference
    print(f"[INFO] Detected {len(monitors)} monitor(s):")
    for i, m in enumerate(monitors):
        print(f"       [{i}] {m['name']}: {m['width']}x{m['height']} at ({m['x']}, {m['y']})")
    
    if monitor_index >= len(monitors):
        print(f"[WARNING] Monitor {monitor_index} not found. Using monitor 0.")
        monitor_index = 0
    
    monitor = monitors[monitor_index]
    
    print(f"\n[INFO] Selected monitor {monitor_index}: {monitor['name']}")
    print(f"       Resolution: {monitor['width']}x{monitor['height']}")
    print(f"       Position: ({monitor['x']}, {monitor['y']})")
    
    # Windows 11 fix: Use WINDOW_NORMAL with manual sizing instead of FULLSCREEN
    # WINDOW_FULLSCREEN on Windows often snaps back to primary monitor
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Remove window borders/decorations by setting WINDOW_FULLSCREEN briefly
    # then immediately setting size - this is a workaround for Windows
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(1)  # Allow window to process
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.waitKey(1)
    
    # Move to target monitor FIRST (before resizing)
    cv2.moveWindow(window_name, monitor['x'], monitor['y'])
    cv2.waitKey(1)
    
    # Resize to exactly fill the monitor
    cv2.resizeWindow(window_name, monitor['width'], monitor['height'])
    cv2.waitKey(1)
    
    # Move again after resize (Windows sometimes resets position on resize)
    cv2.moveWindow(window_name, monitor['x'], monitor['y'])
    cv2.waitKey(1)
    
    # Try fullscreen one more time - on some systems this now works
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    return monitor


def render_board_on_canvas(board_image: np.ndarray,
                           canvas_width: int,
                           canvas_height: int,
                           scale: float,
                           position: Tuple[float, float]) -> np.ndarray:
    """
    Render the ChArUco board on a canvas with specified scale and position.
    
    This function handles the scaling (to simulate depth) and positioning
    (to simulate pan) of the calibration target on the display.
    
    Args:
        board_image: The original board image (grayscale)
        canvas_width: Width of the output canvas
        canvas_height: Height of the output canvas
        scale: Scale factor (0.0-1.0, where 1.0 = maximum fit)
        position: (x, y) position offset (0.0-1.0, where 0.5 = center)
    
    Returns:
        BGR canvas with the board rendered at the specified scale/position
    """
    # Create black canvas (BGR format for color display)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:] = BACKGROUND_COLOR
    
    # Calculate the maximum size that fits on screen with margins
    max_width = canvas_width - 2 * BOARD_MARGIN
    max_height = canvas_height - 2 * BOARD_MARGIN
    
    # Calculate aspect ratio of the board
    board_h, board_w = board_image.shape[:2]
    aspect_ratio = board_w / board_h
    
    # Calculate scaled dimensions maintaining aspect ratio
    if max_width / max_height > aspect_ratio:
        # Height is the limiting factor
        scaled_height = int(max_height * scale)
        scaled_width = int(scaled_height * aspect_ratio)
    else:
        # Width is the limiting factor
        scaled_width = int(max_width * scale)
        scaled_height = int(scaled_width / aspect_ratio)
    
    # Resize the board image
    scaled_board = cv2.resize(board_image, (scaled_width, scaled_height), 
                              interpolation=cv2.INTER_LINEAR)
    
    # Convert to BGR (the board image is grayscale)
    scaled_board_bgr = cv2.cvtColor(scaled_board, cv2.COLOR_GRAY2BGR)
    
    # Calculate position range (how far the board can move)
    x_range = canvas_width - scaled_width - 2 * BOARD_MARGIN
    y_range = canvas_height - scaled_height - 2 * BOARD_MARGIN
    
    # Ensure non-negative ranges
    x_range = max(0, x_range)
    y_range = max(0, y_range)
    
    # Calculate top-left position based on offset
    x_pos = int(BOARD_MARGIN + position[0] * x_range)
    y_pos = int(BOARD_MARGIN + position[1] * y_range)
    
    # Place the board on the canvas
    canvas[y_pos:y_pos+scaled_height, x_pos:x_pos+scaled_width] = scaled_board_bgr
    
    return canvas


def draw_info_overlay(canvas: np.ndarray, state: DisplayState, 
                      show_help: bool = True) -> np.ndarray:
    """
    Draw information overlay on the canvas showing current state.
    
    Args:
        canvas: The canvas to draw on
        state: Current display state
        show_help: Whether to show keyboard shortcuts
    
    Returns:
        Canvas with overlay drawn
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 255, 0)  # Green text
    thickness = 1
    
    # Status information (top-left)
    y_offset = 30
    info_lines = [
        f"Scale: {state.current_scale:.1%} (Level {state.scale_index + 1}/{len(SCALE_LEVELS)})",
        f"Position: ({state.current_position[0]:.2f}, {state.current_position[1]:.2f})",
        f"Auto-cycle: {'ON' if state.auto_cycle else 'OFF'}",
        f"Frame: {state.frame_count}",
    ]
    
    for line in info_lines:
        cv2.putText(canvas, line, (10, y_offset), font, font_scale, color, thickness)
        y_offset += 25
    
    # Help text (bottom-left)
    if show_help:
        help_lines = [
            "Keys: Q=Quit, A=Auto, S=Scale, P=Position, R=Reset",
            "+/-=Zoom, Arrows=Pan, SPACE=Log config"
        ]
        y_offset = canvas.shape[0] - 20
        for line in reversed(help_lines):
            cv2.putText(canvas, line, (10, y_offset), font, font_scale * 0.8, 
                       (128, 128, 128), thickness)
            y_offset -= 20
    
    return canvas


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def handle_keypress(key: int, state: DisplayState) -> Tuple[DisplayState, bool]:
    """
    Handle keyboard input and update state accordingly.
    
    Args:
        key: The key code from cv2.waitKey()
        state: Current display state
    
    Returns:
        Tuple of (updated state, should_quit flag)
    """
    should_quit = False
    
    if key == ord('q') or key == 27:  # 'q' or ESC
        should_quit = True
        
    elif key == ord('a'):  # Toggle auto-cycle
        state.auto_cycle = not state.auto_cycle
        state.last_cycle_time = time.time()
        print(f"[INFO] Auto-cycle: {'ENABLED' if state.auto_cycle else 'DISABLED'}")
        
    elif key == ord('s'):  # Next scale
        state.scale_index = (state.scale_index + 1) % len(SCALE_LEVELS)
        print(f"[INFO] Scale changed to {state.current_scale:.1%}")
        
    elif key == ord('p'):  # Next position preset
        state.use_custom_position = False
        state.position_index = (state.position_index + 1) % len(POSITION_PRESETS)
        print(f"[INFO] Position changed to {state.current_position}")
        
    elif key == ord('r'):  # Reset
        state.scale_index = len(SCALE_LEVELS) - 1  # Max scale
        state.position_index = 0  # Center
        state.use_custom_position = False
        print("[INFO] Reset to default scale and position")
        
    elif key == ord('+') or key == ord('='):  # Zoom in
        if state.scale_index < len(SCALE_LEVELS) - 1:
            state.scale_index += 1
            print(f"[INFO] Zoom in: {state.current_scale:.1%}")
            
    elif key == ord('-'):  # Zoom out
        if state.scale_index > 0:
            state.scale_index -= 1
            print(f"[INFO] Zoom out: {state.current_scale:.1%}")
            
    elif key == 81 or key == 2424832:  # Left arrow
        state.use_custom_position = True
        state.custom_offset_x = max(0.0, state.custom_offset_x - 0.05)
        state.custom_offset_y = state.current_position[1] if not state.use_custom_position else state.custom_offset_y
        
    elif key == 83 or key == 2555904:  # Right arrow
        state.use_custom_position = True
        state.custom_offset_x = min(1.0, state.custom_offset_x + 0.05)
        
    elif key == 82 or key == 2490368:  # Up arrow
        state.use_custom_position = True
        state.custom_offset_y = max(0.0, state.custom_offset_y - 0.05)
        
    elif key == 84 or key == 2621440:  # Down arrow
        state.use_custom_position = True
        state.custom_offset_y = min(1.0, state.custom_offset_y + 0.05)
        
    elif key == ord(' '):  # Space - log configuration
        config = {
            'frame': state.frame_count,
            'scale': state.current_scale,
            'position': state.current_position,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        print(f"[CONFIG] {json.dumps(config)}")
    
    return state, should_quit


def update_auto_cycle(state: DisplayState, rng: np.random.Generator) -> DisplayState:
    """
    Update state for automatic cycling mode.
    
    This function handles the automatic transitioning between scales
    and positions when auto-cycle mode is enabled.
    
    Args:
        state: Current display state
        rng: NumPy random generator for reproducible sequences
    
    Returns:
        Updated display state
    """
    if not state.auto_cycle:
        return state
    
    current_time = time.time()
    if current_time - state.last_cycle_time >= AUTO_CYCLE_INTERVAL:
        state.last_cycle_time = current_time
        
        # Randomly decide whether to change scale, position, or both
        action = rng.choice(['scale', 'position', 'both'], p=[0.3, 0.3, 0.4])
        
        if action in ['scale', 'both']:
            state.scale_index = int(rng.integers(0, len(SCALE_LEVELS)))
            
        if action in ['position', 'both']:
            state.use_custom_position = False
            state.position_index = int(rng.integers(0, len(POSITION_PRESETS)))
        
        print(f"[AUTO] Scale: {state.current_scale:.1%}, Position: {state.current_position}")
    
    return state


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main entry point for the ChArUco display application.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Display ChArUco calibration target on secondary monitor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python display_pattern.py                    # Use primary monitor
    python display_pattern.py --monitor 1       # Use secondary monitor
    python display_pattern.py --auto            # Start with auto-cycling
    python display_pattern.py --seed 123        # Use specific random seed
        """
    )
    parser.add_argument('--monitor', type=int, default=0,
                       help='Monitor index to display on (default: 0)')
    parser.add_argument('--auto', action='store_true',
                       help='Start with automatic cycling enabled')
    parser.add_argument('--seed', type=int, default=DEFAULT_RANDOM_SEED,
                       help=f'Random seed for reproducibility (default: {DEFAULT_RANDOM_SEED})')
    parser.add_argument('--no-overlay', action='store_true',
                       help='Hide the information overlay')
    
    args = parser.parse_args()
    
    # Initialize random number generator with seed for reproducibility
    # This is CRITICAL for thesis - ensures same sequence of scales/positions
    rng = np.random.default_rng(args.seed)
    print(f"[INFO] Random seed: {args.seed} (for reproducibility)")
    
    # Create the ChArUco board
    print("[INFO] Creating ChArUco board...")
    board, aruco_dict = create_charuco_board()
    print(f"       Board size: {SQUARES_X}x{SQUARES_Y} squares")
    print(f"       Dictionary: DICT_6X6_250")
    
    # Create fullscreen window on specified monitor
    window_name = "ChArUco Calibration Target"
    monitor = create_fullscreen_window(window_name, args.monitor)
    
    # Generate high-resolution board image
    # Use monitor resolution for maximum quality
    board_image = generate_board_image(board, monitor['width'], monitor['height'])
    print(f"[INFO] Board image generated: {board_image.shape[1]}x{board_image.shape[0]} pixels")
    
    # Initialize display state
    state = DisplayState()
    state.auto_cycle = args.auto
    if args.auto:
        print("[INFO] Auto-cycling enabled at startup")
    
    print("\n" + "="*60)
    print("ChArUco Display Running - Press 'Q' to quit")
    print("="*60 + "\n")
    
    # Main display loop
    try:
        while True:
            # Update auto-cycle if enabled
            state = update_auto_cycle(state, rng)
            
            # Render the board with current scale and position
            canvas = render_board_on_canvas(
                board_image,
                monitor['width'],
                monitor['height'],
                state.current_scale,
                state.current_position
            )
            
            # Add information overlay unless disabled
            if not args.no_overlay:
                canvas = draw_info_overlay(canvas, state)
            
            # Display the frame
            cv2.imshow(window_name, canvas)
            state.frame_count += 1
            
            # Handle keyboard input (30ms wait = ~33 FPS max)
            key = cv2.waitKey(30) & 0xFF
            if key != 255:  # A key was pressed
                state, should_quit = handle_keypress(key, state)
                if should_quit:
                    break
                    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        print("[INFO] Display closed")


if __name__ == "__main__":
    main()
