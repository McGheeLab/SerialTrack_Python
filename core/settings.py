"""
Application settings and Dracula theme constants.
"""


class Settings:
    # App
    APP_NAME = "SerialTrack"
    APP_VERSION = "2.0.0"
    APP_DESCRIPTION = "Particle Tracking & Traction Force Microscopy"

    # Window
    ENABLE_CUSTOM_TITLE_BAR = True
    MENU_WIDTH = 52          # Icon-only collapsed sidebar
    MENU_WIDTH_EXPANDED = 220  # Expanded sidebar (on hover)
    LEFT_BOX_WIDTH = 240
    RIGHT_BOX_WIDTH = 280
    TIME_ANIMATION = 400
    MIN_WIDTH = 1100
    MIN_HEIGHT = 700

    # Dracula Palette
    BG_PRIMARY = "#282a36"       # Main background
    BG_SECONDARY = "#21252b"     # Sidebar / darker panels
    BG_TERTIARY = "#2c313c"      # Cards / panels
    BG_HOVER = "#343b48"         # Hover state
    FG_PRIMARY = "#f8f8f2"       # Primary text
    FG_SECONDARY = "#b0b0b0"     # Secondary text
    ACCENT_PURPLE = "#bd93f9"    # Primary accent
    ACCENT_PINK = "#ff79c6"      # Secondary accent
    ACCENT_GREEN = "#50fa7b"     # Success
    ACCENT_CYAN = "#8be9fd"      # Info
    ACCENT_ORANGE = "#ffb86c"    # Warning
    ACCENT_RED = "#ff5555"       # Error
    ACCENT_YELLOW = "#f1fa8c"    # Highlight
    BORDER_COLOR = "#44475a"     # Borders

    # Menu
    BTN_LEFT_BOX_COLOR = f"background-color: {BG_TERTIARY};"
    BTN_RIGHT_BOX_COLOR = f"background-color: {ACCENT_PINK};"
    MENU_SELECTED_STYLESHEET = f"""
    border-left: 3px solid {ACCENT_PURPLE};
    background-color: {BG_TERTIARY};
    """

    # Tab names
    TAB_IMAGES = "Images"
    TAB_MASK = "Mask"
    TAB_DETECTION = "Detection"
    TAB_PARAMETERS = "Parameters"
    TAB_ANALYSIS = "Analysis"
    TAB_POSTPROCESS = "Post-Process"
    TAB_STRESS = "Stress"
    TAB_PLOTS = "Plots"