"""Qt stylesheet and design tokens for the application interface."""

# -- Palette ----------------------------------------------------------------
ACCENT = "#0C5DA5"
ACCENT_HOVER = "#0A4C87"
ACCENT_DEEP = "#083D6E"
ACCENT_WASH = "#EDF3F9"
ACCENT_WASH_2 = "#DCE8F4"

VERMILION = "#C0441A"
VERMILION_WASH = "#FBEFEA"

INK = "#12151A"
INK_SOFT = "#384049"
MUTED = "#6A7280"
FAINT = "#9BA1AA"

PAPER = "#FFFFFF"
PAPER_TINT = "#F7F8FA"
PAPER_TINT_2 = "#FAFBFC"
RULE = "#E3E6EA"

# Run-log console: black terminal surface, deliberately the one dark element
# in the interface - a log is machine output, not typeset copy.
CONSOLE_BG = "#000000"
CONSOLE_INK = "#D6DAE0"
CONSOLE_SELECT = "#1F4F7A"
RULE_STRONG = "#C8CDD4"

# -- Type stacks ------------------------------------------------------------
SERIF_STACK = '"Source Serif 4", "Source Serif Pro", "Charter", "Georgia", "Times New Roman", serif'
FONT_STACK = '"Inter", "Segoe UI", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif'
MONO_STACK = '"JetBrains Mono", "Consolas", "SF Mono", "Courier New", monospace'

# -- Backwards-compatible aliases (kept so older call sites keep working) ---
NATURE_BLUE = ACCENT
NATURE_BLUE_HOVER = ACCENT_HOVER
NATURE_BLUE_WASH = ACCENT_WASH
NATURE_RED = VERMILION
NATURE_RED_WASH = VERMILION_WASH
NATURE_GREY = INK_SOFT
LIGHT_GREY = PAPER_TINT
BORDER = RULE
BORDER_GREY = RULE
BORDER_STRONG = RULE_STRONG
SIDE_PANEL = PAPER_TINT

QSS = f"""
* {{
    font-family: {FONT_STACK};
    color: {INK_SOFT};
}}

QMainWindow, QDialog {{
    background-color: {PAPER};
}}

QToolTip {{
    background-color: {PAPER};
    color: {INK_SOFT};
    border: 1px solid {RULE_STRONG};
    padding: 7px 9px;
    font-size: 11px;
}}

/* == Side column ======================================================= */
QWidget#SidePanel {{
    background-color: {PAPER_TINT};
    border-right: 1px solid {RULE};
}}
QWidget#SidePanelBody {{
    background-color: {PAPER_TINT};
}}
/* Pinned footer holding RUN PIPELINE. A hairline separates it from the
   scrolling body, so a partially scrolled column reads as cut off rather
   than as ending there. */
QWidget#SidePanelFooter {{
    background-color: {PAPER_TINT};
    border-top: 1px solid {RULE};
}}

/* Masthead: serif wordmark over a hairline rule, as on a journal cover */
QLabel#Masthead {{
    font-family: {SERIF_STACK};
    font-size: 26px;
    font-weight: 600;
    color: {INK};
    letter-spacing: -0.015em;
    padding: 0px;
}}
QLabel#MastheadRule {{
    background-color: {INK};
    max-height: 2px;
    min-height: 2px;
    margin: 8px 0px 7px 0px;
}}
QLabel#MastheadSub {{
    font-size: 11px;
    color: {MUTED};
    line-height: 158%;
    padding-bottom: 4px;
}}

/* Section heads: small-caps tracking over a hairline, numbered by the caller */
QLabel#SectionLabel {{
    font-size: 10px;
    font-weight: 700;
    color: {ACCENT};
    letter-spacing: 0.14em;
    padding-top: 13px;
    padding-bottom: 5px;
    border-top: 1px solid {RULE};
    margin-top: 6px;
}}
QLabel#FieldLabel {{
    font-size: 11px;
    color: {MUTED};
    letter-spacing: 0.01em;
    padding-bottom: 1px;
}}
QLabel#Caption {{
    font-size: 10.5px;
    color: {MUTED};
    line-height: 150%;
}}
QLabel#CaptionMono {{
    font-family: {MONO_STACK};
    font-size: 9.5px;
    color: {FAINT};
    letter-spacing: 0.03em;
}}

/* == Cards ============================================================= */
QGroupBox {{
    background-color: {PAPER};
    border: 1px solid {RULE};
    border-radius: 3px;
    margin-top: 10px;
    padding-top: 16px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: {MUTED};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 11px;
    padding: 0 6px;
    color: {MUTED};
}}
QGroupBox#ResourceCard {{
    padding-top: 8px;
}}
QGroupBox#PlateFrame {{
    background-color: {PAPER};
    border: 1px solid {RULE};
    padding-top: 18px;
}}

/* Status badges */
QLabel#StatusBadge {{
    font-family: {MONO_STACK};
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 4px 0px;
    border-radius: 2px;
    background-color: {PAPER_TINT};
    color: {FAINT};
    border: 1px solid {RULE};
}}
QLabel#StatusBadge[state="ok"] {{
    background-color: {ACCENT_WASH};
    color: {ACCENT};
    border: 1px solid {ACCENT_WASH_2};
}}
QLabel#StatusBadge[state="missing"] {{
    background-color: {VERMILION_WASH};
    color: {VERMILION};
    border: 1px solid #F0DAD2;
}}
QLabel#ResourceName {{
    font-size: 11px;
    color: {INK_SOFT};
}}

/* == Buttons =========================================================== */
QPushButton {{
    background-color: {PAPER};
    border: 1px solid {RULE_STRONG};
    border-radius: 3px;
    padding: 6px 14px;
    font-size: 11.5px;
    color: {INK_SOFT};
}}
QPushButton:hover {{
    border-color: {ACCENT};
    color: {ACCENT};
    background-color: {ACCENT_WASH};
}}
QPushButton:pressed {{
    background-color: {ACCENT_WASH_2};
}}
QPushButton:disabled {{
    color: {FAINT};
    border-color: {RULE};
    background-color: {PAPER_TINT};
}}

QPushButton#PrimaryButton {{
    background-color: {ACCENT};
    color: #FFFFFF;
    border: 1px solid {ACCENT};
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 0.09em;
    padding: 11px 20px;
    border-radius: 3px;
}}
QPushButton#PrimaryButton:hover {{
    background-color: {ACCENT_HOVER};
    border-color: {ACCENT_HOVER};
}}
QPushButton#PrimaryButton:pressed {{
    background-color: {ACCENT_DEEP};
}}
QPushButton#PrimaryButton:disabled {{
    background-color: #A9C3DC;
    border-color: #A9C3DC;
    color: #F2F5F8;
}}

/* Quiet text button - for secondary affordances such as "Nomenclature" */
QPushButton#LinkButton {{
    background: transparent;
    border: none;
    padding: 3px 0px;
    color: {ACCENT};
    font-size: 10.5px;
    text-align: left;
}}
QPushButton#LinkButton:hover {{
    color: {ACCENT_HOVER};
    text-decoration: underline;
    background: transparent;
}}

/* == Inputs ============================================================ */
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
    border: 1px solid {RULE_STRONG};
    border-radius: 3px;
    padding: 5px 9px;
    background: {PAPER};
    font-size: 11.5px;
    color: {INK};
    selection-background-color: {ACCENT_WASH_2};
    selection-color: {INK};
}}
QDoubleSpinBox, QSpinBox {{
    font-family: {MONO_STACK};
    font-size: 11px;
}}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{
    border: 1px solid {ACCENT};
}}
QLineEdit:hover, QComboBox:hover {{
    border-color: {FAINT};
}}
QLineEdit:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled {{
    background: {PAPER_TINT};
    color: {FAINT};
}}
QLineEdit::placeholder {{
    color: {FAINT};
}}
/* Spin-box / combo-box arrows are deliberately left to Qt's own rendering:
   hand-drawn CSS-triangle sub-controls consistently look worse than the
   native ones once a global stylesheet is active. Only the surrounding
   field is themed. */
QCheckBox {{
    font-size: 11.5px;
    color: {INK_SOFT};
    spacing: 9px;
    padding: 2px 0px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {RULE_STRONG};
    border-radius: 2px;
    background: {PAPER};
}}
QCheckBox::indicator:hover {{
    border-color: {ACCENT};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}

/* == Tabs - flat, underlined, no chrome ================================ */
QTabWidget::pane {{
    border: none;
    border-top: 1px solid {RULE};
    top: -1px;
    background: {PAPER};
}}
QTabBar {{
    qproperty-drawBase: 0;
}}
QTabBar::tab {{
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 9px 2px;
    margin-right: 22px;
    font-size: 11.5px;
    letter-spacing: 0.02em;
    color: {MUTED};
}}
QTabBar::tab:selected {{
    color: {INK};
    font-weight: 600;
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover:!selected {{
    color: {INK_SOFT};
}}

/* == Tables - journal rules: top rule, header rule, bottom rule ======== */
QTableView, QTableWidget {{
    background: {PAPER};
    gridline-color: transparent;
    border: none;
    border-top: 2px solid {INK};
    border-bottom: 2px solid {INK};
    selection-background-color: {ACCENT_WASH};
    selection-color: {INK};
    font-size: 11px;
    alternate-background-color: {PAPER_TINT_2};
    color: {INK_SOFT};
}}
QTableView::item, QTableWidget::item {{
    padding: 4px 10px;
    border: none;
    border-bottom: 1px solid #EFF1F3;
}}
QHeaderView {{
    background: {PAPER};
}}
/* Font properties are deliberately omitted here. A stylesheet that sets any
   font property on QHeaderView::section overrides per-item QFont settings, and
   app.py needs item-level control to set descriptor headers in italic while
   leaving identifier headers (Zeolite, Direction) roman. Weight and size are
   applied there instead. */
QHeaderView::section {{
    background-color: {PAPER};
    padding: 8px 10px;
    border: none;
    border-bottom: 1px solid {INK};
    color: {INK};
}}
QHeaderView::section:vertical {{
    border-bottom: 1px solid {RULE};
    border-right: 1px solid {RULE};
    color: {FAINT};
    font-family: {MONO_STACK};
    font-size: 9px;
    font-weight: 400;
}}
QTableCornerButton::section {{
    background: {PAPER};
    border: none;
    border-bottom: 1px solid {INK};
}}

QListWidget {{
    background: {PAPER};
    border: 1px solid {RULE};
    border-radius: 3px;
    font-size: 11px;
    padding: 3px;
}}
QListWidget::item {{
    padding: 6px 8px;
    border-radius: 2px;
}}
QListWidget::item:selected {{
    background: {ACCENT_WASH};
    color: {INK};
}}
QListWidget::item:hover:!selected {{
    background: {PAPER_TINT};
}}

/* == Progress - a thin rule that fills, not a chunky bar =============== */
QProgressBar {{
    border: none;
    background: {RULE};
    height: 3px;
    max-height: 3px;
    border-radius: 2px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 2px;
}}
QLabel#StageLabel {{
    font-size: 10.5px;
    color: {INK_SOFT};
    letter-spacing: 0.02em;
}}
QLabel#StageCounter {{
    font-family: {MONO_STACK};
    font-size: 9.5px;
    color: {FAINT};
    letter-spacing: 0.06em;
}}
QWidget#StatusStrip {{
    background: {PAPER};
}}

/* == Run log - black console, monospaced =============================== */
QPlainTextEdit#LogConsole {{
    background-color: {CONSOLE_BG};
    color: {CONSOLE_INK};
    font-family: {MONO_STACK};
    font-size: 11.5px;
    border: 1px solid {CONSOLE_BG};
    border-radius: 3px;
    padding: 14px 16px;
    selection-background-color: {CONSOLE_SELECT};
    selection-color: #FFFFFF;
}}

/* == Scrollbars ======================================================== */
QScrollBar:vertical {{
    width: 9px;
    background: transparent;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: {RULE_STRONG};
    border-radius: 4px;
    min-height: 26px;
}}
QScrollBar::handle:vertical:hover {{
    background: {FAINT};
}}
QScrollBar:horizontal {{
    height: 9px;
    background: transparent;
}}
QScrollBar::handle:horizontal {{
    background: {RULE_STRONG};
    border-radius: 4px;
    min-width: 26px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {FAINT};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    width: 0px; height: 0px; border: none; background: none;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}
QScrollArea {{
    background: transparent;
    border: none;
}}

/* == Figure plate - the waterfall panel, framed like a printed figure == */
QWidget#PlateCanvas {{
    background: {PAPER};
    border: 1px solid {RULE};
}}
QLabel#PlatePlaceholder {{
    color: {FAINT};
    font-size: 11px;
    letter-spacing: 0.02em;
}}

/* == Nomenclature dialog =============================================== */
QLabel#GlossaryTitle {{
    font-family: {SERIF_STACK};
    font-size: 20px;
    font-weight: 600;
    color: {INK};
}}
QLabel#GlossaryFamily {{
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 0.14em;
    color: {ACCENT};
    padding-top: 16px;
    padding-bottom: 4px;
    border-top: 1px solid {RULE};
}}
QLabel#GlossarySymbol {{
    font-size: 15px;
    font-weight: 600;
    color: {INK};
}}
QLabel#GlossaryName {{
    font-size: 11.5px;
    font-weight: 600;
    color: {INK};
}}
QLabel#GlossaryUnit {{
    font-family: {MONO_STACK};
    font-size: 9.5px;
    color: {MUTED};
}}
QLabel#GlossaryDef {{
    font-size: 11px;
    color: {INK_SOFT};
}}
QLabel#GlossaryKey {{
    font-family: {MONO_STACK};
    font-size: 9px;
    color: {FAINT};
}}
QWidget#GlossaryBody {{
    background: {PAPER};
}}
QWidget#GlossaryRow {{
    background: {PAPER};
    border-bottom: 1px solid {RULE};
}}

/* == DeepDiffusion-X assistant panel =================================== */
QWidget#ChatHeader, QWidget#ChatHeader QLabel,
QWidget#ChatControlBar, QWidget#ChatControlBar QLabel, QWidget#ChatControlBar QComboBox,
QWidget#ChatCanvas, QWidget#ChatCanvas QLabel,
QWidget#ChatInputBar, QWidget#ChatInputBar QLineEdit {{
    font-family: {FONT_STACK};
}}

QWidget#ChatHeader {{
    background-color: {PAPER};
    border-bottom: 1px solid {RULE};
}}
QLabel#ChatAssistantName {{
    font-family: {SERIF_STACK};
    font-size: 17px;
    font-weight: 600;
    color: {INK};
    letter-spacing: -0.01em;
}}
QLabel#ChatAssistantSubtitle {{
    font-size: 10.5px;
    color: {MUTED};
}}
QWidget#ChatControlBar {{
    background-color: {PAPER_TINT};
    border-bottom: 1px solid {RULE};
}}
QWidget#ChatControlBar QLabel {{
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: {MUTED};
}}
QWidget#ChatCanvas {{
    background-color: {PAPER};
}}
QScrollArea#ChatScroll {{
    border: none;
    background: {PAPER};
}}
QLabel#ChatIdleLabel {{
    color: {FAINT};
    font-size: 11px;
    letter-spacing: 0.03em;
    padding: 26px 0px;
}}

/* Bubbles read as annotated manuscript blocks, not chat pills: the query is
   a tinted card, the reply a paper card with an accent spine; no colour is
   used for semantic emphasis anywhere in this palette. */
QWidget#UserBubble {{
    background-color: {PAPER_TINT};
    color: {INK};
    border: 1px solid {RULE};
    border-radius: 3px;
    font-size: 11.5px;
}}
QWidget#UserBubble QLabel {{
    background: transparent;
    color: {INK};
    font-size: 11.5px;
}}
QWidget#AssistantBubble {{
    background-color: {PAPER};
    color: {INK};
    border: 1px solid {RULE};
    border-left: 2px solid {ACCENT};
    border-radius: 3px;
    font-size: 12px;
}}
QWidget#AssistantBubble QLabel {{
    background: transparent;
    color: {INK};
    font-size: 12px;
}}
QWidget#ThinkingBubble {{
    background-color: {PAPER};
    color: {FAINT};
    border: 1px solid {RULE};
    border-left: 2px solid {RULE_STRONG};
    border-radius: 3px;
    font-size: 11px;
}}
QWidget#ThinkingBubble QLabel {{
    background: transparent;
    color: {FAINT};
    font-size: 11px;
    font-style: italic;
}}
QWidget#ChatInputBar {{
    background-color: {PAPER};
    border-top: 1px solid {RULE};
}}
"""
