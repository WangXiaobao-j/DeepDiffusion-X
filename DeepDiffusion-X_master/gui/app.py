"""PySide6 interface of the DeepDiffusion-X zeolite diffusivity prediction platform."""
import os
import sys
import traceback

from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QPixmap, QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QGroupBox, QDoubleSpinBox,
    QSpinBox, QComboBox, QTabWidget, QTableWidget, QTableWidgetItem, QProgressBar,
    QPlainTextEdit, QScrollArea, QCheckBox, QMessageBox, QListWidget, QListWidgetItem,
    QSizePolicy, QFrame
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.pipeline import (PipelineConfig, run_full_pipeline, DEFAULT_EXTERNAL_EXCEL,
                            DEFAULT_MODEL_PATH, DEFAULT_SCALER_PATH)
from core import llm_chat
from core import descriptor_names as dn
from gui.style import QSS, ACCENT, INK, MUTED, RULE
from gui.glossary import NomenclatureDialog

# Build marker. No longer displayed in the side column; kept for the
# window title and for anyone grepping the source for a version.
APP_VERSION = "V2.0"


def _build_symbol_regex():
    """
    Match any descriptor symbol written in the flat form the model emits
    (E_a, sigma_E, P_tau, G_E,mean, rho_barrier ...) or in its Greek form,
    so the chat panel can typeset them with a real subscript. Longest
    alternatives first, and both edges guarded, so E_a never matches inside
    E_acc and PLD never matches inside PLDX.
    """
    import re
    variants = {}
    for key, d in dn.DESCRIPTORS.items():
        for form in (d.symbol, d.ascii):
            variants[form] = d.html
    pattern = "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z0-9_]){pattern}(?![A-Za-z0-9])"), variants


_SYMBOL_RE, _SYMBOL_HTML = _build_symbol_regex()


# ====================================================================
# Background thread: run the pipeline, forward stdout to the log console
# ====================================================================
class _StreamRedirect:
    def __init__(self, emit_fn):
        self._emit = emit_fn

    def write(self, text):
        if text.strip():
            self._emit(text.rstrip("\n"))

    def flush(self):
        pass


class PipelineWorker(QThread):
    log_line = Signal(str)
    stage_changed = Signal(str, float)
    finished_ok = Signal(object)
    finished_error = Signal(str)

    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs

    def run(self):
        old_stdout = sys.stdout
        sys.stdout = _StreamRedirect(self.log_line.emit)
        try:
            def log(msg):
                self.log_line.emit(str(msg))

            def stage_cb(name, frac):
                self.stage_changed.emit(name, frac)

            result = run_full_pipeline(log=log, stage_cb=stage_cb, **self.kwargs)
            self.finished_ok.emit(result)
        except Exception as exc:
            self.log_line.emit(traceback.format_exc())
            self.finished_error.emit(str(exc))
        finally:
            sys.stdout = old_stdout


# ====================================================================
# Background thread: send a chat message to the LLM API without blocking the UI
# ====================================================================
class ChatWorker(QThread):
    delta = Signal(str)          # incremental text fragment, as it arrives from the API
    reply_ok = Signal(str)       # full accumulated reply, once the stream completes
    reply_error = Signal(str)

    def __init__(self, api_key, model, history, context_summary):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.history = history
        self.context_summary = context_summary

    def run(self):
        accumulated = []
        try:
            for fragment in llm_chat.send_message_stream(
                    self.api_key, self.model, self.history, context_summary=self.context_summary):
                accumulated.append(fragment)
                self.delta.emit(fragment)
            self.reply_ok.emit("".join(accumulated))
        except Exception as exc:
            if accumulated:
                # partial reply already streamed to the UI; report the error
                # but keep what was received rather than discarding it
                self.reply_error.emit(f"{exc} (stream interrupted after partial reply)")
            else:
                self.reply_error.emit(str(exc))


# ====================================================================
# Small helper: path field with a browse button
# ====================================================================
class PathField(QWidget):
    def __init__(self, placeholder="", is_dir=False, filter_str="All files (*)"):
        super().__init__()
        self.is_dir = is_dir
        self.filter_str = filter_str
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        btn = QPushButton("Browse")
        btn.setFixedWidth(82)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(self._browse)
        layout.addWidget(self.edit)
        layout.addWidget(btn)

    def _browse(self):
        if self.is_dir:
            path = QFileDialog.getExistingDirectory(self, "Select Folder")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select File", "", self.filter_str)
        if path:
            self.edit.setText(path)

    def text(self):
        return self.edit.text().strip()

    def setText(self, t):
        self.edit.setText(t)


# ====================================================================
# Main window
# ====================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"DeepDiffusion-X  \u00b7  Zeolite Diffusivity Prediction Platform  \u00b7  {APP_VERSION}")
        self.resize(1400, 900)
        self.setMinimumSize(1120, 720)
        self.worker = None
        self.last_result = None

        self._build_ui()
        self._refresh_resource_status()

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_side_panel(), 0)

        right = QVBoxLayout()
        right.setContentsMargins(30, 22, 30, 22)
        right.setSpacing(0)
        right_widget = QWidget()
        right_widget.setLayout(right)
        root.addWidget(right_widget, 1)

        # -- Status strip: stage name left, counter right, hairline progress --
        strip = QWidget()
        strip.setObjectName("StatusStrip")
        strip_lay = QVBoxLayout(strip)
        strip_lay.setContentsMargins(0, 0, 0, 14)
        strip_lay.setSpacing(7)

        head_row = QHBoxLayout()
        head_row.setSpacing(10)
        self.stage_label = QLabel("Idle \u2014 select an input directory to begin")
        self.stage_label.setObjectName("StageLabel")
        self.stage_counter = QLabel("00 %")
        self.stage_counter.setObjectName("StageCounter")
        head_row.addWidget(self.stage_label, 1)
        head_row.addWidget(self.stage_counter, 0, Qt.AlignRight)
        strip_lay.addLayout(head_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(3)
        strip_lay.addWidget(self.progress)
        right.addWidget(strip)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        right.addWidget(self.tabs, 1)

        self.tab_log = self._build_log_tab()
        # No caption: the descriptor table stands on its own. Column meanings
        # remain available as header tooltips.
        self.tab_features = self._build_table_tab()
        self.tab_predictions = self._build_table_tab(
            "Self-diffusion coefficients inferred by the pathway-informed ANN, "
            "back-transformed from ln(Ds) to the original scale."
        )
        self.tab_shap = self._build_shap_tab()
        self.tab_html = self._build_html_tab()
        self.tab_chat = self._build_chat_tab()

        self.tabs.addTab(self.tab_log, "Run log")
        self.tabs.addTab(self.tab_features, "Descriptors")
        self.tabs.addTab(self.tab_predictions, "Diffusivity")
        self.tabs.addTab(self.tab_shap, "SHAP analysis")
        self.tabs.addTab(self.tab_html, "Pathways")
        self.tabs.addTab(self.tab_chat, "DeepDiffusion\u2011X")

    # ------------------------------------------------------------------
    def _build_side_panel(self):
        """
        The column is a scroll area with a *pinned* footer: everything from the
        masthead to the last parameter scrolls, RUN PIPELINE does not. Keeping
        the button outside the scroll area means it can never be clipped by a
        short window or by Windows display scaling, which is what happened when
        the whole column was one scrolling widget.
        """
        container = QWidget()
        container.setObjectName("SidePanel")
        container.setFixedWidth(400)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        panel = QWidget()
        panel.setObjectName("SidePanelBody")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll, 1)

        v = QVBoxLayout(panel)
        v.setContentsMargins(26, 20, 26, 14)
        v.setSpacing(4)

        # -- Masthead: serif wordmark, hairline rule, standfirst --
        title = QLabel("DeepDiffusion\u2011X")
        title.setObjectName("Masthead")
        rule = QLabel()
        rule.setObjectName("MastheadRule")
        rule.setFixedHeight(2)
        subtitle = QLabel(
            "Pathway-resolved descriptors and interpretable machine learning "
            "for anisotropic guest diffusion in zeolites."
        )
        subtitle.setObjectName("MastheadSub")
        subtitle.setWordWrap(True)
        v.addWidget(title)
        v.addWidget(rule)
        v.addWidget(subtitle)
        v.addSpacing(4)

        # -- I. Input / output --
        v.addWidget(self._section("I \u00b7 Input & Output"))
        self.f_cif_dir = PathField("Directory containing input CIF files", is_dir=True)
        self.f_work_dir = PathField("Working directory for all output", is_dir=True)
        v.addWidget(self._labeled(self.f_cif_dir, "CIF directory"))
        v.addWidget(self._labeled(self.f_work_dir, "Working directory"))

        v.addSpacing(6)
        self.chk_skip_symmetry = QCheckBox("Skip symmetry removal (CIFs already P1)")
        self.chk_skip_symmetry.setChecked(True)
        v.addWidget(self.chk_skip_symmetry)
        self.chk_save_html = QCheckBox("Generate interactive pathway visualizations")
        self.chk_save_html.setChecked(True)
        v.addWidget(self.chk_save_html)

        # -- II. Bundled resources status --
        v.addWidget(self._section("II \u00b7 Bundled Resources"))
        self.resource_box = QGroupBox()
        self.resource_box.setObjectName("ResourceCard")
        res_lay = QVBoxLayout(self.resource_box)
        res_lay.setSpacing(8)
        res_lay.setContentsMargins(13, 10, 13, 10)
        row_excel, self.badge_excel, self.name_excel = self._make_resource_row("External Excel")
        row_model, self.badge_model, self.name_model = self._make_resource_row("ANN Model")
        row_scaler, self.badge_scaler, self.name_scaler = self._make_resource_row("Scaler")
        for row_widget in (row_excel, row_model, row_scaler):
            res_lay.addWidget(row_widget)
        v.addWidget(self.resource_box)

        # -- III. Energy grid & pathway parameters --
        v.addWidget(self._section("III \u00b7 Energy Grid & Pathway"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(12)
        grid.setColumnStretch(0, 1)

        self.sp_spacing = self._dspin(0.05, 1.0, 0.2, 0.05)
        self.sp_cutoff = self._dspin(5.0, 30.0, 14.0, 1.0)
        self.cb_probe = QComboBox()
        self.cb_probe.addItems(["ch4", "xe"])
        self.sp_supercell = self._dspin(10.0, 60.0, 28.0, 1.0)
        self.sp_ethresh = self._dspin(50.0, 1000.0, 300.0, 10.0)

        params = [
            ("Grid spacing", "\u212b", self.sp_spacing,
             "Lennard-Jones energy grid resolution."),
            ("Interaction cut-off", "\u212b", self.sp_cutoff,
             "Real-space cut-off for the guest\u2013framework LJ summation."),
            ("Probe species", "", self.cb_probe,
             "United-atom guest used to sample the energy landscape."),
            ("Min. supercell length", "\u212b", self.sp_supercell,
             "Lower bound on each supercell edge before grid construction."),
            ("Accessibility threshold", "kJ mol\u207b\u00b9", self.sp_ethresh,
             "Grid points above this energy are treated as inaccessible."),
        ]
        for r, (name, unit, widget, tip) in enumerate(params):
            lbl = QLabel(f"{name}  <span style='color:{MUTED}'>{unit}</span>" if unit else name)
            lbl.setObjectName("FieldLabel")
            lbl.setTextFormat(Qt.RichText)
            lbl.setToolTip(tip)
            widget.setToolTip(tip)
            widget.setFixedWidth(112)
            grid.addWidget(lbl, r, 0)
            grid.addWidget(widget, r, 1)
        v.addLayout(grid)

        v.addStretch(1)

        # -- Pinned footer, outside the scroll area --
        footer = QWidget()
        footer.setObjectName("SidePanelFooter")
        f_lay = QVBoxLayout(footer)
        f_lay.setContentsMargins(26, 12, 26, 16)
        self.btn_run = QPushButton("RUN PIPELINE")
        self.btn_run.setObjectName("PrimaryButton")
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.clicked.connect(self.on_run)
        f_lay.addWidget(self.btn_run)
        outer.addWidget(footer, 0)

        return container

    def _section(self, text):
        lbl = QLabel(text.upper())
        lbl.setObjectName("SectionLabel")
        return lbl

    def _labeled(self, widget, label_text):
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 3, 0, 3)
        lay.setSpacing(4)
        lbl = QLabel(label_text)
        lbl.setObjectName("FieldLabel")
        lay.addWidget(lbl)
        lay.addWidget(widget)
        return wrap

    def on_show_nomenclature(self):
        """
        Open (or raise) the descriptor symbol reference sheet.

        Currently unwired: no control in the interface calls this. It is kept,
        along with gui/glossary.py, so the panel can be brought back with one
        line -- a QPushButton with objectName "LinkButton" connected here,
        placed wherever you want it. Column meanings are still reachable in the
        meantime as table header tooltips.
        """
        if getattr(self, "_nomenclature_dialog", None) is None:
            self._nomenclature_dialog = NomenclatureDialog(self)
        self._nomenclature_dialog.show()
        self._nomenclature_dialog.raise_()
        self._nomenclature_dialog.activateWindow()

    def _dspin(self, lo, hi, val, step):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSingleStep(step)
        return s

    def _ispin(self, lo, hi, val, step):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSingleStep(step)
        return s

    def _make_resource_row(self, label_text: str):
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        badge = QLabel("...")
        badge.setObjectName("StatusBadge")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedWidth(68)
        name = QLabel(label_text)
        name.setObjectName("ResourceName")
        lay.addWidget(badge)
        lay.addWidget(name, 1)
        return row, badge, name

    def _refresh_resource_status(self):
        specs = [
            (self.badge_excel, self.name_excel, "External Excel", DEFAULT_EXTERNAL_EXCEL),
            (self.badge_model, self.name_model, "ANN Model", DEFAULT_MODEL_PATH),
            (self.badge_scaler, self.name_scaler, "Scaler", DEFAULT_SCALER_PATH),
        ]
        for badge, name, label, path in specs:
            ok = os.path.exists(path)
            badge.setText("OK" if ok else "MISSING")
            badge.setProperty("state", "ok" if ok else "missing")
            badge.style().unpolish(badge)
            badge.style().polish(badge)
            filename = os.path.basename(path)
            name.setText(f"{label}: {filename}")
            name.setToolTip(path)

    # ------------------------------------------------------------------
    @staticmethod
    def _caption(text: str) -> QLabel:
        """Figure/table caption: muted, wrapped, set below the tab rule."""
        lbl = QLabel(text)
        lbl.setObjectName("Caption")
        lbl.setWordWrap(True)
        return lbl

    def _build_log_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 16, 0, 0)
        lay.setSpacing(10)
        self.log_console = QPlainTextEdit()
        self.log_console.setObjectName("LogConsole")
        self.log_console.setReadOnly(True)
        lay.addWidget(self.log_console, 1)
        return w

    def _build_table_tab(self, caption: str = ""):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 16, 0, 0)
        lay.setSpacing(10)
        if caption:
            lay.addWidget(self._caption(caption))
        table = QTableWidget()
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setShowGrid(False)
        table.verticalHeader().setDefaultSectionSize(26)
        table.horizontalHeader().setHighlightSections(False)
        table.verticalHeader().setHighlightSections(False)
        lay.addWidget(table, 1)
        w.table = table
        return w

    def _build_shap_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 16, 0, 0)
        lay.setSpacing(10)

        lay.addWidget(self._caption(
            "Single-sample SHAP decomposition of diffusion predictions for representative "
            "zeolite structures, illustrating the contribution of individual descriptors "
            "to the predicted diffusion coefficients."))

        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(10)
        sample_lbl = QLabel("SAMPLE")
        sample_lbl.setObjectName("CaptionMono")
        ctrl_row.addWidget(sample_lbl)
        self.shap_sample_combo = QComboBox()
        self.shap_sample_combo.setMinimumWidth(220)
        ctrl_row.addWidget(self.shap_sample_combo, 0)
        self.btn_gen_waterfall = QPushButton("Plot attribution")
        self.btn_gen_waterfall.clicked.connect(self.on_generate_waterfall)
        ctrl_row.addWidget(self.btn_gen_waterfall)
        ctrl_row.addStretch(1)
        lay.addLayout(ctrl_row)

        # Figure plate: a bordered white canvas with a caption beneath, in the
        # proportions of a printed single-column figure.
        plate = QWidget()
        plate.setObjectName("PlateCanvas")
        plate_lay = QVBoxLayout(plate)
        plate_lay.setContentsMargins(18, 18, 18, 18)
        self.waterfall_label = QLabel("No attribution plotted yet")
        self.waterfall_label.setObjectName("PlatePlaceholder")
        self.waterfall_label.setAlignment(Qt.AlignCenter)
        self.waterfall_label.setMinimumHeight(300)
        plate_lay.addWidget(self.waterfall_label, 1)

        plate_scroll = QScrollArea()
        plate_scroll.setWidgetResizable(True)
        plate_scroll.setFrameShape(QScrollArea.NoFrame)
        plate_scroll.setWidget(plate)
        lay.addWidget(plate_scroll, 1)

        return w

    def _build_html_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 16, 0, 0)
        lay.setSpacing(10)
        lay.addWidget(self._caption(
            "Three-dimensional visualization of molecular diffusion pathways within "
            "zeolite frameworks."))
        self.html_list = QListWidget()
        self.html_list.itemDoubleClicked.connect(self._open_html_item)
        lay.addWidget(self.html_list, 1)
        btn_row = QHBoxLayout()
        open_btn = QPushButton("Open in browser")
        open_btn.clicked.connect(lambda: self._open_html_item(self.html_list.currentItem()))
        btn_row.addWidget(open_btn)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)
        return w

    def _open_html_item(self, item: QListWidgetItem):
        if item is None:
            return
        path = item.data(Qt.UserRole)
        if path and os.path.exists(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _fill_html_tab(self, html_paths: list):
        self.html_list.clear()
        for p in html_paths:
            item = QListWidgetItem(os.path.relpath(p, start=os.path.dirname(os.path.dirname(p))))
            item.setData(Qt.UserRole, p)
            item.setToolTip(p)
            self.html_list.addItem(item)

    # ------------------------------------------------------------------
    def _build_chat_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        header = QWidget()
        header.setObjectName("ChatHeader")
        hlay = QVBoxLayout(header)
        hlay.setContentsMargins(20, 14, 20, 12)
        hlay.setSpacing(2)
        name_row = QHBoxLayout()
        name_lbl = QLabel(llm_chat.ASSISTANT_NAME)
        name_lbl.setObjectName("ChatAssistantName")
        name_row.addWidget(name_lbl)
        name_row.addStretch(1)
        hlay.addLayout(name_row)
        subtitle_lbl = QLabel(f"{llm_chat.ASSISTANT_FULL_NAME} \u2014 zeolite & porous-material diffusion")
        subtitle_lbl.setObjectName("ChatAssistantSubtitle")
        hlay.addWidget(subtitle_lbl)
        lay.addWidget(header)

        ctrl_row = QHBoxLayout()
        ctrl_row.setContentsMargins(20, 8, 20, 8)
        ctrl_row.addWidget(QLabel("Model"))
        self.chat_model_combo = QComboBox()
        self.chat_model_combo.addItems(llm_chat.AVAILABLE_MODELS)
        ctrl_row.addWidget(self.chat_model_combo)
        ctrl_row.addSpacing(12)
        ctrl_row.addWidget(QLabel("API Key"))
        self.chat_api_key = QLineEdit()
        self.chat_api_key.setEchoMode(QLineEdit.Password)
        self.chat_api_key.setPlaceholderText("sk-... (or set DEEPSEEK_API_KEY)")
        self.chat_api_key.setText(os.environ.get("DEEPSEEK_API_KEY", ""))
        ctrl_row.addWidget(self.chat_api_key, 1)
        self.btn_clear_chat = QPushButton("Clear")
        self.btn_clear_chat.clicked.connect(self.on_clear_chat)
        ctrl_row.addWidget(self.btn_clear_chat)
        ctrl_wrap = QWidget()
        ctrl_wrap.setObjectName("ChatControlBar")
        ctrl_wrap.setLayout(ctrl_row)
        lay.addWidget(ctrl_wrap)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setObjectName("ChatScroll")

        # Centering is done via computed side margins on this layout
        # (see _apply_chat_center_margins), rather than nested
        # stretch/max-width widgets -- Qt's box-layout stretch
        # distribution doesn't reliably respect a child's maximumWidth
        # as a growth target, so a fixed content width computed directly
        # from the viewport size is far more predictable.
        chat_inner = QWidget()
        chat_inner.setObjectName("ChatCanvas")
        self.chat_messages_layout = QVBoxLayout(chat_inner)
        self.chat_messages_layout.setContentsMargins(20, 16, 20, 16)
        self.chat_messages_layout.setSpacing(16)

        self.chat_idle_label = QLabel(
            "Grounded in the current session\u2019s descriptors, predictions and "
            "attributions. State a framework, a direction, or a comparison.")
        self.chat_idle_label.setObjectName("ChatIdleLabel")
        self.chat_idle_label.setAlignment(Qt.AlignCenter)
        self.chat_messages_layout.addWidget(self.chat_idle_label)
        self.chat_messages_layout.addStretch(1)

        self.chat_scroll.setWidget(chat_inner)
        lay.addWidget(self.chat_scroll, 1)

        input_row = QHBoxLayout()
        input_row.setContentsMargins(20, 10, 20, 16)
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText(
            "e.g. why is Ds along z an order of magnitude lower in MFI?")
        self.chat_input.returnPressed.connect(self.on_send_chat)
        input_row.addWidget(self.chat_input, 1)
        self.btn_send_chat = QPushButton("Send")
        self.btn_send_chat.setObjectName("PrimaryButton")
        self.btn_send_chat.clicked.connect(self.on_send_chat)
        input_row.addWidget(self.btn_send_chat)
        input_wrap = QWidget()
        input_wrap.setObjectName("ChatInputBar")
        input_wrap.setLayout(input_row)
        lay.addWidget(input_wrap)

        self.chat_worker = None
        self.chat_messages = []  # OpenAI-format history: [{"role": ..., "content": ...}, ...]
        self._thinking_timer = None
        self._thinking_row = None
        self._thinking_bubble = None
        self._stream_row = None
        self._stream_bubble = None
        self._stream_buffer = ""
        self._stream_render_timer = None

        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._apply_chat_center_margins)
        return w

    def _set_chat_idle(self, idle: bool):
        """Show the neutral 'system ready' status only when no turn has occurred yet
        -- this is a static UI state, not a message the assistant proactively sends."""
        self.chat_idle_label.setVisible(idle)

    # ------------------------------------------------------------------
    # Chat bubble rendering
    # ------------------------------------------------------------------
    def _format_chat_text(self, text: str) -> str:
        """
        Render assistant/user text as clean HTML: prose segments get the
        markdown-lite cleanup below; segments that are Markdown pipe
        tables (the one structured exception allowed by the system
        prompt, for side-by-side numeric comparisons) are rendered as
        proper Nature-style HTML tables instead of raw '|' characters.
        Called on the full accumulated text after every streamed delta
        (see ChatWorker / on_chat_delta), so a table only renders once
        its separator row has fully arrived; partial tables render as
        plain text in the meantime and resolve automatically.
        """
        blocks = self._split_table_blocks(text)
        return "".join(
            self._render_table_html(content) if kind == "table" else self._format_prose(content)
            for kind, content in blocks
        )

    _TABLE_SEP_CHARS = set("-|: ")

    def _looks_like_table_separator(self, line: str) -> bool:
        s = line.strip()
        return bool(s) and "-" in s and "|" in s and set(s) <= self._TABLE_SEP_CHARS

    def _split_table_row(self, line: str) -> list:
        s = line.strip()
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|"):
            s = s[:-1]
        return [c.strip() for c in s.split("|")]

    def _split_table_blocks(self, text: str) -> list:
        """Split text into ('text', str) and ('table', (header, rows)) segments."""
        lines = text.split("\n")
        blocks = []
        buf = []
        i, n = 0, len(lines)
        while i < n:
            line = lines[i]
            if ("|" in line and i + 1 < n and "|" in lines[i + 1]
                    and self._looks_like_table_separator(lines[i + 1])):
                if buf:
                    blocks.append(("text", "\n".join(buf)))
                    buf = []
                header = self._split_table_row(line)
                i += 2
                rows = []
                while i < n and lines[i].strip() and "|" in lines[i]:
                    rows.append(self._split_table_row(lines[i]))
                    i += 1
                blocks.append(("table", (header, rows)))
            else:
                buf.append(line)
                i += 1
        if buf:
            blocks.append(("text", "\n".join(buf)))
        return blocks

    def _render_table_html(self, content) -> str:
        """
        Natural (auto) table layout -- Qt's rich-text engine aligns
        columns reliably under auto layout, the same way a plain HTML
        table does. An earlier fix forced table-layout:fixed with
        percentage columns to keep wide tables inside the bubble, but
        that desynced column boundaries and row heights whenever a
        header wrapped onto two lines while its data cells stayed on
        one, producing visibly misaligned/overlapping rows. The bubble
        no longer tries to compress the table to fit -- see
        _make_table_scroll_widget, which lets the table take its
        natural width and scrolls horizontally instead.

        All cells -- header and data alike, numeric or not -- are
        left-aligned uniformly. An earlier version right-aligned
        numeric columns (a common convention for magnitude comparison),
        but with several numeric columns of differing widths next to
        text columns, the left/right mix itself read as inconsistent;
        one alignment for the whole table is visually calmer.
        """
        import html
        header, rows = content

        def cell(text, is_header):
            esc = html.escape(text)
            style = ("padding:6px 16px 6px 8px; border-bottom:1px solid #EFF1F3; "
                     "white-space:nowrap; text-align:left;")
            if is_header:
                style += "font-weight:600; color:#12151A; border-bottom:1px solid #12151A;"
            return f'<td style="{style}">{esc}</td>'

        thead = "<tr>" + "".join(cell(h, True) for h in header) + "</tr>"
        tbody = "".join(
            "<tr>" + "".join(cell(c, False) for c in row) + "</tr>"
            for row in rows
        )
        return (f'<table style="border-collapse:collapse; margin:2px 0; '
                f'font-size:11px;">{thead}{tbody}</table>')

    def _format_prose(self, text: str) -> str:
        """
        Markdown-lite -> HTML. The system prompt instructs the model to
        never emit raw markdown outside tables and the single controlled
        bold-emphasis exception (see SYSTEM_PROMPT), so this is mainly a
        defensive fallback for stray symbols, plus rendering of two
        deliberate structures: a line opening with "Structure:", "Energy:",
        or "Mechanism:" gets that label bolded as a section spine, and text
        the model wraps in **...** is treated as a key conclusion and
        rendered as a bold, light-grey "card" span (Nature/AI4Science
        low-saturation emphasis) rather than plain markdown-bold or a
        colored highlight.
        """
        import re, html

        raw_lines = text.split("\n")
        cleaned_lines = []
        for line in raw_lines:
            stripped = line.strip()
            # Convert leading bullet/dash list markers into plain prose
            stripped = re.sub(r"^[\-\*\u2022]\s+", "", stripped)
            # Strip markdown heading hashes ("## Title" -> "Title")
            stripped = re.sub(r"^#{1,6}\s*", "", stripped)
            cleaned_lines.append(stripped)
        text = "\n".join(cleaned_lines)

        escaped = html.escape(text)
        # Descriptor symbols -> subscripted rich text, so the reply reads with
        # the same notation as the tables, figures and the manuscript.
        escaped = _SYMBOL_RE.sub(lambda m: _SYMBOL_HTML[m.group(0)], escaped)
        # section-label structure: bold the "Structure:" / "Energy:" /
        # "Mechanism:" label itself when it opens a line, giving the
        # three-part analysis a clear visual spine without a markdown heading
        escaped = re.sub(
            r"(?im)^(Structure|Energy|Mechanism):",
            r"<b>\1:</b>", escaped)
        # key-conclusion exception: **text** -> bold + light-grey card span
        escaped = re.sub(
            r"\*\*(.+?)\*\*",
            r'<span style="background:#EDF3F9; padding:1px 6px; border-radius:2px; '
            r'font-weight:600; color:#12151A;">\1</span>', escaped)
        escaped = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", escaped)
        # any remaining stray asterisks (unpaired) are dropped rather than shown literally
        escaped = escaped.replace("*", "")
        # inline code spans
        escaped = re.sub(
            r"`([^`]+)`",
            r'<span style="background:#F1F0FA; padding:1px 4px; border-radius:3px;">\1</span>',
            escaped)
        escaped = escaped.replace("\n", "<br>")
        return escaped

    THINKING_STAGES = [
        "Understanding question",
        "Analyzing structural factors",
        "Evaluating energetic factors",
        "Synthesizing conclusion",
    ]

    def _chat_content_width(self) -> int:
        """
        The effective reading-column width: capped at 1100px, but shrinks
        on narrower windows rather than overflowing. This is the
        reference width bubble percentages are computed against, and the
        basis for the side margins that center the column
        (_apply_chat_center_margins).
        """
        viewport_w = self.chat_scroll.viewport().width()
        if viewport_w <= 0:
            viewport_w = self.width() or 1200
        return max(480, min(1100, viewport_w - 40))

    def _apply_chat_center_margins(self):
        """Center the reading column by setting equal side margins on the message layout."""
        viewport_w = self.chat_scroll.viewport().width()
        if viewport_w <= 0:
            return
        content_w = self._chat_content_width()
        margin = max(20, (viewport_w - content_w) // 2)
        self.chat_messages_layout.setContentsMargins(margin, 16, margin, 16)

    def _bubble_max_width(self, is_user: bool, has_table: bool = False) -> int:
        """
        Bubble width as a fraction of the centered reading column's width
        (see _chat_content_width), not the raw window/viewport width, and
        not a fixed pixel value -- so long scientific sentences and tables
        aren't force-wrapped onto many short lines. User messages read at
        ~62% of the column width (kept narrower than the reply, since a
        query is typically a single short-to-medium sentence); the assistant's
        replies are relaxed to ~80% (85% when the reply contains a table),
        giving long structure/energy/mechanism paragraphs room to unfold
        as a single continuous block instead of fragmenting into many
        short lines.
        """
        content_w = self._chat_content_width()
        if is_user:
            fraction = 0.62
        else:
            fraction = 0.85 if has_table else 0.80
        return max(280, int(content_w * fraction))

    def _size_bubble(self, bubble: QLabel, formatted_html: str, max_w: int, min_w: int = 90):
        """
        Explicitly size a bubble to fit its content, up to max_w.

        A word-wrapping QLabel inside a QHBoxLayout that also contains a
        stretch item does not reliably expand to its maximumWidth in Qt --
        its sizeHint() for wrapped rich text tends to collapse toward a
        much narrower "wrap at every opportunity" width, with the stretch
        item happily absorbing the rest. That produced fragmented,
        heavily-wrapped bubbles even for short messages. The fix is to
        measure the text's natural (unconstrained) width with a QTextDocument
        and set an explicit fixed width: the smaller of that natural width
        and max_w, so short text stays compact and long text wraps exactly
        at the intended column fraction rather than at some narrower,
        Qt-chosen width. Tables never reach this function -- see
        _make_table_scroll_widget.
        """
        from PySide6.QtGui import QTextDocument
        doc = QTextDocument()
        doc.setDefaultFont(bubble.font())
        doc.setHtml(formatted_html)
        doc.setTextWidth(-1)
        natural_w = doc.idealWidth()
        padding = 20  # measurement slack; layout margins are applied at the container level
        target_w = min(int(natural_w) + padding, max_w)
        bubble.setFixedWidth(max(target_w, min_w))

    def _make_prose_label(self, html_text: str, max_w: int) -> QLabel:
        lbl = QLabel()
        lbl.setTextFormat(Qt.RichText)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lbl.setStyleSheet("background: transparent;")
        lbl.setText(html_text)
        self._size_bubble(lbl, html_text, max_w)
        return lbl

    def _make_table_scroll_widget(self, content, max_w: int) -> QScrollArea:
        """
        Render a table at its natural (auto-layout) width, which keeps
        columns reliably aligned, inside a horizontally-scrollable,
        vertically-fixed container capped at max_w. Wide tables scroll
        instead of being compressed into misaligned or overflowing cells.
        """
        from PySide6.QtGui import QTextDocument

        table_html = self._render_table_html(content)
        inner = QLabel()
        inner.setTextFormat(Qt.RichText)
        inner.setWordWrap(False)
        inner.setTextInteractionFlags(Qt.TextSelectableByMouse)
        inner.setStyleSheet("background: transparent;")
        inner.setText(table_html)

        # Measure the rendered table directly via QTextDocument rather than
        # QLabel's own adjustSize()/sizeHint() chain, which can slightly
        # under-measure a <table>'s true width -- that under-measurement
        # permanently clips the last column's content (it's not a scroll
        # position issue: the widget itself is sized too narrow to hold
        # it, no matter how far you scroll). A generous safety buffer is
        # added on top for the same reason.
        doc = QTextDocument()
        doc.setDefaultFont(inner.font())
        doc.setHtml(table_html)
        doc.setTextWidth(-1)
        natural_w = int(doc.idealWidth()) + 28
        natural_h = int(doc.size().height()) + 12
        inner.setFixedSize(natural_w, natural_h)

        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(False)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)
        # Explicit local scrollbar styling, redundant with the global QSS
        # rule for QScrollBar:horizontal -- Qt's style engine can render
        # an orientation with no styling of its own as zero-height/
        # invisible once a stylesheet touches that widget type at all, so
        # this is set directly on the instance as a second guarantee that
        # the horizontal scrollbar stays visible and grabbable.
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; } "
            "QScrollBar:horizontal { height: 10px; background: transparent; } "
            "QScrollBar::handle:horizontal { background: #C9CBCE; border-radius: 5px; min-width: 20px; } "
            "QScrollBar::handle:horizontal:hover { background: #A6A9AD; } "
            "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal "
            "{ width: 0px; border: none; background: none; } "
            "QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; }"
        )
        scroll.setMaximumWidth(max_w)
        scroll.setFixedHeight(natural_h + 14)
        return scroll

    def _populate_bubble(self, container: QWidget, text: str, is_user: bool) -> bool:
        """
        (Re)build a bubble container's children from raw text: each
        Markdown pipe-table block becomes a horizontally-scrollable table
        widget, everything else becomes a wrapped prose QLabel. Used both
        for initial bubble construction and to re-render on window resize
        or as streamed text grows. Returns whether the content includes a
        table (so the caller can size the container's own max width using
        the wider table fraction when relevant).
        """
        layout = container.layout()
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        blocks = self._split_table_blocks(text) if text else []
        has_table = any(kind == "table" for kind, _ in blocks)
        prose_max_w = self._bubble_max_width(is_user, False)
        table_max_w = self._bubble_max_width(is_user, True)

        added = False
        for kind, content in blocks:
            if kind == "table":
                layout.addWidget(self._make_table_scroll_widget(content, table_max_w))
                added = True
            else:
                html_text = self._format_prose(content)
                if not html_text.strip():
                    continue
                layout.addWidget(self._make_prose_label(html_text, prose_max_w))
                added = True
        if not added:
            layout.addWidget(self._make_prose_label("", prose_max_w))

        container.setProperty("chat_text", text)
        return has_table

    def _make_bubble(self, text: str, is_user: bool):
        """Build a bubble row + composite container pair without inserting it into the layout."""
        row = QWidget()
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)

        container = QWidget()
        container.setObjectName("UserBubble" if is_user else "AssistantBubble")
        vlay = QVBoxLayout(container)
        vlay.setContentsMargins(14, 10, 14, 10)
        vlay.setSpacing(8)

        has_table = self._populate_bubble(container, text, is_user)
        container.setMaximumWidth(self._bubble_max_width(is_user, has_table))

        # No drop shadow: the assistant card is distinguished by an accent
        # spine on its left edge (see QSS #AssistantBubble), which reads as a
        # margin rule in a manuscript rather than a floating chat pill.

        if is_user:
            row_lay.addStretch(1)
            row_lay.addWidget(container)
        else:
            row_lay.addWidget(container)
            row_lay.addStretch(1)
        return row, container

    def _insert_row(self, row: QWidget):
        # insert before the trailing stretch
        self.chat_messages_layout.insertWidget(self.chat_messages_layout.count() - 1, row)
        self._scroll_chat_to_bottom()

    def _reflow_chat_bubble_widths(self):
        """Re-fit existing bubbles to their content after the window is resized."""
        if not hasattr(self, "chat_messages_layout"):
            return
        for i in range(self.chat_messages_layout.count()):
            item = self.chat_messages_layout.itemAt(i)
            row = item.widget()
            if row is None or row is self.chat_idle_label:
                continue
            for container in row.findChildren(QWidget):
                name = container.objectName()
                if name not in ("UserBubble", "AssistantBubble", "ThinkingBubble"):
                    continue
                text = container.property("chat_text")
                if text is None:
                    continue
                is_user = name == "UserBubble"
                has_table = self._populate_bubble(container, text, is_user)
                container.setMaximumWidth(self._bubble_max_width(is_user, has_table))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_chat_center_margins()
        self._reflow_chat_bubble_widths()

    def _scroll_chat_to_bottom(self):
        from PySide6.QtCore import QTimer

        def _do_scroll():
            bar = self.chat_scroll.verticalScrollBar()
            bar.setValue(bar.maximum())
        QTimer.singleShot(0, _do_scroll)

    def _add_chat_bubble(self, text: str, is_user: bool):
        row, _bubble = self._make_bubble(text, is_user)
        self._insert_row(row)

    # ------------------------------------------------------------------
    # Staged "thinking" indicator, shown between sending a question and
    # the first streamed token arriving -- gives a lightweight sense of
    # progression (question understood -> structure -> energetics ->
    # synthesis) without implying four separate model calls; it is purely
    # a client-side loading affordance ahead of the real streamed answer.
    # ------------------------------------------------------------------
    def _start_thinking_indicator(self):
        from PySide6.QtCore import QTimer
        self._thinking_stage = 0
        row, bubble = self._make_bubble(f"{self.THINKING_STAGES[0]}...", is_user=False)
        bubble.setObjectName("ThinkingBubble")
        self._thinking_row = row
        self._thinking_bubble = bubble
        self._insert_row(row)

        self._thinking_timer = QTimer(self)
        self._thinking_timer.setInterval(1100)
        self._thinking_timer.timeout.connect(self._advance_thinking_stage)
        self._thinking_timer.start()

    def _advance_thinking_stage(self):
        if self._thinking_bubble is None:
            return
        self._thinking_stage = (self._thinking_stage + 1) % len(self.THINKING_STAGES)
        text = f"{self.THINKING_STAGES[self._thinking_stage]}..."
        self._populate_bubble(self._thinking_bubble, text, is_user=False)

    def _stop_thinking_indicator(self):
        if getattr(self, "_thinking_timer", None) is not None:
            self._thinking_timer.stop()
            self._thinking_timer = None
        if getattr(self, "_thinking_row", None) is not None:
            self._thinking_row.deleteLater()
            self._thinking_row = None
        self._thinking_bubble = None

    def on_clear_chat(self):
        self.chat_messages = []
        # Remove all message rows, but keep the idle_label widget itself (just
        # re-shown via _set_chat_idle) and the trailing stretch item.
        i = 0
        while i < self.chat_messages_layout.count():
            item = self.chat_messages_layout.itemAt(i)
            widget = item.widget()
            if widget is None or widget is self.chat_idle_label:
                i += 1
                continue
            self.chat_messages_layout.takeAt(i)
            widget.deleteLater()
        self._set_chat_idle(True)

    def on_send_chat(self):
        question = self.chat_input.text().strip()
        if not question:
            return
        if self.last_result is None:
            QMessageBox.information(self, "No session data",
                                     "Run the pipeline first \u2014 the assistant answers only from "
                                     "the current session\u2019s results.")
            return
        api_key = self.chat_api_key.text().strip()
        model = self.chat_model_combo.currentText()

        self.chat_input.clear()
        self._set_chat_idle(False)
        self._add_chat_bubble(question, is_user=True)
        self.chat_messages.append({"role": "user", "content": question})

        context_summary = llm_chat.build_context_summary(
            self.last_result.prediction_display,
            self.last_result.shap_original_df,
            self.last_result.feature_table,
        )

        self.btn_send_chat.setEnabled(False)
        self._stream_buffer = ""
        self._stream_row = None
        self._stream_bubble = None
        self._start_thinking_indicator()

        self.chat_worker = ChatWorker(api_key, model, list(self.chat_messages), context_summary)
        self.chat_worker.delta.connect(self.on_chat_delta)
        self.chat_worker.reply_ok.connect(self.on_chat_reply_ok)
        self.chat_worker.reply_error.connect(self.on_chat_reply_error)
        self.chat_worker.start()

    def on_chat_delta(self, fragment: str):
        if self._stream_bubble is None:
            # first token arrived: swap the thinking indicator for the real,
            # progressively-filling reply bubble
            self._stop_thinking_indicator()
            self._stream_row, self._stream_bubble = self._make_bubble("", is_user=False)
            self._insert_row(self._stream_row)
        self._stream_buffer += fragment
        self._schedule_stream_render()

    def _schedule_stream_render(self):
        """
        Coalesce rapid deltas into a UI rebuild at most every ~120ms.
        Rebuilding the composite bubble (_populate_bubble) on every single
        network chunk -- DeepSeek's stream can emit several fragments per
        second -- visibly stutters, especially once a table's QScrollArea
        is involved. Text keeps accumulating in _stream_buffer immediately;
        only the (comparatively expensive) widget rebuild is throttled.
        """
        from PySide6.QtCore import QTimer
        if self._stream_render_timer is None:
            self._stream_render_timer = QTimer(self)
            self._stream_render_timer.setSingleShot(True)
            self._stream_render_timer.timeout.connect(self._render_stream_buffer)
        if not self._stream_render_timer.isActive():
            self._stream_render_timer.start(120)

    def _render_stream_buffer(self):
        if self._stream_bubble is None:
            return
        has_table = self._populate_bubble(self._stream_bubble, self._stream_buffer, is_user=False)
        self._stream_bubble.setMaximumWidth(self._bubble_max_width(False, has_table))
        self._scroll_chat_to_bottom()

    def on_chat_reply_ok(self, full_text: str):
        self.btn_send_chat.setEnabled(True)
        self._stop_thinking_indicator()
        if self._stream_render_timer is not None:
            self._stream_render_timer.stop()
        if self._stream_bubble is None:
            # no deltas were streamed (e.g. an empty/instant response) -- show it now
            self._add_chat_bubble(full_text, is_user=False)
        else:
            # final, authoritative render -- guarantees the displayed content
            # exactly matches the complete reply even if a throttled render
            # was still pending
            has_table = self._populate_bubble(self._stream_bubble, full_text, is_user=False)
            self._stream_bubble.setMaximumWidth(self._bubble_max_width(False, has_table))
        self.chat_messages.append({"role": "assistant", "content": full_text})
        self._stream_bubble = None

    def on_chat_reply_error(self, msg: str):
        self.btn_send_chat.setEnabled(True)
        self._stop_thinking_indicator()
        if self._stream_render_timer is not None:
            self._stream_render_timer.stop()
        self._add_chat_bubble(f"Request failed: {msg}", is_user=False)
        self._stream_bubble = None

    def on_generate_waterfall(self):
        if self.last_result is None or self.last_result.shap_original_df is None:
            QMessageBox.information(self, "No results yet",
                                     "Run the pipeline first \u2014 there is no attribution to plot.")
            return
        sample_name = self.shap_sample_combo.currentText()
        if not sample_name:
            return
        from core import shap_original
        shap_dir = os.path.join(self.last_result.work_dir, "03_shap")
        try:
            path = shap_original.waterfall_for_sample(
                self.last_result.shap_original_df, sample_name,
                feature_table=self.last_result.feature_table,
                output_dir=shap_dir,
            )
            self._show_waterfall(path, sample_name)
        except Exception as exc:
            QMessageBox.warning(self, "Plot failed", str(exc))

    # ------------------------------------------------------------------
    def append_log(self, text):
        self.log_console.appendPlainText(text)

    def set_stage(self, name, frac):
        self.stage_label.setText(name)
        self.stage_counter.setText(f"{int(round(frac * 100)):02d} %")
        self.progress.setValue(int(frac * 100))

    # ------------------------------------------------------------------
    def on_run(self):
        cif_dir = self.f_cif_dir.text()
        work_dir = self.f_work_dir.text()

        missing = [n for n, v in [("CIF Directory", cif_dir), ("Working Directory", work_dir)] if not v]
        if missing:
            QMessageBox.warning(self, "Incomplete input", f"Still required: {', '.join(missing)}")
            return

        for label, path in [("External Excel", DEFAULT_EXTERNAL_EXCEL),
                             ("ANN Model", DEFAULT_MODEL_PATH),
                             ("Scaler", DEFAULT_SCALER_PATH)]:
            if not os.path.exists(path):
                QMessageBox.critical(self, "Missing resource",
                                      f"{label} not found at the expected location:\n{path}\n\n"
                                      f"Place the file there before running.")
                return

        cfg = PipelineConfig(
            grid_spacing=self.sp_spacing.value(),
            grid_cutoff=self.sp_cutoff.value(),
            probe_type=self.cb_probe.currentText(),
            min_supercell_length=self.sp_supercell.value(),
            energy_threshold=self.sp_ethresh.value(),
            save_html=self.chk_save_html.isChecked(),
        )

        self.btn_run.setEnabled(False)
        self.log_console.clear()
        self.tabs.setCurrentWidget(self.tab_log)

        self.worker = PipelineWorker(dict(
            cif_dir=cif_dir, work_dir=work_dir, cfg=cfg,
            skip_symmetry=self.chk_skip_symmetry.isChecked(),
        ))
        self.worker.log_line.connect(self.append_log)
        self.worker.stage_changed.connect(self.set_stage)
        self.worker.finished_ok.connect(self.on_finished_ok)
        self.worker.finished_error.connect(self.on_finished_error)
        self.worker.start()

    def on_finished_error(self, msg):
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "Run failed", msg)

    def on_finished_ok(self, result):
        self.btn_run.setEnabled(True)
        self.last_result = result
        if result.feature_table is not None:
            self._fill_table(self.tab_features.table, result.feature_table)
        if result.prediction_display is not None:
            self._fill_table(self.tab_predictions.table, result.prediction_display)
        self._fill_html_tab(result.html_paths)

        # Which SHAP reference distribution was used matters when reading the
        # plot, so it is stated in the run log; it is no longer printed under
        # the figure.
        self.append_log(
            "  SHAP reference distribution: training feature set."
            if result.shap_background_source else
            "  SHAP reference distribution: current batch only "
            "(assets/shap_background.xlsx absent) -- descriptors without variance "
            "across this batch will register a zero contribution."
        )

        self.shap_sample_combo.clear()
        if result.shap_sample_names:
            self.shap_sample_combo.addItems(result.shap_sample_names)
        if result.default_waterfall_path and os.path.exists(result.default_waterfall_path):
            self._show_waterfall(result.default_waterfall_path,
                                  self.shap_sample_combo.currentText())

        QMessageBox.information(self, "Run complete",
                                 f"Pipeline finished.\n\nConsolidated results written to:\n{result.excel_path}")

    # ------------------------------------------------------------------
    def _show_waterfall(self, path: str, sample_name: str = ""):
        """
        Show the rendered plot on the plate. No caption is drawn beneath it --
        the plot carries its own axis labels, and the SHAP background actually
        used is reported in the run log rather than under the figure.
        """
        pix = QPixmap(path)
        if not pix.isNull():
            self.waterfall_label.setPixmap(pix.scaledToWidth(820, Qt.SmoothTransformation))
            self.waterfall_label.setText("")

    @staticmethod
    def _format_cell(val, key: str | None) -> str:
        """
        Typeset a cell the way a journal table would: descriptor values to four
        significant figures, diffusivities in scientific notation, everything
        else verbatim.
        """
        import math
        if val is None:
            return ""
        if isinstance(val, bool):
            return "yes" if val else "no"
        if isinstance(val, (int,)) and not isinstance(val, bool):
            return str(val)
        if isinstance(val, float):
            if math.isnan(val):
                return "\u2014"
            if val != 0 and (abs(val) < 1e-3 or abs(val) >= 1e5):
                return f"{val:.3e}"
            return f"{val:.4g}"
        return str(val)

    def _fill_table(self, table: QTableWidget, df):
        """
        Render a results frame with publication symbols in the header. The
        underlying frame is untouched -- only the displayed header text is
        mapped through core.descriptor_names.

        Headers show the bare symbol, without a unit: a table of thirteen
        descriptors becomes unreadable when every column carries a
        parenthesised unit, and the unit is one hover away in the tooltip and
        one click away in the nomenclature panel. Descriptor headers are set
        in italic, as variable names are in a manuscript; identifier columns
        (Zeolite, Direction ...) stay roman. Everything is left-aligned.
        """
        from PySide6.QtGui import QFont

        table.clear()
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))

        headers, keys = [], []
        for col in df.columns:
            key = col if col in dn.DESCRIPTORS else None
            keys.append(key)
            headers.append(dn.symbol(col) if key else str(col))
        table.setHorizontalHeaderLabels(headers)

        for j, key in enumerate(keys):
            head_item = table.horizontalHeaderItem(j)
            if head_item is None:
                continue
            # Weight and italics are set here rather than in the stylesheet:
            # any font property in QHeaderView::section QSS would override
            # these per-item settings wholesale (see the note in style.py).
            font = QFont(table.font())
            font.setBold(True)
            font.setItalic(key is not None)
            head_item.setFont(font)
            if key:
                head_item.setToolTip(dn.tooltip(key))

        left_align = int(Qt.AlignLeft | Qt.AlignVCenter)
        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                val = df.iloc[i, j]
                item = QTableWidgetItem(self._format_cell(val, keys[j]))
                item.setTextAlignment(left_align)
                table.setItem(i, j, item)
        table.resizeColumnsToContents()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
