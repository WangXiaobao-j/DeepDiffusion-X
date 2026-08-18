"""Nomenclature dialog listing descriptor symbols, units, and definitions."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QScrollArea,
    QPushButton, QApplication,
)

from core import descriptor_names as dn


class NomenclatureDialog(QDialog):
    """Modeless reference sheet of descriptor symbols."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nomenclature - descriptor symbols")
        self.resize(760, 720)
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        head = QWidget()
        head_lay = QVBoxLayout(head)
        head_lay.setContentsMargins(30, 26, 30, 16)
        head_lay.setSpacing(4)
        title = QLabel("Nomenclature")
        title.setObjectName("GlossaryTitle")
        sub = QLabel(
            "Symbols used throughout the interface, figures and exported tables. "
            "Units are listed here and in the table header tooltips only \u2014 displayed "
            "labels carry the bare symbol. The internal column key beneath each entry is "
            "the frozen name stored in the model checkpoint and is never renamed."
        )
        sub.setObjectName("Caption")
        sub.setWordWrap(True)
        head_lay.addWidget(title)
        head_lay.addWidget(sub)
        outer.addWidget(head)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        body = QWidget()
        body.setObjectName("GlossaryBody")
        lay = QVBoxLayout(body)
        lay.setContentsMargins(30, 0, 30, 26)
        lay.setSpacing(0)

        for family, members in dn.glossary_rows():
            fam_lbl = QLabel(family.upper())
            fam_lbl.setObjectName("GlossaryFamily")
            lay.addWidget(fam_lbl)
            for d in members:
                lay.addWidget(self._entry(d))

        lay.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        foot = QWidget()
        foot_lay = QHBoxLayout(foot)
        foot_lay.setContentsMargins(30, 12, 30, 18)
        copy_btn = QPushButton("Copy as plain text")
        copy_btn.clicked.connect(self._copy_plain)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        foot_lay.addWidget(copy_btn)
        foot_lay.addStretch(1)
        foot_lay.addWidget(close_btn)
        outer.addWidget(foot)

    # ------------------------------------------------------------------
    @staticmethod
    def _entry(d: "dn.Descriptor") -> QWidget:
        row = QWidget()
        row.setObjectName("GlossaryRow")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 12, 0, 12)
        lay.setSpacing(18)

        sym = QLabel(d.html)
        sym.setTextFormat(Qt.RichText)
        sym.setObjectName("GlossarySymbol")
        sym.setFixedWidth(118)
        sym.setAlignment(Qt.AlignRight | Qt.AlignTop)
        lay.addWidget(sym)

        right = QVBoxLayout()
        right.setSpacing(3)
        name_row = QHBoxLayout()
        name_row.setSpacing(10)
        name = QLabel(d.name)
        name.setObjectName("GlossaryName")
        unit = QLabel(d.unit if d.unit else "dimensionless")
        unit.setObjectName("GlossaryUnit")
        name_row.addWidget(name)
        name_row.addWidget(unit)
        name_row.addStretch(1)
        right.addLayout(name_row)

        definition = QLabel(d.definition)
        definition.setObjectName("GlossaryDef")
        definition.setWordWrap(True)
        right.addWidget(definition)

        key = QLabel(f"internal key: {d.key}")
        key.setObjectName("GlossaryKey")
        right.addWidget(key)

        lay.addLayout(right, 1)
        return row

    def _copy_plain(self):
        QApplication.clipboard().setText(dn.nomenclature_block())
