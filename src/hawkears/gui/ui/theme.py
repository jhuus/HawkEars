"""Application-wide visual styling."""

STYLESHEET = """
QWidget {
    color: #17231d;
    font-family: "Inter", "Segoe UI", sans-serif;
    font-size: 13px;
}
QMainWindow, #appRoot, #pageSurface { background: #f5f7f4; }
#sidebar {
    background: #163f35;
    border: none;
}
#brand { color: white; font-size: 22px; font-weight: 700; }
#brandSubtle { color: #a9c7bd; font-size: 11px; }
#projectChip {
    background: #205246;
    border: 1px solid #356b5e;
    border-radius: 8px;
    color: white;
    padding: 10px;
    text-align: left;
}
#projectChip:hover { background: #285f52; border-color: #4a7c70; }
QPushButton[nav="true"] {
    background: transparent;
    border: none;
    border-radius: 7px;
    color: #cee0da;
    padding: 10px 12px;
    text-align: left;
}
QPushButton[nav="true"]:hover { background: #205246; color: white; }
QPushButton[nav="true"]:checked { background: #e7f1ed; color: #123b32; font-weight: 600; }
QPushButton[nav="true"]:disabled { color: #668d82; }
#pageTitle { color: #132d26; font-size: 25px; font-weight: 700; }
#pageSubtitle { color: #68766f; font-size: 13px; }
#sectionTitle { color: #1b382f; font-size: 16px; font-weight: 650; }
#metricValue { color: #153f35; font-size: 25px; font-weight: 700; }
#muted { color: #6d7973; }
QFrame[card="true"] {
    background: white;
    border: 1px solid #dde4df;
    border-radius: 10px;
}
QPushButton {
    background: white;
    border: 1px solid #bdc9c3;
    border-radius: 6px;
    padding: 7px 13px;
}
QPushButton:hover { border-color: #377b68; background: #f7faf8; }
QPushButton:disabled {
    background: #eef1ef;
    border-color: #d5ddd8;
    color: #98a39d;
}
QPushButton[primary="true"] {
    background: #24715e;
    border-color: #24715e;
    color: white;
    font-weight: 600;
}
QPushButton[primary="true"]:hover { background: #1d604f; }
QPushButton[primary="true"]:disabled {
    background: #aebbb6;
    border-color: #aebbb6;
    color: #edf1ef;
}
QPushButton[verdict="true"]:checked {
    background: #dceee7;
    border-color: #24715e;
    color: #153f35;
    font-weight: 600;
}
QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit {
    background: white;
    border: 1px solid #cbd4cf;
    border-radius: 6px;
    padding: 6px 8px;
}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #24715e;
}
QTableWidget {
    background: white;
    border: 1px solid #dde4df;
    border-radius: 8px;
    gridline-color: #edf0ee;
    selection-background-color: #dceee7;
    selection-color: #153f35;
}
QHeaderView::section {
    background: #f0f4f1;
    border: none;
    border-bottom: 1px solid #d9e1dc;
    color: #52615a;
    font-weight: 600;
    padding: 8px;
}
QProgressBar {
    background: #e3e9e5;
    border: none;
    border-radius: 5px;
    height: 10px;
    text-align: center;
}
QProgressBar::chunk { background: #4a9a81; border-radius: 5px; }
QSplitter::handle { background: #e0e5e2; width: 1px; }
"""
