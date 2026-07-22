"""Application-wide visual styling."""

STYLESHEET = """
QWidget {
    color: #2f211f;
    font-family: "Inter", "Segoe UI", sans-serif;
    font-size: 13px;
}
QMainWindow, #appRoot, #pageSurface { background: #faf6f1; }
#sidebar {
    background: #4f1612;
    border: none;
}
#brand { color: #f8f0e7; font-size: 22px; font-weight: 700; }
#brandSubtle { color: #fcbb73; font-size: 11px; }
#projectChip {
    background: #220e0d;
    border: 1px solid #79352d;
    border-radius: 8px;
    color: #f8f0e7;
    padding: 10px;
    text-align: left;
}
#projectChip:hover { background: #68241e; border-color: #fbb040; }
QPushButton[nav="true"] {
    background: transparent;
    border: none;
    border-radius: 7px;
    color: #f8f0e7;
    padding: 10px 12px;
    text-align: left;
}
QPushButton[nav="true"]:hover { background: #220e0d; color: white; }
QPushButton[nav="true"]:checked { background: #f8f0e7; color: #4f1612; font-weight: 600; }
QPushButton[nav="true"]:disabled { color: #9d7771; }
#pageTitle { color: #4f1612; font-size: 25px; font-weight: 700; }
#pageSubtitle { color: #756763; font-size: 13px; }
#sectionTitle { color: #4f1612; font-size: 16px; font-weight: 650; }
#metricValue { color: #4f1612; font-size: 25px; font-weight: 700; }
#muted { color: #786b67; }
QLabel:disabled { color: #a39792; }
QFrame[card="true"] {
    background: white;
    border: 1px solid #e5dcd4;
    border-radius: 10px;
}
QPushButton {
    background: white;
    border: 1px solid #cdbfb7;
    border-radius: 6px;
    padding: 7px 13px;
}
QPushButton:hover { border-color: #4f1612; background: #f8f0e7; }
QPushButton:disabled {
    background: #eee9e5;
    border-color: #ddd4ce;
    color: #9b918c;
}
QPushButton[primary="true"] {
    background: #4f1612;
    border-color: #4f1612;
    color: #f8f0e7;
    font-weight: 600;
}
QPushButton[primary="true"]:hover { background: #220e0d; border-color: #220e0d; }
QPushButton[primary="true"]:disabled {
    background: #b9aaa4;
    border-color: #b9aaa4;
    color: #eee7e2;
}
QPushButton[verdictValue="correct"]:checked {
    background: #e1efe7;
    border-color: #347a53;
    color: #204a34;
    font-weight: 600;
}
QPushButton[verdictValue="incorrect"]:checked {
    background: #f3dedb;
    border-color: #9b3028;
    color: #6d1d18;
    font-weight: 600;
}
QPushButton[verdictValue="uncertain"]:checked {
    background: #fff0cf;
    border-color: #fbb040;
    color: #4f1612;
    font-weight: 600;
}
QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit {
    background: white;
    border: 1px solid #d5c9c2;
    border-radius: 6px;
    padding: 6px 8px;
}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #4f1612;
}
QTableWidget {
    background: white;
    border: 1px solid #e5dcd4;
    border-radius: 8px;
    gridline-color: #eee8e3;
    selection-background-color: #f8e6c7;
    selection-color: #4f1612;
}
QHeaderView::section {
    background: #f8f0e7;
    border: none;
    border-bottom: 1px solid #e1d5cc;
    color: #62524e;
    font-weight: 600;
    padding: 8px;
}
QProgressBar {
    background: #eadfd7;
    border: none;
    border-radius: 5px;
    height: 10px;
    text-align: center;
}
QProgressBar::chunk { background: #fbb040; border-radius: 5px; }
QSplitter::handle { background: #e3d9d2; width: 1px; }
"""
