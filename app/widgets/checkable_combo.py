from PyQt6.QtWidgets import QComboBox
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QStandardItemModel, QStandardItem


SELECT_ALL_TEXT = "全选"


class CheckableComboBox(QComboBox):
    """带勾选框的多选下拉框。

    下拉列表中每个传感器前面都有一个勾选框，可勾选多个；
    顶部提供「全选」选项；收起状态下，行编辑框显示所有已勾选项。
    """
    checked_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.lineEdit().setReadOnly(True)
        self.setModel(QStandardItemModel(self))
        self._select_all_row = -1
        self.view().pressed.connect(self._on_item_pressed)

    def _on_item_pressed(self, index):
        item = self.model().itemFromIndex(index)
        if item is None or item.checkState() is None:
            return
        m = self.model()
        if index.row() == self._select_all_row:
            new_state = (
                Qt.CheckState.Unchecked
                if item.checkState() == Qt.CheckState.Checked
                else Qt.CheckState.Checked
            )
            for i in range(m.rowCount()):
                it = m.item(i)
                if it.checkState() is not None:
                    it.setCheckState(new_state)
        else:
            item.setCheckState(
                Qt.CheckState.Unchecked
                if item.checkState() == Qt.CheckState.Checked
                else Qt.CheckState.Checked
            )
        self._update_display()
        self.checked_changed.emit()

    def hidePopup(self):
        # 勾选过程中保持下拉展开，点击外部才收起
        if self.view().isVisible() and self.view().underMouse():
            return
        super().hidePopup()

    def add_select_all_item(self):
        item = QStandardItem(SELECT_ALL_TEXT)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        item.setData(Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
        self.model().insertRow(0, item)
        self._select_all_row = 0
        self._update_display()

    def add_item(self, text, checked=False):
        item = QStandardItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        item.setData(
            Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked,
            Qt.ItemDataRole.CheckStateRole,
        )
        self.model().appendRow(item)
        self._update_display()

    def add_placeholder(self, text):
        item = QStandardItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.model().appendRow(item)
        self._update_display()

    def clear_items(self):
        self._select_all_row = -1
        self.model().clear()
        self._update_display()

    def _sync_select_all(self):
        if self._select_all_row < 0:
            return
        m = self.model()
        sa = m.item(self._select_all_row)
        if sa is None:
            return
        checked = 0
        total = 0
        for i in range(m.rowCount()):
            if i == self._select_all_row:
                continue
            it = m.item(i)
            if it.checkState() is None:
                continue
            total += 1
            if it.checkState() == Qt.CheckState.Checked:
                checked += 1
        if total == 0 or checked == 0:
            sa.setCheckState(Qt.CheckState.Unchecked)
        elif checked == total:
            sa.setCheckState(Qt.CheckState.Checked)
        else:
            sa.setCheckState(Qt.CheckState.PartiallyChecked)

    def _update_display(self):
        self._sync_select_all()
        self.lineEdit().setText(", ".join(self.checked_items()))

    def checked_items(self):
        m = self.model()
        out = []
        for i in range(m.rowCount()):
            if i == self._select_all_row:
                continue
            it = m.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                out.append(it.text())
        return out

    def set_checked(self, text, checked):
        m = self.model()
        for i in range(m.rowCount()):
            it = m.item(i)
            if it.text() == text and it.checkState() is not None:
                it.setCheckState(
                    Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                )
        self._update_display()
