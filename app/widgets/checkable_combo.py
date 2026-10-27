from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QStyle,
    QApplication,
)
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPointF, QEvent
from PyQt6.QtGui import (
    QStandardItemModel,
    QStandardItem,
    QPainter,
    QPen,
    QColor,
    QPalette,
    QFont,
)


SELECT_ALL_TEXT = "全选"


def _as_check_state(value):
    """把模型返回的 CheckState（可能是枚举，也可能是 int）统一成枚举。"""
    if value is None:
        return None
    try:
        return Qt.CheckState(value)
    except (TypeError, ValueError):
        return None


class _CheckColorDelegate(QStyledItemDelegate):
    """自绘勾选框：选中时填充强调色并描粗对号，未选中时只画空心描边。"""

    _BOX = 16
    _MARGIN = 6
    _GAP = 9

    _ACCENT = QColor("#22d3ee")
    _ACCENT_HOVER = QColor("#67e8f9")
    _BORDER = QColor("#5b6b82")
    _MARK = QColor("#0a0f1a")

    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        check = _as_check_state(index.data(Qt.ItemDataRole.CheckStateRole))
        has_check = check is not None
        row_rect = QRect(opt.rect)
        hovered = bool(opt.state & QStyle.StateFlag.State_MouseOver)
        selected = bool(opt.state & QStyle.StateFlag.State_Selected)

        if has_check:
            # 去掉默认指示器，改由下方自绘；并为文字腾出勾选框宽度
            opt.features &= ~QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator
            indent = self._MARGIN + self._BOX + self._GAP
            if opt.direction == Qt.LayoutDirection.RightToLeft:
                opt.rect.setRight(opt.rect.right() - indent)
            else:
                opt.rect.setLeft(opt.rect.left() + indent)
            # 已勾选项加粗，让"哪些被选中"在一行里就能看出来
            if check == Qt.CheckState.Checked:
                font = QFont(opt.font)
                font.setWeight(QFont.Weight.DemiBold)
                opt.font = font

        widget = opt.widget
        style = widget.style() if widget is not None else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)

        if not has_check:
            return

        box = self._indicator_rect(row_rect, opt.direction)
        checked = check == Qt.CheckState.Checked
        partial = check == Qt.CheckState.PartiallyChecked

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if checked or partial:
            # 选中的勾选框：整块填充强调色，形成明显色块
            fill = self._ACCENT_HOVER if hovered else self._ACCENT
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill)
            painter.drawRoundedRect(box, 3.5, 3.5)

            # 对号使用深色描边，在亮色填充上对比最高
            painter.setPen(
                QPen(
                    self._MARK,
                    2.2,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
            )
            x, y, w, h = box.x(), box.y(), box.width(), box.height()
            if checked:
                p1 = QPointF(x + w * 0.24, y + h * 0.52)
                p2 = QPointF(x + w * 0.42, y + h * 0.70)
                p3 = QPointF(x + w * 0.76, y + h * 0.30)
                painter.drawLine(p1, p2)
                painter.drawLine(p2, p3)
            else:
                painter.drawLine(
                    QPointF(x + w * 0.25, y + h * 0.5),
                    QPointF(x + w * 0.75, y + h * 0.5),
                )
        else:
            # 未选中的勾选框：空心描边；悬停时边框高亮，给出可点击反馈
            if selected:
                border = opt.palette.color(QPalette.ColorRole.HighlightedText)
                if not border.isValid() or border.alpha() == 0:
                    border = QColor("#ffffff")
            elif hovered:
                border = self._ACCENT
            else:
                border = self._BORDER
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(border, 1.5))
            painter.drawRoundedRect(box, 3.5, 3.5)

        painter.restore()

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        # 保证行高容得下 16px 勾选框，视觉上更透气（不依赖外部样式表是否加载）
        if index.data(Qt.ItemDataRole.CheckStateRole) is not None:
            size.setHeight(max(size.height(), 30))
            size.setWidth(
                size.width() + self._MARGIN + self._BOX + self._GAP
            )
        return size

    def editorEvent(self, event, model, option, index):
        # 整行都可点击切换，而不只是点在勾选框上
        if index.data(Qt.ItemDataRole.CheckStateRole) is not None:
            if (
                event.type() == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.LeftButton
                and option.rect.contains(event.position().toPoint())
            ):
                self._toggle(model, index)
                return True
            if event.type() == QEvent.Type.KeyPress and event.key() in (
                Qt.Key.Key_Space,
                Qt.Key.Key_Select,
            ):
                self._toggle(model, index)
                return True
        return super().editorEvent(event, model, option, index)

    @staticmethod
    def _toggle(model, index):
        checked = (
            _as_check_state(index.data(Qt.ItemDataRole.CheckStateRole))
            == Qt.CheckState.Checked
        )
        model.setData(
            index,
            Qt.CheckState.Unchecked if checked else Qt.CheckState.Checked,
            Qt.ItemDataRole.CheckStateRole,
        )

    @classmethod
    def _indicator_rect(cls, row_rect: QRect, direction) -> QRect:
        size = cls._BOX
        margin = cls._MARGIN
        y = row_rect.y() + (row_rect.height() - size) // 2
        if direction == Qt.LayoutDirection.RightToLeft:
            x = row_rect.right() - margin - size
        else:
            x = row_rect.x() + margin
        return QRect(int(x), int(y), size, size)


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
        self.setItemDelegate(_CheckColorDelegate(self))
        # 开启悬停反馈：鼠标移到行上时勾选框边框/填充会高亮
        self.view().setMouseTracking(True)
        self.view().viewport().setMouseTracking(True)
        self.view().setCursor(Qt.CursorShape.PointingHandCursor)
        self.view().setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._select_all_row = -1
        self._updating = 0
        # 用 itemChanged 捕获所有勾选变化（包括点击复选框本体，而不只是 pressed），
        # 这样才能在「取消某一个勾选」时同步「全选」状态。
        self.model().itemChanged.connect(self._on_item_changed)

    def _on_item_changed(self, item):
        if self._updating > 0 or item.checkState() is None:
            return
        m = self.model()
        sa = m.item(self._select_all_row) if self._select_all_row >= 0 else None
        if item is sa:
            # 点击「全选」：把勾选状态同步到所有可勾选项
            state = item.checkState()
            if state == Qt.CheckState.PartiallyChecked:
                state = Qt.CheckState.Checked
            self._updating += 1
            try:
                for i in range(m.rowCount()):
                    it = m.item(i)
                    if it.checkState() is not None:
                        it.setCheckState(state)
            finally:
                self._updating -= 1
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
        # 只有全部勾选时「全选」才显示勾选，否则一律显示为未勾选
        # （取消任意一个，就取消「全选」的勾，而不是显示半选状态）
        target = (
            Qt.CheckState.Checked
            if (total > 0 and checked == total)
            else Qt.CheckState.Unchecked
        )
        if sa.checkState() != target:
            self._updating += 1
            try:
                sa.setCheckState(target)
            finally:
                self._updating -= 1

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

    def all_items(self):
        """返回所有可勾选的传感器 SN（不含「全选」与占位项）。"""
        m = self.model()
        out = []
        for i in range(m.rowCount()):
            if i == self._select_all_row:
                continue
            it = m.item(i)
            if it is None:
                continue
            if it.checkState() is None:
                continue
            out.append(it.text())
        return out

    def set_checked(self, text, checked):
        m = self.model()
        self._updating += 1
        try:
            for i in range(m.rowCount()):
                it = m.item(i)
                if it.text() == text and it.checkState() is not None:
                    it.setCheckState(
                        Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                    )
        finally:
            self._updating -= 1
        self._update_display()
