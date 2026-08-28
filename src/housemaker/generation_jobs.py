# ### Imports ###
from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QCloseEvent, QGuiApplication, QScreen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


# ### Constants ###
JOB_STATUS_RUNNING = "running"
JOB_STATUS_CANCELLING = "cancelling"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_CANCELLED = "cancelled"
TERMINAL_JOB_STATUSES = frozenset(
    {JOB_STATUS_COMPLETED, JOB_STATUS_FAILED, JOB_STATUS_CANCELLED}
)
_PERCENT_PATTERN = re.compile(r"(?<!\d)(100|[1-9]?\d)\s*%")
_WINDOW_EDGE_OFFSET = 32
_WINDOW_DEFAULT_WIDTH = 460
_WINDOW_DEFAULT_HEIGHT = 520
_PROGRESS_UNSET = object()


# ### Job data ###
@dataclass(frozen=True)
class GenerationJob:
    """Immutable presentation state for one background generation task."""

    job_id: str
    name: str
    kind: str
    stage: str
    status: str = JOB_STATUS_RUNNING
    progress: int | None = None

    @property
    def is_finished(self) -> bool:
        return self.status in TERMINAL_JOB_STATUSES


# ### Job manager ###
class GenerationJobManager(QObject):
    """Own session job state and expose cancellation without owning workers."""

    job_added = Signal(object)
    job_updated = Signal(object)
    jobs_cleared = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._jobs: dict[str, GenerationJob] = {}
        self._cancel_callbacks: dict[str, Callable[[], bool]] = {}

    def jobs(self) -> tuple[GenerationJob, ...]:
        return tuple(self._jobs.values())

    def get_job(self, job_id: str) -> GenerationJob | None:
        return self._jobs.get(str(job_id))

    def create_job(
        self,
        *,
        kind: str,
        requested_name: str,
        default_name: str,
        stage: str,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> GenerationJob:
        name = str(requested_name).strip() or str(default_name).strip()
        if not name:
            name = "Generation job"
        job = GenerationJob(
            job_id=uuid.uuid4().hex,
            name=name,
            kind=str(kind).strip() or "Generation",
            stage=str(stage).strip() or "Queued",
            progress=_cap_running_progress(
                _extract_progress_percent(stage)
            ),
        )
        self._jobs[job.job_id] = job
        if cancel_callback is not None:
            self._cancel_callbacks[job.job_id] = cancel_callback
        self.job_added.emit(job)
        return job

    def set_cancel_callback(
        self,
        job_id: str,
        callback: Callable[[], bool] | None,
    ) -> None:
        normalized_job_id = str(job_id)
        job = self._jobs.get(normalized_job_id)
        if job is None or job.is_finished:
            self._cancel_callbacks.pop(normalized_job_id, None)
            return
        if callback is None:
            self._cancel_callbacks.pop(normalized_job_id, None)
            return
        self._cancel_callbacks[normalized_job_id] = callback

    def rename_job(
        self,
        job_id: str,
        name: str,
    ) -> GenerationJob | None:
        """Rename one existing row while preserving all other job state."""

        current = self._jobs.get(str(job_id))
        normalized_name = str(name).strip()
        if current is None or current.is_finished or not normalized_name:
            return current
        updated = replace(current, name=normalized_name)
        self._jobs[current.job_id] = updated
        self.job_updated.emit(updated)
        return updated

    def update_job(
        self,
        job_id: str,
        *,
        stage: str | None = None,
        progress: int | None | object = _PROGRESS_UNSET,
        status: str | None = None,
    ) -> GenerationJob | None:
        current = self._jobs.get(str(job_id))
        if current is None:
            return None
        if current.is_finished:
            return current
        if current.status == JOB_STATUS_CANCELLING and status is None:
            return current
        next_stage = current.stage if stage is None else str(stage).strip()
        next_status = current.status if status is None else str(status)
        next_progress = progress
        if next_progress is _PROGRESS_UNSET and stage is not None:
            extracted_progress = _extract_progress_percent(next_stage)
            next_progress = (
                current.progress
                if extracted_progress is None
                else extracted_progress
            )
        if next_progress is _PROGRESS_UNSET:
            next_progress = current.progress
        elif next_progress is not None:
            next_progress = max(0, min(100, int(next_progress)))
        if next_status != JOB_STATUS_COMPLETED:
            next_progress = _cap_running_progress(next_progress)
            if current.progress is not None:
                next_progress = (
                    current.progress
                    if next_progress is None
                    else max(current.progress, next_progress)
                )
        updated = replace(
            current,
            stage=next_stage or current.stage,
            progress=next_progress,
            status=next_status,
        )
        self._jobs[current.job_id] = updated
        if updated.is_finished:
            self._cancel_callbacks.pop(updated.job_id, None)
        self.job_updated.emit(updated)
        return updated

    def complete_job(self, job_id: str, stage: str = "Completed") -> None:
        self.update_job(
            job_id,
            stage=stage,
            progress=100,
            status=JOB_STATUS_COMPLETED,
        )

    def fail_job(self, job_id: str, stage: str) -> None:
        self.update_job(job_id, stage=stage, status=JOB_STATUS_FAILED)

    def cancel_job(self, job_id: str) -> bool:
        normalized_job_id = str(job_id)
        job = self._jobs.get(normalized_job_id)
        if job is None or job.status != JOB_STATUS_RUNNING:
            return False
        callback = self._cancel_callbacks.get(normalized_job_id)
        if callback is None:
            return False

        self.update_job(
            normalized_job_id,
            stage="Cancelling...",
            status=JOB_STATUS_CANCELLING,
        )
        accepted = False
        try:
            accepted = bool(callback())
            return accepted
        finally:
            current = self._jobs.get(normalized_job_id)
            if not accepted and current is not None and not current.is_finished:
                self.update_job(
                    normalized_job_id,
                    stage=job.stage,
                    progress=job.progress,
                    status=JOB_STATUS_RUNNING,
                )

    def mark_cancelled(self, job_id: str, stage: str = "Cancelled") -> None:
        self.update_job(job_id, stage=stage, status=JOB_STATUS_CANCELLED)

    def clear_finished(self) -> None:
        finished_ids = {
            job_id
            for job_id, job in self._jobs.items()
            if job.is_finished
        }
        if not finished_ids:
            return
        for job_id in finished_ids:
            self._jobs.pop(job_id, None)
            self._cancel_callbacks.pop(job_id, None)
        self.jobs_cleared.emit()


# ### Jobs window ###
class JobsWindow(QWidget):
    """Detached session job monitor that is independent of active tabs."""

    def __init__(
        self,
        manager: GenerationJobManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setObjectName("jobs_window")
        self.setWindowTitle("Jobs")
        self.resize(_WINDOW_DEFAULT_WIDTH, _WINDOW_DEFAULT_HEIGHT)
        self._manager = manager
        self._target_screen: QScreen | None = None
        self._rows: dict[str, _JobRow] = {}
        self._is_disposed = False
        self._has_been_shown = False
        self._build_ui()
        manager.job_added.connect(self._handle_job_added)
        manager.job_updated.connect(self._handle_job_updated)
        manager.jobs_cleared.connect(self._rebuild_rows)
        self._rebuild_rows()

    def set_target_screen(self, screen: QScreen | None) -> None:
        if self._is_disposed:
            return
        self._target_screen = screen
        if self.isVisible():
            self._move_to_target_screen()

    def dispose(self) -> None:
        if self._is_disposed:
            return
        self._is_disposed = True
        self.hide()
        for signal, slot in (
            (self._manager.job_added, self._handle_job_added),
            (self._manager.job_updated, self._handle_job_updated),
            (self._manager.jobs_cleared, self._rebuild_rows),
        ):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        # Keep the QObject parent as a synchronous fallback if the application
        # event loop stops before this deferred deletion can be delivered.
        self.deleteLater()

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self.hide()
        event.ignore()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        header_layout = QHBoxLayout()
        title = QLabel("Generation jobs")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        header_layout.addWidget(title)
        header_layout.addStretch(1)
        self.clear_finished_button = QPushButton("Clear finished")
        self.clear_finished_button.clicked.connect(
            self._manager.clear_finished
        )
        header_layout.addWidget(self.clear_finished_button)
        root_layout.addLayout(header_layout)

        self.empty_label = QLabel("No jobs in this session.")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root_layout.addWidget(self.empty_label, 1)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.rows_widget = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_widget)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(8)
        self.rows_layout.addStretch(1)
        self.scroll_area.setWidget(self.rows_widget)
        root_layout.addWidget(self.scroll_area, 1)

    def _handle_job_added(self, raw_job: object) -> None:
        if self._is_disposed or not isinstance(raw_job, GenerationJob):
            return
        self._add_or_update_row(raw_job)
        is_first_show = not self._has_been_shown
        self.show()
        self._move_to_target_screen()
        if is_first_show:
            self.showMaximized()
            self._has_been_shown = True
        self.raise_()

    def _handle_job_updated(self, raw_job: object) -> None:
        if self._is_disposed or not isinstance(raw_job, GenerationJob):
            return
        self._add_or_update_row(raw_job)

    def _rebuild_rows(self) -> None:
        if self._is_disposed:
            return
        for row in tuple(self._rows.values()):
            self.rows_layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()
        for job in self._manager.jobs():
            self._add_or_update_row(job)
        self._sync_empty_state()

    def _add_or_update_row(self, job: GenerationJob) -> None:
        row = self._rows.get(job.job_id)
        if row is None:
            row = _JobRow(job.job_id, self._manager.cancel_job)
            self._rows[job.job_id] = row
            self.rows_layout.insertWidget(self.rows_layout.count() - 1, row)
        row.set_job(job)
        self._sync_empty_state()

    def _sync_empty_state(self) -> None:
        has_jobs = bool(self._rows)
        self.empty_label.setVisible(not has_jobs)
        self.scroll_area.setVisible(has_jobs)
        self.clear_finished_button.setEnabled(
            any(job.is_finished for job in self._manager.jobs())
        )

    def _move_to_target_screen(self) -> None:
        if self._is_disposed:
            return
        preferred_screen = self._target_screen
        try:
            primary_screen = QGuiApplication.primaryScreen()
        except RuntimeError:
            primary_screen = None
        screens: list[QScreen] = []
        if preferred_screen is not None:
            screens.append(preferred_screen)
        if (
            primary_screen is not None
            and primary_screen is not preferred_screen
        ):
            screens.append(primary_screen)
        for screen in screens:
            if self._try_move_to_screen(screen):
                if screen is not preferred_screen:
                    self._target_screen = None
                return

    def _try_move_to_screen(self, screen: QScreen) -> bool:
        """Move safely, returning false for a display removed mid-update."""

        try:
            window_handle = self.windowHandle()
            if window_handle is not None:
                window_handle.setScreen(screen)
            geometry = screen.availableGeometry()
            x = geometry.right() - self.width() - _WINDOW_EDGE_OFFSET + 1
            maximum_y = geometry.bottom() - self.height() + 1
            y = min(
                geometry.top() + _WINDOW_EDGE_OFFSET,
                max(geometry.top(), maximum_y),
            )
            self.move(max(geometry.left(), x), y)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False
        return True


class _JobRow(QFrame):
    """Compact visual state for one managed generation job."""

    def __init__(
        self,
        job_id: str,
        cancel_job: Callable[[str], bool],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._job_id = job_id
        self._cancel_job = cancel_job
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(5)

        title_layout = QHBoxLayout()
        self.name_label = QLabel()
        self.name_label.setStyleSheet("font-weight: 600;")
        title_layout.addWidget(self.name_label, 1)
        self.kind_label = QLabel()
        self.kind_label.setStyleSheet("color: #666;")
        title_layout.addWidget(self.kind_label)
        layout.addLayout(title_layout)

        self.stage_label = QLabel()
        self.stage_label.setWordWrap(True)
        layout.addWidget(self.stage_label)

        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar, 1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(
            lambda: self._cancel_job(self._job_id)
        )
        progress_layout.addWidget(self.cancel_button)
        layout.addLayout(progress_layout)

    def set_job(self, job: GenerationJob) -> None:
        self.name_label.setText(job.name)
        self.kind_label.setText(job.kind)
        self.stage_label.setText(job.stage)
        if job.progress is None and not job.is_finished:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("Working")
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(job.progress or 0)
            self.progress_bar.setFormat("%p%")
        self.cancel_button.setEnabled(
            job.status == JOB_STATUS_RUNNING
        )
        self.cancel_button.setVisible(not job.is_finished)


# ### Helpers ###
def _cap_running_progress(progress: int | None) -> int | None:
    """Reserve 100 percent for a transaction that has actually committed."""

    if progress is None:
        return None
    return min(max(int(progress), 0), 99)


def _extract_progress_percent(message: str) -> int | None:
    matches = _PERCENT_PATTERN.findall(str(message))
    if not matches:
        return None
    return int(matches[-1])
