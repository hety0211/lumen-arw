"""1.3.0: explicit work state and the single-owner job scheduler."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from test_ui import app, wait_until
from lumen.scheduler import Activity as A, JobScheduler, WorkState


def test_conflict_table_matches_previous_entry_point_checks(app):
    work = WorkState()
    assert work.phase == 'idle' and work.describe() == '空闲'
    assert work.begin(A.LOADING)
    assert not work.can_start(A.EXPORTING) and not work.can_start(A.AI) and not work.can_start(A.RENDER)
    assert not work.begin(A.LOADING)
    work.end(A.LOADING)
    assert work.begin(A.SELECTION)
    # A running mask inference blocks export and AI but not opening another photograph.
    assert not work.can_start(A.EXPORTING) and not work.can_start(A.AI) and work.can_start(A.LOADING)
    assert work.busy() and work.phase == 'selection'
    work.end(A.SELECTION)
    assert work.begin(A.RENDER) and work.can_start(A.EXPORTING) and not work.busy()
    work.closing = True
    assert not work.can_start(A.THUMBNAILS) and work.phase == 'closing'
    events = [event for _, event, _ in work.transitions]
    assert events == ['begin', 'end', 'begin', 'end', 'begin']


def test_legacy_window_flags_route_through_work_state(app):
    from lumen.app import MainWindow
    w = MainWindow()
    w.ai_busy = True
    assert w.work.active(A.AI) and not w.work.can_start(A.EXPORTING)
    w.selection_busy = True
    assert w.work.busy(A.SELECTION)
    w.ai_busy = w.selection_busy = False
    assert not w.work.busy() and w.jobs is w.scheduler.jobs and w.queued_jobs is w.scheduler.queue
    w.close()


class RecordingPool:
    def __init__(self):
        self.started = []

    def start(self, job, priority):
        self.started.append(job)


def test_scheduler_runs_one_job_highest_priority_first_then_fifo(app):
    pool = RecordingPool()
    scheduler = JobScheduler(pool)
    names = []
    jobs = [scheduler.submit(lambda n=n: n, names.append, names.append, priority=p)
            for n, p in (('first', 0), ('low-a', 0), ('high', 10), ('low-b', 0))]
    assert pool.started == [jobs[0]] and scheduler.active is jobs[0]
    for expected in ('first', 'high', 'low-a', 'low-b'):
        job = scheduler.active
        job.run()
        wait_until(lambda: scheduler.active is not job)
    assert names == ['first', 'high', 'low-a', 'low-b'] and not scheduler.jobs


def test_scheduler_shutdown_drops_queue_and_drains(app):
    pool = RecordingPool()
    scheduler = JobScheduler(pool)
    drained, results = [], []
    scheduler.drained.connect(lambda: drained.append(True))
    active = scheduler.submit(lambda: 1, results.append, results.append)
    scheduler.submit(lambda: 2, results.append, results.append)
    assert scheduler.shutdown() is False and not scheduler.queue
    active.run()
    wait_until(lambda: drained)
    assert results == [] and scheduler.active is None
    assert scheduler.submit(lambda: 3, results.append, results.append) is None
