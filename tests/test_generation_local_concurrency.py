# ### Imports ###
from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from housemaker import generation_workspace, local_work_scheduler


# ### Local generation concurrency tests ###
class LocalGenerationConcurrencyTests(unittest.TestCase):
    def test_memory_heavy_local_stages_share_a_bounded_lane(self) -> None:
        slot_count = generation_workspace.MAX_CONCURRENT_LOCAL_GENERATION_STAGES
        stage_slots = threading.BoundedSemaphore(slot_count)
        release_operations = threading.Event()
        saturated = threading.Event()
        state_lock = threading.Lock()
        active_count = 0
        maximum_active_count = 0
        failures: list[BaseException] = []

        def operation() -> None:
            nonlocal active_count, maximum_active_count
            with state_lock:
                active_count += 1
                maximum_active_count = max(maximum_active_count, active_count)
                if active_count == slot_count:
                    saturated.set()
            try:
                if not release_operations.wait(timeout=5.0):
                    raise TimeoutError("The local-stage concurrency test timed out.")
            finally:
                with state_lock:
                    active_count -= 1

        def run_operation() -> None:
            try:
                generation_workspace._invoke_bounded_local_generation_stage(
                    operation,
                    None,
                )
            except (OSError, RuntimeError) as error:  # pragma: no cover
                failures.append(error)

        workers = [
            threading.Thread(target=run_operation, daemon=True)
            for _index in range(slot_count + 2)
        ]
        with patch.object(
            local_work_scheduler,
            "_LOCAL_WORK_SLOTS",
            stage_slots,
        ):
            for worker in workers:
                worker.start()
            self.assertTrue(saturated.wait(timeout=2.0))
            with state_lock:
                self.assertEqual(active_count, slot_count)
                self.assertEqual(maximum_active_count, slot_count)
            release_operations.set()
            for worker in workers:
                worker.join(timeout=2.0)

        self.assertFalse(failures)
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertEqual(maximum_active_count, slot_count)

    def test_cancelled_stage_does_not_consume_a_slot(self) -> None:
        stage_slots = threading.BoundedSemaphore(1)
        cancel_event = threading.Event()
        cancel_event.set()
        operation_called = False

        def operation() -> None:
            nonlocal operation_called
            operation_called = True

        with (
            patch.object(
                local_work_scheduler,
                "_LOCAL_WORK_SLOTS",
                stage_slots,
            ),
            self.assertRaises(generation_workspace._GenerationCancelled),
        ):
            generation_workspace._invoke_bounded_local_generation_stage(
                operation,
                cancel_event,
            )

        self.assertFalse(operation_called)
        self.assertTrue(stage_slots.acquire(blocking=False))
        stage_slots.release()


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
