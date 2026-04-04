"""
Log viewer component for the Gradio UI.
"""

from collections import deque
from typing import Any, Dict, List, Optional

import gradio as gr


class LogViewer:
    """
    Encapsulates the log viewing accordion and its background update logic.
    """

    def __init__(self, logger: Any, progress_queue: Any, log_level_choices: List[str]):
        self.logger = logger
        self.progress_queue = progress_queue
        self.log_level_choices = log_level_choices
        # Fix Issue: Use deque for thread-safety and bounded memory
        self.all_logs: deque[str] = deque(maxlen=1000)
        self.log_filter_level = "INFO"
        self.components: Dict[str, Any] = {}
        self._last_rendered_log: str = ""  # Track last emitted content

    def build(self) -> gr.Accordion:
        """Constructs the log viewer UI components."""
        with gr.Accordion("📋 System Logs", open=False, elem_id="system_logs_accordion") as accordion:
            self.components["unified_log"] = gr.Textbox(
                label="System Logs Output",
                lines=15,
                interactive=False,
                autoscroll=True,
                elem_classes=["log-container"],
                elem_id="unified_log",
                value="Ready. Operations will be logged here.",
            )
            with gr.Row():
                self.components["show_debug_logs"] = gr.Checkbox(label="Show Debug Logs", value=False)
                self.components["refresh_logs_button"] = gr.Button(
                    "🔄 Refresh", scale=1, visible=True, elem_id="refresh_logs_button"
                )
                self.components["clear_logs_button"] = gr.Button("Clear", scale=1)

        return accordion

    def get_log_update_dict(self, new_log_msg: Optional[str] = None) -> Dict[Any, str]:
        """Returns a Gradio update dictionary for the unified log component."""
        if new_log_msg:
            self.all_logs.append(new_log_msg)

        log_level_map = {level: i for i, level in enumerate(self.log_level_choices)}
        current_filter_level = log_level_map.get(self.log_filter_level.upper(), 1)
        # Convert deque to list for filtering
        filtered_logs = [
            l
            for l in list(self.all_logs)
            if any(f"[{level}]" in l for level in self.log_level_choices[current_filter_level:])
        ]
        return {self.components["unified_log"]: "\n".join(filtered_logs)}

    def setup_handlers(self, timer_outputs: List[Any], full_outputs: List[Any]):
        """Sets up background log refresh and filter toggles."""

        def update_logs(filter_debug):
            """Refreshes log display by draining queue and applying filter."""
            # Fix Issue 2: Drain all messages atomically before filtering to ensure determinism
            new_msgs = []
            while not self.progress_queue.empty():
                try:
                    msg = self.progress_queue.get_nowait()
                    if "log" in msg:
                        new_msgs.append(msg["log"])
                except Exception:
                    break

            if new_msgs:
                self.all_logs.extend(new_msgs)

            level = "DEBUG" if filter_debug else "INFO"
            self.log_filter_level = level
            log_level_map = {l: i for i, l in enumerate(self.log_level_choices)}
            current_filter_level = log_level_map.get(level.upper(), 1)

            # Use atomic snapshot of logs
            current_logs = list(self.all_logs)
            filtered_logs = [
                l for l in current_logs if any(f"[{lvl}]" in l for lvl in self.log_level_choices[current_filter_level:])
            ]
            new_content = "\n".join(filtered_logs)
            self._last_rendered_log = new_content
            # Return (not yield) so Gradio always applies the update.
            return {self.components["unified_log"]: new_content}

        c = self.components
        c["clear_logs_button"].click(lambda: (self.all_logs.clear(), "")[1], [], c["unified_log"])

        c["show_debug_logs"].change(update_logs, inputs=[c["show_debug_logs"]], outputs=full_outputs)

        # Log Auto-Refresh (every 0.5s)
        self.components["log_timer"] = gr.Timer(0.5)
        self.components["log_timer"].tick(update_logs, inputs=[c["show_debug_logs"]], outputs=timer_outputs)

        # Manual refresh (full outputs)
        c["refresh_logs_button"].click(update_logs, inputs=[c["show_debug_logs"]], outputs=full_outputs)
