"""
Log viewer component for the Gradio UI.
"""

from typing import Any, Dict, List

import gradio as gr


class LogViewer:
    """
    Encapsulates the log viewing accordion and its background update logic.
    """

    def __init__(self, logger: Any, progress_queue: Any, log_level_choices: List[str]):
        self.logger = logger
        self.progress_queue = progress_queue
        self.log_level_choices = log_level_choices
        self.all_logs: List[str] = []
        self.log_filter_level = "INFO"
        self.components: Dict[str, Any] = {}

    def build(self) -> gr.Accordion:
        """Constructs the log viewer UI components."""
        with gr.Accordion("📋 System Logs", open=False) as accordion:
            self.components["unified_log"] = gr.Textbox(
                lines=15,
                interactive=False,
                autoscroll=True,
                elem_classes=["log-container"],
                elem_id="unified_log",
                value="Ready. Operations will be logged here.",
            )
            with gr.Row():
                self.components["show_debug_logs"] = gr.Checkbox(
                    label="Show Debug Logs", value=False
                )
                self.components["refresh_logs_button"] = gr.Button(
                    "🔄 Refresh", scale=1, visible=True
                )
                self.components["clear_logs_button"] = gr.Button("🗑️ Clear", scale=1)

        return accordion

    def get_log_update_dict(self, new_log_msg: str = None) -> Dict[Any, str]:
        """Returns a Gradio update dictionary for the unified log component."""
        if new_log_msg:
            self.all_logs.append(new_log_msg)

        log_level_map = {level: i for i, level in enumerate(self.log_level_choices)}
        current_filter_level = log_level_map.get(self.log_filter_level.upper(), 1)
        filtered_logs = [
            l
            for l in self.all_logs
            if any(
                f"[{level}]" in l for level in self.log_level_choices[current_filter_level:]
            )
        ]
        return {self.components["unified_log"]: "\n".join(filtered_logs[-1000:])}

    def setup_handlers(self, all_outputs: List[Any]):
        """Sets up background log refresh and filter toggles."""

        def update_logs(filter_debug):
            """Refreshes log display by draining queue and applying filter."""
            # First drain any pending log messages from the queue
            while not self.progress_queue.empty():
                try:
                    msg = self.progress_queue.get_nowait()
                    if "log" in msg:
                        self.all_logs.append(msg["log"])
                    if "ui_update" in msg:
                        yield msg["ui_update"]
                except Exception:
                    break

            level = "DEBUG" if filter_debug else "INFO"
            self.log_filter_level = level
            log_level_map = {l: i for i, l in enumerate(self.log_level_choices)}
            current_filter_level = log_level_map.get(level.upper(), 1)
            filtered_logs = [
                l
                for l in self.all_logs
                if any(f"[{lvl}]" in l for lvl in self.log_level_choices[current_filter_level:])
            ]
            yield {self.components["unified_log"]: "\n".join(filtered_logs[-1000:])}

        c = self.components
        c["clear_logs_button"].click(
            lambda: (self.all_logs.clear(), "")[1], [], c["unified_log"]
        )

        c["show_debug_logs"].change(
            update_logs, inputs=[c["show_debug_logs"]], outputs=all_outputs
        )

        # Log Auto-Refresh (every 1s)
        self.components["log_timer"] = gr.Timer(1.0)
        self.components["log_timer"].tick(
            update_logs, inputs=[c["show_debug_logs"]], outputs=all_outputs
        )

        # Keep manual refresh just in case
        c["refresh_logs_button"].click(
            update_logs, inputs=[c["show_debug_logs"]], outputs=all_outputs
        )
