"""Always-on-top manual-input overlay for the TFT advisor prototype."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import Any

from .live_advisor import LiveAdvisorApp


ARCHETYPE_LABELS = {
    "빠른 8레벨 캐리": "fast_8_carry",
    "리롤 결투가": "reroll_duelist",
    "브루저 전방 조합": "bruiser_frontline",
    "스나이퍼 아이템 조합": "sniper_item_slam",
}

BOARD_LABELS = {
    "전방 중심 배치": "frontline",
    "후방 캐리 배치": "backline",
    "구석 캐리 배치": "corner_carry",
    "분산 배치": "spread",
}

ACTION_LABELS = {
    "level_up": "레벨 업",
    "roll_down": "리롤",
    "hold_economy": "골드 유지",
    "slam_item": "아이템 즉시 장착",
    "reposition_carry": "캐리 위치 변경",
}


def display_values(payload: dict[str, Any]) -> dict[str, str]:
    recommendation = payload["recommendation"]
    return {
        "status": f"{payload['source']['label']} · {payload['updated_at']}",
        "detail": payload["source"]["detail"],
        "placement": f"#{recommendation['predicted_placement']}",
        "top4": f"{round(recommendation['predicted_top4_probability'] * 100)}%",
        "cluster": str(recommendation["nearest_cluster"]),
        "accuracy": f"{round(payload['metrics']['top4_accuracy'] * 100)}%",
        "action": ACTION_LABELS.get(recommendation["action"], recommendation["action"]),
        "improvement": f"예상 등수 개선 {recommendation['expected_placement_improvement']:+.2f}",
    }


class OverlayAdvisor:
    refresh_ms = 2000

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.advisor: LiveAdvisorApp | None = None
        self.result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.refresh_in_progress = False
        self.drag_x = 0
        self.drag_y = 0
        self.collapsed = False

        self._configure_window()
        self._build_styles()
        self._build_ui()
        self.root.after(100, self._poll_results)
        threading.Thread(target=self._train_model, daemon=True).start()

    def _configure_window(self) -> None:
        self.root.title("TFT 실시간 예측 오버레이")
        self.root.geometry("430x650+40+80")
        self.root.configure(bg="#0b1220")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.94)
        try:
            self.root.attributes("-toolwindow", True)
        except tk.TclError:
            pass

    def _build_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "Overlay.TCombobox",
            fieldbackground="#111827",
            background="#111827",
            foreground="#f8fafc",
            arrowcolor="#38bdf8",
            bordercolor="#334155",
            lightcolor="#334155",
            darkcolor="#334155",
            padding=8,
        )
        style.map(
            "Overlay.TCombobox",
            fieldbackground=[("readonly", "#111827")],
            foreground=[("readonly", "#f8fafc")],
            selectbackground=[("readonly", "#111827")],
            selectforeground=[("readonly", "#f8fafc")],
        )

    def _build_ui(self) -> None:
        self.header = tk.Frame(self.root, bg="#172033", height=54)
        self.header.pack(fill="x")
        self.header.pack_propagate(False)
        self.header.bind("<ButtonPress-1>", self._start_drag)
        self.header.bind("<B1-Motion>", self._drag)

        title = tk.Label(
            self.header,
            text="TFT 실시간 예측",
            bg="#172033",
            fg="#f8fafc",
            font=("Malgun Gothic", 13, "bold"),
        )
        title.pack(side="left", padx=14)
        title.bind("<ButtonPress-1>", self._start_drag)
        title.bind("<B1-Motion>", self._drag)

        tk.Button(
            self.header,
            text="×",
            command=self.root.destroy,
            bg="#172033",
            fg="#f8fafc",
            activebackground="#ef4444",
            activeforeground="#ffffff",
            relief="flat",
            font=("Arial", 16, "bold"),
            width=3,
        ).pack(side="right")
        tk.Button(
            self.header,
            text="−",
            command=self._toggle_collapsed,
            bg="#172033",
            fg="#f8fafc",
            activebackground="#334155",
            activeforeground="#ffffff",
            relief="flat",
            font=("Arial", 15, "bold"),
            width=3,
        ).pack(side="right")

        self.status_var = tk.StringVar(value="딥러닝 모델 학습 중...")
        tk.Label(
            self.root,
            textvariable=self.status_var,
            bg="#0b1220",
            fg="#7dd3fc",
            anchor="w",
            font=("Malgun Gothic", 9, "bold"),
        ).pack(fill="x", padx=14, pady=(10, 4))

        self.content = tk.Frame(self.root, bg="#0b1220")
        self.content.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        input_panel = tk.Frame(self.content, bg="#172033", padx=12, pady=10)
        input_panel.pack(fill="x", pady=(0, 10))

        self.archetype_var = tk.StringVar(value=next(iter(ARCHETYPE_LABELS)))
        self.board_var = tk.StringVar(value=next(iter(BOARD_LABELS)))
        self._add_combo(input_panel, "현재 메타 덱", self.archetype_var, list(ARCHETYPE_LABELS))
        self._add_combo(input_panel, "현재 보드 배치", self.board_var, list(BOARD_LABELS))

        metrics = tk.Frame(self.content, bg="#0b1220")
        metrics.pack(fill="x")
        self.metric_vars = {
            "placement": tk.StringVar(value="-"),
            "top4": tk.StringVar(value="-"),
            "cluster": tk.StringVar(value="-"),
            "accuracy": tk.StringVar(value="-"),
        }
        metric_labels = [
            ("예측 등수", "placement"),
            ("Top 4 확률", "top4"),
            ("메타 군집", "cluster"),
            ("평가 정확도", "accuracy"),
        ]
        for index, (label, key) in enumerate(metric_labels):
            card = tk.Frame(metrics, bg="#111827", padx=9, pady=8)
            card.grid(row=index // 2, column=index % 2, sticky="nsew", padx=3, pady=3)
            tk.Label(card, text=label, bg="#111827", fg="#94a3b8", font=("Malgun Gothic", 8)).pack(anchor="w")
            tk.Label(
                card,
                textvariable=self.metric_vars[key],
                bg="#111827",
                fg="#f8fafc",
                font=("Malgun Gothic", 19, "bold"),
            ).pack(anchor="w")
        metrics.grid_columnconfigure(0, weight=1)
        metrics.grid_columnconfigure(1, weight=1)

        action_panel = tk.Frame(self.content, bg="#f59e0b", padx=14, pady=12)
        action_panel.pack(fill="x", pady=10)
        tk.Label(
            action_panel,
            text="추천하는 다음 행동",
            bg="#f59e0b",
            fg="#111827",
            font=("Malgun Gothic", 9, "bold"),
        ).pack(anchor="w")
        self.action_var = tk.StringVar(value="모델 준비 중")
        tk.Label(
            action_panel,
            textvariable=self.action_var,
            bg="#f59e0b",
            fg="#111827",
            font=("Malgun Gothic", 23, "bold"),
        ).pack(anchor="w")
        self.improvement_var = tk.StringVar(value="")
        tk.Label(
            action_panel,
            textvariable=self.improvement_var,
            bg="#f59e0b",
            fg="#422006",
            font=("Malgun Gothic", 9),
        ).pack(anchor="w")

        tk.Label(
            self.content,
            text="1~8등 예측 분포",
            bg="#0b1220",
            fg="#cbd5e1",
            font=("Malgun Gothic", 9, "bold"),
        ).pack(anchor="w")
        self.probability_canvas = tk.Canvas(
            self.content,
            height=168,
            bg="#111827",
            highlightthickness=0,
        )
        self.probability_canvas.pack(fill="x", pady=(5, 8))

        self.detail_var = tk.StringVar(value="게임 상태는 오버레이에서 직접 선택해야 합니다.")
        tk.Label(
            self.content,
            textvariable=self.detail_var,
            wraplength=390,
            justify="left",
            anchor="w",
            bg="#0b1220",
            fg="#94a3b8",
            font=("Malgun Gothic", 8),
        ).pack(fill="x")

    def _add_combo(
        self,
        parent: tk.Widget,
        label: str,
        variable: tk.StringVar,
        values: list[str],
    ) -> None:
        tk.Label(
            parent,
            text=label,
            bg="#172033",
            fg="#cbd5e1",
            font=("Malgun Gothic", 9),
        ).pack(anchor="w")
        combo = ttk.Combobox(
            parent,
            textvariable=variable,
            values=values,
            state="readonly",
            style="Overlay.TCombobox",
        )
        combo.pack(fill="x", pady=(3, 9))
        combo.bind("<<ComboboxSelected>>", lambda _event: self._request_refresh())

    def _train_model(self) -> None:
        try:
            advisor = LiveAdvisorApp()
            self.result_queue.put(("ready", advisor))
        except Exception as exc:
            self.result_queue.put(("error", f"모델 준비 실패: {exc}"))

    def _request_refresh(self) -> None:
        if self.advisor is None or self.refresh_in_progress:
            return
        self.refresh_in_progress = True
        archetype = ARCHETYPE_LABELS[self.archetype_var.get()]
        board_style = BOARD_LABELS[self.board_var.get()]
        threading.Thread(
            target=self._calculate_recommendation,
            args=(archetype, board_style),
            daemon=True,
        ).start()

    def _calculate_recommendation(self, archetype: str, board_style: str) -> None:
        try:
            payload = self.advisor.recommend(archetype, board_style) if self.advisor else None
            self.result_queue.put(("payload", payload))
        except Exception as exc:
            self.result_queue.put(("error", f"예측 실패: {exc}"))

    def _poll_results(self) -> None:
        try:
            while True:
                kind, value = self.result_queue.get_nowait()
                if kind == "ready":
                    self.advisor = value
                    self.status_var.set("모델 준비 완료")
                    self._request_refresh()
                elif kind == "payload" and value is not None:
                    self.refresh_in_progress = False
                    self._apply_payload(value)
                elif kind == "error":
                    self.refresh_in_progress = False
                    self.status_var.set(str(value))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_results)

    def _apply_payload(self, payload: dict[str, Any]) -> None:
        values = display_values(payload)
        self.status_var.set(values["status"])
        self.detail_var.set(values["detail"] + "\n실제 보드 변화는 자동 감지하지 않습니다.")
        for key in self.metric_vars:
            self.metric_vars[key].set(values[key])
        self.action_var.set(values["action"])
        self.improvement_var.set(values["improvement"])
        self._draw_probabilities(payload["recommendation"]["placement_probabilities"])
        self.root.after(self.refresh_ms, self._request_refresh)

    def _draw_probabilities(self, probabilities: list[float]) -> None:
        canvas = self.probability_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 360)
        for index, probability in enumerate(probabilities):
            y = 12 + index * 19
            canvas.create_text(12, y, text=f"#{index + 1}", fill="#cbd5e1", anchor="w", font=("Arial", 8))
            canvas.create_rectangle(45, y - 5, width - 52, y + 5, fill="#263244", outline="")
            bar_width = (width - 97) * probability
            canvas.create_rectangle(45, y - 5, 45 + bar_width, y + 5, fill="#38bdf8", outline="")
            canvas.create_text(
                width - 8,
                y,
                text=f"{round(probability * 100)}%",
                fill="#cbd5e1",
                anchor="e",
                font=("Arial", 8),
            )

    def _start_drag(self, event: tk.Event) -> None:
        self.drag_x = event.x_root - self.root.winfo_x()
        self.drag_y = event.y_root - self.root.winfo_y()

    def _drag(self, event: tk.Event) -> None:
        self.root.geometry(f"+{event.x_root - self.drag_x}+{event.y_root - self.drag_y}")

    def _toggle_collapsed(self) -> None:
        self.collapsed = not self.collapsed
        if self.collapsed:
            self.content.pack_forget()
            self.root.geometry("430x92")
        else:
            self.content.pack(fill="both", expand=True, padx=14, pady=(0, 14))
            self.root.geometry("430x650")


def main() -> None:
    print("TFT 수동 입력형 실시간 오버레이를 시작합니다.", flush=True)
    print("TFT는 창 모드 또는 테두리 없는 창 모드를 권장합니다.", flush=True)
    root = tk.Tk()
    OverlayAdvisor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
