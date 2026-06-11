"""Always-on-top manual-input overlay for the TFT advisor prototype."""

from __future__ import annotations

import queue
import threading
from typing import Any

from .game_state import GameState, with_flow
from .live_advisor import LiveAdvisorApp

tk: Any = None
ttk: Any = None


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
        "reason": recommendation.get("reason", ""),
    }


class OverlayAdvisor:
    refresh_ms = 2000

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.window_width = min(500, max(440, self.root.winfo_screenwidth() - 40))
        self.window_height = min(820, max(700, self.root.winfo_screenheight() - 30))
        self.advisor: LiveAdvisorApp | None = None
        self.result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.refresh_in_progress = False
        self.drag_x = 0
        self.drag_y = 0
        self.collapsed = False
        self.current_game_state: GameState | None = None

        self._configure_window()
        self._build_styles()
        self._build_ui()
        self.root.after(100, self._poll_results)
        threading.Thread(target=self._train_model, daemon=True).start()

    def _configure_window(self) -> None:
        self.root.title("TFT 실시간 예측 오버레이")
        self.root.geometry(f"{self.window_width}x{self.window_height}+20+10")
        self.root.minsize(440, 700)
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

        self.content_container = tk.Frame(self.root, bg="#0b1220")
        self.content_container.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self.content_canvas = tk.Canvas(
            self.content_container,
            bg="#0b1220",
            highlightthickness=0,
        )
        scrollbar = tk.Scrollbar(
            self.content_container,
            orient="vertical",
            command=self.content_canvas.yview,
        )
        self.content_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.content_canvas.pack(side="left", fill="both", expand=True)

        self.content = tk.Frame(self.content_canvas, bg="#0b1220")
        self.content_window = self.content_canvas.create_window(
            (0, 0),
            window=self.content,
            anchor="nw",
        )
        self.content.bind(
            "<Configure>",
            lambda _event: self.content_canvas.configure(
                scrollregion=self.content_canvas.bbox("all")
            ),
        )
        self.content_canvas.bind(
            "<Configure>",
            lambda event: self.content_canvas.itemconfigure(
                self.content_window,
                width=event.width,
            ),
        )
        self.content_canvas.bind_all("<MouseWheel>", self._scroll_content)

        input_panel = tk.Frame(self.content, bg="#172033", padx=12, pady=8)
        input_panel.pack(fill="x", pady=(0, 10))

        self.archetype_var = tk.StringVar(value=next(iter(ARCHETYPE_LABELS)))
        self.board_var = tk.StringVar(value=next(iter(BOARD_LABELS)))
        self._add_combo(input_panel, "현재 메타 덱", self.archetype_var, list(ARCHETYPE_LABELS))
        self._add_combo(input_panel, "현재 보드 배치", self.board_var, list(BOARD_LABELS))

        state_grid = tk.Frame(input_panel, bg="#172033")
        state_grid.pack(fill="x", pady=(2, 4))
        self.state_vars = {
            "stage": tk.StringVar(value="3.2"),
            "health": tk.StringVar(value="80"),
            "gold": tk.StringVar(value="30"),
            "level": tk.StringVar(value="6"),
            "streak": tk.StringVar(value="0"),
            "board_strength": tk.StringVar(value="6"),
            "unspent_items": tk.StringVar(value="1"),
        }
        state_fields = [
            ("스테이지", "stage", 2.1, 6.9, 0.1),
            ("체력", "health", 1, 100, 1),
            ("골드", "gold", 0, 100, 1),
            ("레벨", "level", 4, 10, 1),
            ("연승(+)/연패(-)", "streak", -5, 5, 1),
            ("보드 강도", "board_strength", 1, 10, 1),
            ("대기 아이템", "unspent_items", 0, 6, 1),
        ]
        for index, field in enumerate(state_fields):
            self._add_state_field(
                state_grid,
                *field,
                row=index // 4,
                column=index % 4,
            )
        for column in range(4):
            state_grid.grid_columnconfigure(column, weight=1)

        tk.Button(
            input_panel,
            text="현재 게임 상태 반영",
            command=self._commit_state,
            bg="#0ea5e9",
            fg="#ffffff",
            activebackground="#0284c7",
            activeforeground="#ffffff",
            relief="flat",
            font=("Malgun Gothic", 9, "bold"),
            pady=5,
        ).pack(fill="x", pady=(4, 0))

        self.flow_var = tk.StringVar(value="실제 게임 상태를 입력하고 상태 반영을 눌러주세요.")
        tk.Label(
            input_panel,
            textvariable=self.flow_var,
            bg="#172033",
            fg="#7dd3fc",
            anchor="w",
            font=("Malgun Gothic", 8),
        ).pack(fill="x", pady=(4, 0))

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

        action_panel = tk.Frame(self.content, bg="#f59e0b", padx=14, pady=9)
        action_panel.pack(fill="x", pady=8)
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
        self.reason_var = tk.StringVar(value="")
        tk.Label(
            action_panel,
            textvariable=self.reason_var,
            wraplength=430,
            justify="left",
            bg="#f59e0b",
            fg="#422006",
            font=("Malgun Gothic", 8),
        ).pack(anchor="w", pady=(2, 0))

        tk.Label(
            self.content,
            text="1~8등 예측 분포",
            bg="#0b1220",
            fg="#cbd5e1",
            font=("Malgun Gothic", 9, "bold"),
        ).pack(anchor="w")
        self.probability_canvas = tk.Canvas(
            self.content,
            height=130,
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
        combo.pack(fill="x", pady=(3, 6))
        combo.bind("<<ComboboxSelected>>", lambda _event: self._request_refresh())

    def _add_state_field(
        self,
        parent: tk.Widget,
        label: str,
        key: str,
        minimum: float,
        maximum: float,
        increment: float,
        row: int,
        column: int,
    ) -> None:
        frame = tk.Frame(parent, bg="#172033")
        frame.grid(row=row, column=column, sticky="ew", padx=3, pady=3)
        tk.Label(
            frame,
            text=label,
            bg="#172033",
            fg="#94a3b8",
            font=("Malgun Gothic", 7),
        ).pack(anchor="w")
        tk.Spinbox(
            frame,
            textvariable=self.state_vars[key],
            from_=minimum,
            to=maximum,
            increment=increment,
            bg="#111827",
            fg="#f8fafc",
            buttonbackground="#334155",
            insertbackground="#ffffff",
            relief="flat",
            width=7,
            font=("Malgun Gothic", 9),
        ).pack(fill="x")

    def _read_game_state(self) -> GameState:
        return GameState(
            stage=float(self.state_vars["stage"].get()),
            health=int(float(self.state_vars["health"].get())),
            gold=int(float(self.state_vars["gold"].get())),
            level=int(float(self.state_vars["level"].get())),
            streak=int(float(self.state_vars["streak"].get())),
            board_strength=int(float(self.state_vars["board_strength"].get())),
            unspent_items=int(float(self.state_vars["unspent_items"].get())),
        )

    def _commit_state(self) -> None:
        try:
            raw_state = self._read_game_state()
        except ValueError:
            self.status_var.set("입력값을 숫자로 확인해 주세요.")
            return

        previous = self.current_game_state
        self.current_game_state = with_flow(raw_state, previous)
        if previous is None:
            self.flow_var.set("첫 게임 상태가 저장되었습니다.")
        else:
            state = self.current_game_state
            self.flow_var.set(
                f"변화: 체력 {state.health_delta:+d}, 골드 {state.gold_delta:+d}, "
                f"레벨 {state.level_delta:+d}, 보드 {state.board_strength_delta:+d}"
            )
        self._request_refresh()

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
            args=(archetype, board_style, self.current_game_state),
            daemon=True,
        ).start()

    def _calculate_recommendation(
        self,
        archetype: str,
        board_style: str,
        game_state: GameState | None,
    ) -> None:
        try:
            payload = (
                self.advisor.recommend(archetype, board_style, game_state)
                if self.advisor
                else None
            )
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
                    self._commit_state()
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
        self.detail_var.set(
            values["detail"]
            + "\n입력한 라운드·체력·골드·레벨·연승/연패·보드 변화가 예측에 반영됩니다."
        )
        for key in self.metric_vars:
            self.metric_vars[key].set(values[key])
        self.action_var.set(values["action"])
        self.improvement_var.set(values["improvement"])
        self.reason_var.set(values["reason"])
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

    def _scroll_content(self, event: tk.Event) -> None:
        if not self.collapsed:
            self.content_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _toggle_collapsed(self) -> None:
        self.collapsed = not self.collapsed
        if self.collapsed:
            self.content_container.pack_forget()
            self.root.geometry(f"{self.window_width}x92")
        else:
            self.content_container.pack(fill="both", expand=True, padx=14, pady=(0, 14))
            self.root.geometry(f"{self.window_width}x{self.window_height}")


def main() -> None:
    global tk, ttk
    try:
        import tkinter as tk_module
        from tkinter import ttk as ttk_module
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Tkinter is required for overlay mode. On Windows, install Python from python.org "
            "with the tcl/tk option enabled. You can still run the browser advisor with "
            "`python -m src.tft_advisor.live_advisor`."
        ) from exc
    tk = tk_module
    ttk = ttk_module
    print("TFT 수동 입력형 실시간 오버레이를 시작합니다.", flush=True)
    print("TFT는 창 모드 또는 테두리 없는 창 모드를 권장합니다.", flush=True)
    root = tk.Tk()
    OverlayAdvisor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
