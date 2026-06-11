"""Small desktop UI for offline TFT recording analysis."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .video_analyzer import RecordingAnalyzer, save_report


class VideoAnalysisApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TFT 녹화 영상 승부 예측")
        self.root.geometry("620x360")
        self.root.configure(bg="#0f172a")
        self.result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.video_path = tk.StringVar()
        self.interval = tk.StringVar(value="5")
        self.status = tk.StringVar(value="분석할 TFT 녹화 영상을 선택하세요.")
        self.report_path: Path | None = None
        self._build_ui()
        self.root.after(100, self._poll_results)

    def _build_ui(self) -> None:
        frame = tk.Frame(self.root, bg="#172033", padx=22, pady=20)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        tk.Label(
            frame,
            text="TFT 녹화 영상 승부 예측",
            bg="#172033",
            fg="#f8fafc",
            font=("Malgun Gothic", 18, "bold"),
        ).pack(anchor="w")
        tk.Label(
            frame,
            text="라이브 게임이 아닌 녹화 영상을 오프라인으로 분석합니다.",
            bg="#172033",
            fg="#94a3b8",
            font=("Malgun Gothic", 9),
        ).pack(anchor="w", pady=(3, 18))

        path_row = tk.Frame(frame, bg="#172033")
        path_row.pack(fill="x")
        tk.Entry(
            path_row,
            textvariable=self.video_path,
            bg="#111827",
            fg="#f8fafc",
            insertbackground="#ffffff",
            relief="flat",
            font=("Malgun Gothic", 9),
        ).pack(side="left", fill="x", expand=True, ipady=8)
        tk.Button(
            path_row,
            text="영상 선택",
            command=self._select_video,
            bg="#0ea5e9",
            fg="#ffffff",
            relief="flat",
            padx=14,
            pady=8,
        ).pack(side="left", padx=(8, 0))

        options = tk.Frame(frame, bg="#172033")
        options.pack(fill="x", pady=16)
        tk.Label(
            options,
            text="프레임 분석 간격(초)",
            bg="#172033",
            fg="#cbd5e1",
        ).pack(side="left")
        tk.Spinbox(
            options,
            textvariable=self.interval,
            from_=1,
            to=30,
            width=6,
            bg="#111827",
            fg="#f8fafc",
            buttonbackground="#334155",
            relief="flat",
        ).pack(side="left", padx=10)

        self.progress = ttk.Progressbar(frame, maximum=100)
        self.progress.pack(fill="x", pady=(4, 10))
        tk.Label(
            frame,
            textvariable=self.status,
            wraplength=550,
            justify="left",
            bg="#172033",
            fg="#7dd3fc",
            font=("Malgun Gothic", 9),
        ).pack(anchor="w")

        buttons = tk.Frame(frame, bg="#172033")
        buttons.pack(fill="x", side="bottom")
        self.analyze_button = tk.Button(
            buttons,
            text="녹화 영상 분석",
            command=self._start_analysis,
            bg="#f59e0b",
            fg="#111827",
            relief="flat",
            padx=18,
            pady=9,
            font=("Malgun Gothic", 9, "bold"),
        )
        self.analyze_button.pack(side="left")
        self.report_button = tk.Button(
            buttons,
            text="결과 보고서 열기",
            command=self._open_report,
            state="disabled",
            bg="#334155",
            fg="#f8fafc",
            relief="flat",
            padx=18,
            pady=9,
        )
        self.report_button.pack(side="left", padx=8)

    def _select_video(self) -> None:
        selected = filedialog.askopenfilename(
            title="TFT 녹화 영상 선택",
            filetypes=[
                ("Video files", "*.mp4 *.mkv *.avi *.mov *.webm"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.video_path.set(selected)

    def _start_analysis(self) -> None:
        path = Path(self.video_path.get())
        if not path.is_file():
            messagebox.showerror("영상 오류", "분석할 영상 파일을 선택하세요.")
            return
        try:
            interval = float(self.interval.get())
            if interval <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("입력 오류", "분석 간격은 0보다 큰 숫자여야 합니다.")
            return

        self.analyze_button.configure(state="disabled")
        self.report_button.configure(state="disabled")
        self.progress["value"] = 0
        self.status.set("모델을 준비하고 영상을 분석하는 중입니다...")
        threading.Thread(
            target=self._analyze_worker,
            args=(path, interval),
            daemon=True,
        ).start()

    def _analyze_worker(self, path: Path, interval: float) -> None:
        try:
            analyzer = RecordingAnalyzer()
            predictions = analyzer.analyze(
                path,
                interval,
                lambda progress: self.result_queue.put(("progress", progress)),
            )
            output_dir = Path("outputs/video_analysis") / path.stem
            _, report = save_report(path, predictions, output_dir, analyzer.metrics)
            self.result_queue.put(("done", (report.resolve(), len(predictions))))
        except Exception as exc:
            self.result_queue.put(("error", str(exc)))

    def _poll_results(self) -> None:
        try:
            while True:
                kind, value = self.result_queue.get_nowait()
                if kind == "progress":
                    self.progress["value"] = float(value) * 100
                elif kind == "done":
                    report, count = value
                    self.report_path = report
                    self.progress["value"] = 100
                    self.status.set(f"분석 완료: {count}개 시점의 승부 확률을 계산했습니다.")
                    self.analyze_button.configure(state="normal")
                    self.report_button.configure(state="normal")
                elif kind == "error":
                    self.status.set(f"분석 실패: {value}")
                    self.analyze_button.configure(state="normal")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_results)

    def _open_report(self) -> None:
        if self.report_path:
            webbrowser.open(self.report_path.as_uri())


def main() -> None:
    root = tk.Tk()
    VideoAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
