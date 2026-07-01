from __future__ import annotations

import tkinter as tk

import config


class StatCard(tk.Frame):
    def __init__(self, master: tk.Misc, title: str, accent: str) -> None:
        super().__init__(master, bg=config.COLOR_PANEL_ALT, padx=16, pady=14)

        self._title = tk.Label(
            self,
            text=title,
            font=("Segoe UI", 11, "bold"),
            fg=accent,
            bg=config.COLOR_PANEL_ALT,
            anchor="w",
        )
        self._title.pack(fill="x")

        self._value = tk.Label(
            self,
            text="0",
            font=("Segoe UI", 36, "bold"),
            fg=config.COLOR_TEXT,
            bg=config.COLOR_PANEL_ALT,
            anchor="w",
        )
        self._value.pack(fill="x")

        self._total = tk.Label(
            self,
            text="всего за сессию: 0",
            font=("Segoe UI", 10),
            fg=config.COLOR_TEXT_MUTED,
            bg=config.COLOR_PANEL_ALT,
            anchor="w",
        )
        self._total.pack(fill="x")

    def update_values(self, current: int, total: int) -> None:
        self._value.configure(text=str(current))
        self._total.configure(text=f"всего за сессию: {total}")


class DetailRow(tk.Frame):
    def __init__(self, master: tk.Misc, label: str) -> None:
        super().__init__(master, bg=config.COLOR_PANEL_ALT)

        self._label = tk.Label(
            self,
            text=label,
            font=("Segoe UI", 11),
            fg=config.COLOR_TEXT,
            bg=config.COLOR_PANEL_ALT,
            anchor="w",
        )
        self._label.pack(side="left", padx=(4, 0), pady=3)

        self._count = tk.Label(
            self,
            text="0",
            font=("Segoe UI", 11, "bold"),
            fg=config.COLOR_TEXT,
            bg=config.COLOR_PANEL_ALT,
            anchor="e",
        )
        self._count.pack(side="right", padx=(0, 4), pady=3)

    def set_count(self, value: int) -> None:
        color = config.COLOR_TEXT if value > 0 else config.COLOR_TEXT_MUTED
        self._count.configure(text=str(value), fg=color)
        self._label.configure(fg=color)


class FlatButton(tk.Button):
    def __init__(
        self,
        master: tk.Misc,
        text: str,
        command,
        bg: str,
        hover: str,
        fg: str = config.COLOR_TEXT,
    ) -> None:
        super().__init__(
            master,
            text=text,
            command=command,
            font=("Segoe UI", 10, "bold"),
            bg=bg,
            fg=fg,
            activebackground=hover,
            activeforeground=fg,
            relief="flat",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2",
            highlightthickness=0,
        )
        self._bg = bg
        self._hover = hover
        self.bind("<Enter>", lambda _e: self._on_hover(True))
        self.bind("<Leave>", lambda _e: self._on_hover(False))

    def _on_hover(self, entered: bool) -> None:
        if str(self["state"]) == "disabled":
            return
        self.configure(bg=self._hover if entered else self._bg)

    def set_base_colors(self, bg: str, hover: str) -> None:
        self._bg = bg
        self._hover = hover
        self.configure(bg=bg, activebackground=hover)
