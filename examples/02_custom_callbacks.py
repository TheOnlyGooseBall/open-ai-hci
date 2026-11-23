"""Example 2: bind your own Python functions to gestures.

Run:  python examples/02_custom_callbacks.py            (cursor untouched, prints events)
      python examples/02_custom_callbacks.py --live     (also sends real hotkeys: browser back/forward on head flicks)

Head flicks are opt-in since v0.2 (they are great for harmless actions like back/forward, risky for clicks).

The mouse action is removed here, so nothing moves your cursor; the point is the event API:
  * hci.on(name, fn) / @hci.on(name) / hci.once(...) / hci.on("*", fn)
  * event payloads are readable as attributes (event.x, event.direction, ...)
  * your own events through hci.emit(...)
  * OSKeyboardAction: bind events to hotkeys, macros or callables
"""
import sys

from open_ai_hci import HCIConfig, HCIEngine, MouseConfig, OSKeyboardAction

live = "--live" in sys.argv

config = HCIConfig(mouse=MouseConfig(enabled=False))   # tracking + events only, no cursor control
config.face.flick_enabled = True                       # opt in to head_flick events (turns the face sensor on)
hci = HCIEngine(config)

pinches = 0


@hci.on("pinch_start")                                 # decorator style
def on_pinch(event):
    global pinches
    pinches += 1
    print(f"[{event.frame_index}] pinch #{pinches} at ({event.x:.0f}, {event.y:.0f}), distance {event.distance:.0f}px")


hci.on("pinch_end", lambda e: print(f"    released ({e.reason})"))   # lambda style
hci.on("right_click", lambda e: print(f"right click via {e.via} at ({e.x:.0f}, {e.y:.0f})"))   # thumb + middle finger pinch
hci.on("control_paused", lambda e: print("paused (hold an open palm to resume)"))
hci.on("control_resumed", lambda e: print("resumed"))
hci.on("drag_end", lambda e: print(f"    drag finished at ({e.x:.0f}, {e.y:.0f})"))


@hci.on("head_flick")
def on_flick(event):
    print(f"head flick {event.direction}: dx={event.dx:.0f}px")
    hci.emit("celebrate", reason="flick")              # publish your own event


hci.bus.register_event("celebrate")                    # declare it so typos get caught
hci.on("celebrate", lambda e: print("  🎉 custom event:", e.reason))

hci.once("hand_detected", lambda e: print("first hand seen. Pinch your fingers!"))   # one-shot
hci.on("hand_lost", lambda e: print("hand lost"))

# head flick left/right -> browser back/forward (dry_run only logs, unless you pass --live)
hci.add_action(
    OSKeyboardAction({"head_flick_left": "alt+left", "head_flick_right": "alt+right"}, cooldown_s=1.0, dry_run=not live)
)

if __name__ == "__main__":
    print("Pinch your fingers or flick your head. 'q' in the window quits." + ("  [LIVE hotkeys]" if live else "  [dry-run hotkeys]"))
    hci.run(show_hud=True)
