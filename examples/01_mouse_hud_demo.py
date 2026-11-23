"""Example 1: the touchless mouse, rebuilt on the SDK.

Run:  python examples/01_mouse_hud_demo.py

What you get (v0.2 desktop profile):
  * index fingertip moves the OS cursor (EMA smoothing 0.20, 1280x720 capture, every 2nd frame)
  * thumb + index pinch (< 45 px) = left-click hold / drag
  * thumb + MIDDLE finger pinch (< 45 px) = right click
  * hold an OPEN PALM for 1 s = pause cursor control; hold it again = resume
  * diagnostic HUD with colour-coded pinch state (yellow = right click, magenta = paused). Press 'q' to quit.
  * PyAutoGUI failsafe: throw the physical mouse into a screen corner to abort.

The v0.1 head-flick right click is OFF by default (it fired by accident). To get the old behaviour back use
HCIConfig.from_profile("prototype"), or opt in with config.face.flick_enabled = True.
"""
from open_ai_hci import HCIConfig, HCIEngine

# The defaults already ARE the prototype. They are spelled out here so you can see (and tweak) every knob.
config = HCIConfig()
config.camera.width, config.camera.height = 1280, 720
config.camera.frame_skip = 2                  # process every 2nd frame
config.smoothing.kind, config.smoothing.alpha = "ema", 0.20
config.hand.pinch_threshold_px = 45           # left click: thumb-index distance < 45 px
config.hand.right_pinch_enabled = True        # right click: thumb-MIDDLE distance < 45 px ...
config.hand.right_pinch_threshold_px = 45
config.hand.pause_gesture = "open_palm"       # hold an open palm (or "fist", or "none") to pause / resume ...
config.hand.pause_hold_s = 1.0                # ... for this long
config.face.flick_threshold_px = 28           # head flick (only used if you set config.face.flick_enabled = True):
config.face.flick_cooldown_s = 1.2            # |dx| > 28 and |dx| > 3 * |dy|, at most once every 1.2 s
config.mouse.failsafe = True

# For a steadier cursor, swap in a built-in profile instead:  HCIConfig.from_profile("precise")

if __name__ == "__main__":
    hci = HCIEngine(config)
    print("Pinch = click/drag, thumb + middle finger = right click, hold an open palm = pause/resume. Press 'q' to quit.")
    hci.run(show_hud=True)
