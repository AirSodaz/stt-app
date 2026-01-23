# Palette's Journal

## 2025-02-18 - Accessible File Upload Pattern
**Learning:** The `<label>`-wrapping-input pattern for file uploads is hostile to keyboard users because labels aren't focusable by default. Adding `tabindex` to a label is non-standard.
**Action:** Always refactor file uploads to use a semantic `<button>` that triggers a hidden `<input type="file">` via a React `ref`.

## 2025-05-21 - Focus Management for Conditional Views
**Learning:** When a view change completely unmounts the triggering button (like switching from Main to Settings), focus is lost to the body, disorienting keyboard users.
**Action:** Implement a focus management strategy (refs + useEffect) to capture focus on entry (e.g., Back button) and restore it to the trigger on exit.

## 2026-01-23 - Manual Focus Trap for Modals
**Learning:** Custom modals like `SaveOptionsModal` lack built-in focus containment, allowing keyboard users to tab out into the background content.
**Action:** Implement a `useEffect` with a `keydown` listener to manually trap focus (intercepting Tab) and handle Escape key for closing.
