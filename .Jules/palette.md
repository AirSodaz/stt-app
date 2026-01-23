# Palette's Journal

## 2025-02-18 - Accessible File Upload Pattern
**Learning:** The `<label>`-wrapping-input pattern for file uploads is hostile to keyboard users because labels aren't focusable by default. Adding `tabindex` to a label is non-standard.
**Action:** Always refactor file uploads to use a semantic `<button>` that triggers a hidden `<input type="file">` via a React `ref`.
